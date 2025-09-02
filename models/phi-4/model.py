import os
import json
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from kserve import InferRequest, InferOutput
import base64
import io

from kserve_torch import BaseTorchModel  # type: ignore
import kserve
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = kserve.model_server.app


def get_phi4_model():
    if not hasattr(app.state, "phi4_model") or app.state.phi4_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app.state.phi4_model


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    phi4_model=Depends(get_phi4_model),
):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Convert Pydantic messages to dict format
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Process images if any message contains image data
        images = []
        processed_messages = []

        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                # Handle multimodal content
                text_content = ""
                for item in content:
                    if item.get("type") == "text":
                        text_content += item["text"]
                    elif item.get("type") == "image_url":
                        image_url = item["image_url"]["url"]
                        image = phi4_model.process_image_input(image_url)
                        images.append(image)

                processed_messages.append(
                    {"role": message["role"], "content": text_content}
                )
            else:
                processed_messages.append(message)

        # Update generation config with request parameters
        generation_config = phi4_model.generation_config.copy()
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["max_new_tokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p

        # Generate response
        response_text = phi4_model._generate_response_with_config(
            processed_messages, images if images else None, generation_config
        )

        # Format response in OpenAI format
        response = {
            "id": f"chatcmpl-{torch.randint(0, 1000000, (1,)).item()}",
            "object": "chat.completion",
            "created": int(
                torch.tensor(0).item()
            ),  # You might want to use actual timestamp
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # You could calculate this
                "completion_tokens": 0,  # You could calculate this
                "total_tokens": 0,
            },
        }

        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "chat_completion_error"}},
        )


class Phi4MultimodalModel(BaseTorchModel):
    def __init__(self, name: str):
        self.processor = None
        self.model = None
        self.generation_config = {}

        super().__init__(name)
        # Register this instance with the FastAPI app state
        app.state.phi4_model = self

    def load(self) -> None:
        """Load the Phi-4 Multimodal model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "microsoft/Phi-4-multimodal-instruct")
            self.logger.info(f"Initializing Phi-4 Multimodal model with ID: {model_id}")

            # Check if model directory exists in the mounted volume
            model_dir_exists = self.check_model_dir_exists()
            model_path = None

            if model_dir_exists:
                model_path = self.model_files_base_path
                self.logger.info(f"Found model directory at {model_path}")

            # Load the model from local path or download from HuggingFace
            if model_path and os.path.isdir(model_path):
                self.logger.info(f"Loading model from local path: {model_path}")
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map="auto",
                    ).to(self.device)
                    self.logger.info(f"Successfully loaded model from local path")
                except Exception as e:
                    self.logger.warning(f"Failed to load model from local path: {e}")
                    self.logger.info("Falling back to downloading from HuggingFace")
                    model_path = None

            # If local loading failed or path doesn't exist, download the model
            if not model_path or self.model is None or self.processor is None:
                self.logger.info(f"Loading model from HuggingFace Hub: {model_id}")
                self.processor = AutoProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            self.model_type = self.model.__class__.__name__
            self.logger.info(f"Loaded model type: {self.model_type}")

            # Set up generation config
            self.generation_config = {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": self.processor.tokenizer.eos_token_id,
            }

            self.ready = True
            self.logger.info("Phi-4 Multimodal model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """Format messages into the proper conversation format for Phi-4"""
        conversation = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "user":
                conversation.append(f"<|user|>\n{content}<|end|>")
            elif role == "assistant":
                conversation.append(f"<|assistant|>\n{content}<|end|>")
            elif role == "system":
                conversation.append(f"<|system|>\n{content}<|end|>")

        # Add assistant prompt for generation
        conversation.append("<|assistant|>\n")

        return "".join(conversation)

    def _generate_response(
        self, messages: List[Dict], images: Optional[List[Image.Image]] = None
    ):
        """Generate response using the Phi-4 model"""
        return self._generate_response_with_config(
            messages, images, self.generation_config
        )

    def _generate_response_with_config(
        self,
        messages: List[Dict],
        images: Optional[List[Image.Image]] = None,
        config: Dict = None,
    ):
        """Generate response using the Phi-4 model with custom config"""
        try:
            generation_config = config or self.generation_config

            # Format the conversation
            formatted_prompt = self._format_conversation(messages)

            # Prepare inputs
            if images:
                inputs = self.processor(
                    text=formatted_prompt, images=images, return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(text=formatted_prompt, return_tensors="pt").to(
                    self.device
                )

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_config)

            # Decode the response
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.processor.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            )

            return response.strip()

        except Exception as e:
            self.logger.error(f"Error during generation: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process Phi-4 Multimodal inference request"""
        try:
            inputs = request.inputs
            input_names = [inp.name for inp in inputs]
            self.logger.info(f"Request inputs: {input_names}")

            # Extract messages
            messages_input = next(
                (inp for inp in inputs if inp.name == "messages"), None
            )
            if messages_input is None:
                raise ValueError("'messages' input is required")

            messages_str = messages_input.data[0]
            if isinstance(messages_str, str):
                messages = json.loads(messages_str)
            else:
                messages = messages_str

            # Extract images if present
            images = []
            image_input = next((inp for inp in inputs if inp.name == "images"), None)
            if image_input:
                for image_data in image_input.data:
                    if image_data:  # Only process non-empty image data
                        image = self.process_image_input(image_data)
                        images.append(image)

            # Generate response
            response = self._generate_response(messages, images if images else None)

            # Prepare output
            result = {"response": response, "model": "phi-4-multimodal-instruct"}

            output_list = [
                InferOutput(
                    name="response",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(result)],
                ),
            ]

            return output_list, {}

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
            error_data = {"error": str(e)}
            output_list = [
                InferOutput(
                    name="response",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(error_data)],
                ),
            ]
            return output_list, {}


if __name__ == "__main__":
    BaseTorchModel.serve(
        model_class=Phi4MultimodalModel,
        description="Microsoft Phi-4 Multimodal Instruct Model Server",
        log_level="INFO",
    )

import os
import json
import time
from typing import Dict, List, Any, Tuple, Optional, Union
from PIL import Image
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from kserve import InferRequest, InferOutput
import base64
import io
import requests

from kserve_torch import BaseTorchModel  # type: ignore
import kserve
from fastapi import Depends, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Union

app = kserve.model_server.app


# -------------------------------
# FastAPI dependency to access the model instance
# -------------------------------


def get_fast_vlm_model():
    if not hasattr(app.state, "fast_vlm_model") or app.state.fast_vlm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app.state.fast_vlm_model


# -------------------------------
# OpenAI-compatible request schemas
# -------------------------------


class ImageUrl(BaseModel):
    url: str
    detail: Optional[str] = "auto"


class ContentItem(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[ImageUrl] = None


class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[ContentItem]]  # Support both string and multimodal content


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024
    top_p: Optional[float] = 0.9
    stream: Optional[bool] = False


# -------------------------------
# OpenAI-compatible endpoint
# -------------------------------


@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    fast_vlm_model=Depends(get_fast_vlm_model),
):
    """OpenAI-compatible chat completions endpoint"""
    try:
        # Convert Pydantic messages to dict format
        messages = [
            {"role": msg.role, "content": msg.content} for msg in request.messages
        ]

        # Process images if any message contains image data; also build text-only view
        images: List[Image.Image] = []
        processed_messages: List[Dict[str, Any]] = []

        for message in request.messages:
            role = message.role
            content = message.content

            # If multimodal: content is a list[ContentItem]
            if isinstance(content, list):
                text_parts: List[str] = []
                for item in content:
                    # Support both Pydantic objects and dicts (just in case)
                    if hasattr(item, "type"):
                        ttype = item.type
                        if ttype == "text":
                            text_parts.append(item.text or "")
                        elif (
                            ttype == "image_url"
                            and item.image_url
                            and item.image_url.url
                        ):
                            images.append(
                                fast_vlm_model.process_image_input(item.image_url.url)
                            )
                    else:
                        # dict-like fallback
                        ttype = item.get("type")
                        if ttype == "text":
                            text_parts.append(item.get("text") or "")
                        elif ttype == "image_url":
                            url = (item.get("image_url") or {}).get("url")
                            if url:
                                images.append(fast_vlm_model.process_image_input(url))
                processed_messages.append(
                    {"role": role, "content": "".join(text_parts)}
                )
            else:
                # Plain string content
                processed_messages.append({"role": role, "content": str(content)})

        # Update generation config with request parameters
        generation_config = fast_vlm_model.generation_config.copy()
        if request.temperature is not None:
            generation_config["temperature"] = request.temperature
        if request.max_tokens is not None:
            generation_config["max_new_tokens"] = request.max_tokens
        if request.top_p is not None:
            generation_config["top_p"] = request.top_p

        # Generate response
        response_text = fast_vlm_model._generate_response_with_config(
            processed_messages, images if images else None, generation_config
        )

        # Format response in OpenAI format
        response = {
            "id": f"chatcmpl-{torch.randint(0, 1000000, (1,)).item()}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 0,  # (Optional) Populate with a tokenizer if desired
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }

        return response

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "chat_completion_error"}},
        )


# -------------------------------
# Model implementation
# -------------------------------


class FastVLMModel(BaseTorchModel):
    def __init__(self, name: str):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.processor: Optional[AutoProcessor] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.generation_config: Dict[str, Any] = {}
        super().__init__(name)
        # Register this instance with the FastAPI app state
        app.state.fast_vlm_model = self

        self.IMAGE_TOKEN_INDEX = -200

    # ---------------------------
    # Load
    # ---------------------------
    def load(self) -> None:
        """Load the Fast-VLM model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "apple/FastVLM-0.5B")
            self.logger.info(f"Initializing Fast-VLM model with ID: {model_id}")

            # (Optional) Load from a mounted directory if present
            model_dir_exists = self.check_model_dir_exists()
            model_path: Optional[str] = (
                self.model_files_base_path if model_dir_exists else None
            )

            def _load_from_path(path: str):
                self.logger.info(f"Loading model from local path: {path}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    path, trust_remote_code=True
                )
                # Try to load a processor if available
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        path, trust_remote_code=True
                    )
                except Exception as e:
                    self.logger.warning(f"No AutoProcessor at local path: {e}")
                    self.processor = None
                self.model = AutoModelForCausalLM.from_pretrained(
                    path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            def _load_from_hub(hub_id: str):
                self.logger.info(f"Loading model from HuggingFace Hub: {hub_id}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    hub_id, trust_remote_code=True
                )
                try:
                    self.processor = AutoProcessor.from_pretrained(
                        hub_id, trust_remote_code=True
                    )
                except Exception as e:
                    self.logger.warning(f"No AutoProcessor on hub: {e}")
                    self.processor = None
                self.model = AutoModelForCausalLM.from_pretrained(
                    hub_id,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

            if model_path and os.path.isdir(model_path):
                try:
                    _load_from_path(model_path)
                except Exception as e:
                    self.logger.warning(f"Failed local load: {e}; falling back to hub")
                    _load_from_hub(model_id)
            else:
                _load_from_hub(model_id)

            assert self.model is not None and self.tokenizer is not None
            self.model_type = self.model.__class__.__name__
            self.logger.info(f"Loaded model type: {self.model_type}")

            # Set up generation config
            self.generation_config = {
                "max_new_tokens": 1024,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "pad_token_id": self.tokenizer.eos_token_id,
            }

            self.ready = True
            self.logger.info("Fast-VLM model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    # ---------------------------
    # Image helpers
    # ---------------------------
    def process_image_input(self, image_input: str) -> Image.Image:
        """Process image input from base64 string, URL, or file path"""
        try:
            if image_input.startswith("data:image/"):
                # Handle base64 encoded image
                header, data = image_input.split(",", 1)
                image_data = base64.b64decode(data)
                return Image.open(io.BytesIO(image_data)).convert("RGB")
            elif image_input.startswith("http"):
                # Handle URL - download the image
                self.logger.info(f"Downloading image from URL: {image_input}")
                response = requests.get(image_input, timeout=30)
                response.raise_for_status()
                image_data = response.content
                return Image.open(io.BytesIO(image_data)).convert("RGB")
            else:
                # Handle file path or direct base64
                try:
                    image_data = base64.b64decode(image_input)
                    return Image.open(io.BytesIO(image_data)).convert("RGB")
                except Exception:
                    # Try as file path
                    return Image.open(image_input).convert("RGB")
        except Exception as e:
            self.logger.error(f"Error processing image: {e}")
            raise

    # ---------------------------
    # Prompt helpers
    # ---------------------------
    def _format_conversation(
        self, messages: List[Dict[str, Any]], has_images: bool = False
    ) -> str:
        """
        Build the chat prompt string. For FastVLM, when an image is present
        we must place a literal '<image>' token in the user content.
        """
        assert self.tokenizer is not None

        normalized: List[Dict[str, str]] = []
        for m in messages:
            role = m.get("role", "user")
            content = str(m.get("content", ""))

            # If images exist and the user text doesn't already include <image>, prepend it
            if has_images and role == "user" and "<image>" not in content:
                content = "<image>\n" + content

            normalized.append({"role": role, "content": content})

        if hasattr(self.tokenizer, "apply_chat_template"):
            rendered = self.tokenizer.apply_chat_template(
                normalized, tokenize=False, add_generation_prompt=True
            )
        else:
            convo = []
            for m in normalized:
                r, c = m["role"], m["content"]
                tag = (
                    "system"
                    if r == "system"
                    else ("assistant" if r == "assistant" else "user")
                )
                convo.append(f"<|{tag}|>\n{c}<|end|>")
            convo.append("<|assistant|>\n")
            rendered = "".join(convo)

        # SAFEGUARD: if somehow the template removed '<image>', put it back at the top.
        if has_images and "<image>" not in rendered:
            rendered = "<image>\n" + rendered

        return rendered

    def _tokenize_with_images(
        self, rendered_prompt: str, images: Optional[List[Image.Image]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        FastVLM requires splicing IMAGE_TOKEN_INDEX at the literal '<image>' position.
        For multimodal: split on '<image>', tokenize pre/post (no specials),
        concat pre_ids + [[IMAGE_TOKEN_INDEX]] + post_ids, build attention_mask,
        and preprocess the image via the model's vision tower.
        """
        assert self.tokenizer is not None and self.model is not None

        if images and len(images) > 0 and "<image>" in rendered_prompt:
            pre, post = rendered_prompt.split("<image>", 1)

            pre_ids = self.tokenizer(
                pre, return_tensors="pt", add_special_tokens=False
            ).input_ids
            post_ids = self.tokenizer(
                post, return_tensors="pt", add_special_tokens=False
            ).input_ids

            img_tok = torch.tensor([[self.IMAGE_TOKEN_INDEX]], dtype=pre_ids.dtype)
            input_ids = torch.cat([pre_ids, img_tok, post_ids], dim=1).to(
                self.model.device
            )
            attention_mask = torch.ones_like(input_ids, device=self.model.device)

            # Preprocess the image
            vtower = self.model.get_vision_tower()
            px = vtower.image_processor(images=images[0], return_tensors="pt")[
                "pixel_values"
            ]
            px = px.to(self.model.device, dtype=self.model.dtype)

            return {"inputs": input_ids, "attention_mask": attention_mask, "images": px}

        # Text-only path
        enc = self.tokenizer(rendered_prompt, return_tensors="pt")
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # HF accepts 'inputs' synonymously; normalize for consistency
        if "inputs" not in enc and "input_ids" in enc:
            enc["inputs"] = enc["input_ids"]
        return enc

    # ---------------------------
    # Generation wrappers
    # ---------------------------
    def _generate_response(
        self, messages: List[Dict], images: Optional[List[Image.Image]] = None
    ) -> str:
        return self._generate_response_with_config(
            messages, images, self.generation_config
        )

    def _generate_response_with_config(
        self,
        messages: List[Dict],
        images: Optional[List[Image.Image]] = None,
        config: Optional[Dict] = None,
    ) -> str:
        try:
            assert self.model is not None and self.tokenizer is not None
            generation_config = config or self.generation_config

            rendered = self._format_conversation(messages, has_images=bool(images))
            enc = self._tokenize_with_images(rendered, images)

            # Sanity checks
            if "inputs" not in enc or enc["inputs"] is None:
                return "Error: tokenizer/processor did not produce inputs / input_ids."

            with torch.no_grad():
                out = self.model.generate(
                    inputs=enc["inputs"],
                    attention_mask=enc.get("attention_mask", None),
                    images=enc.get("images", None),
                    **generation_config,
                )

            text = self.tokenizer.decode(out[0], skip_special_tokens=True).strip()
            return text
        except Exception as e:
            self.logger.error(f"Error during generation: {e}", exc_info=True)
            return f"Error generating response: {str(e)}"

    # ---------------------------
    # KServe entrypoint
    # ---------------------------
    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process Multimodal inference request"""
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
            images: List[Image.Image] = []
            image_input = next((inp for inp in inputs if inp.name == "images"), None)
            if image_input:
                for image_data in image_input.data:
                    if image_data:  # Only process non-empty image data
                        image = self.process_image_input(image_data)
                        images.append(image)

            # Generate response
            response = self._generate_response(messages, images if images else None)

            # Prepare output
            result = {"response": response, "model": "fast-vlm"}

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
        model_class=FastVLMModel,
        description="Apple Fast-VLM Model Server",
        log_level="INFO",
    )

import os
import json
from typing import Dict, List, Any, Tuple
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from kserve import InferRequest, InferOutput
from pydantic import BaseModel
import kserve
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import io
import base64
import hashlib
from fastapi import Depends, HTTPException

from kserve_torch import BaseTorchModel  # type: ignore

app = kserve.model_server.app


def get_florence_model():
    if not hasattr(app.state, "florence_model") or app.state.florence_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app.state.florence_model


class GenerateRequest(BaseModel):
    text: str
    image: str


@app.post("v3/generate")
async def generate(request: GenerateRequest, model=Depends(get_florence_model)):
    try:
        result = await model.process_standard_request(request)
        return {"result": result, "status": "success"}
    except Exception as e:
        return {"error": str(e), "status": "error"}


class FlorenceModel(BaseTorchModel):
    def __init__(self, name: str):
        self.processor = None
        self.model = None
        self.tasks = {}

        super().__init__(name)
        app.state.florence_model = self

    def load(self) -> None:
        """Load the Florence model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "microsoft/Florence-2-large")
            self.logger.info(f"Initializing Florence model with ID: {model_id}")

            # Check if model directory exists in the mounted volume
            model_dir_exists = self.check_model_dir_exists()
            model_path = None

            if model_dir_exists:
                # Use the local path if it exists
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
                        model_path, trust_remote_code=True
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
                    model_id, trust_remote_code=True
                ).to(self.device)

            self.model_type = self.model.__class__.__name__
            self.logger.info(f"Loaded model type: {self.model_type}")

            self.ready = True
            self.logger.info(
                f"Florence model loaded successfully. Supported tasks: {list(self.tasks.keys())}"
            )
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def create_visualization(self, image, bboxes, labels):
        """Create visualization with bounding boxes overlaid on the image"""

        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(np.array(image))

        def get_color_for_label(label_text):
            label_hash = int(hashlib.md5(label_text.lower().encode()).hexdigest(), 16)

            distinct_colors = [
                "red",
                "blue",
                "green",
                "purple",
                "orange",
                "cyan",
                "magenta",
                "lime",
                "pink",
                "teal",
                "lavender",
                "brown",
                "olive",
                "navy",
                "maroon",
                "gold",
            ]

            color_index = label_hash % len(distinct_colors)
            return distinct_colors[color_index]

        for bbox, label in zip(bboxes, labels):
            x, y, x2, y2 = bbox
            width = x2 - x
            height = y2 - y

            color = get_color_for_label(label)

            rect = patches.Rectangle(
                (x, y), width, height, linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)
            plt.text(x, y - 5, label, color=color, fontsize=12, weight="bold")

        plt.axis("off")

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)

        img_str = base64.b64encode(buf.read()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def _run_visual_tasks(self, image: Image.Image, task: str):
        """Perform visual tasks."""
        self.logger.info(f"Using prompt: {task}")

        inputs = self.processor(text=task, images=image, return_tensors="pt").to(
            self.device
        )

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"].cuda(),
                    pixel_values=inputs["pixel_values"].cuda(),
                    max_new_tokens=1024,
                    min_new_tokens=5,
                    do_sample=False,
                    num_beams=3,
                    early_stopping=False,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                )

            text_generations = self.processor.batch_decode(
                outputs, skip_special_tokens=False
            )[0]
            parsed_answer = self.processor.post_process_generation(
                text_generations,
                task=task,
                image_size=(image.width, image.height),
            )

            self.logger.info(f"Parsed answer type: {type(parsed_answer)}")

            return parsed_answer

        except Exception as e:
            self.logger.error(f"Error during task handling: {e}", exc_info=True)
            return {"error": f"Error generating answer: {str(e)}"}

    async def process_standard_request(
        self, request: GenerateRequest
    ) -> Dict[str, Any]:
        if not request.text or not request.image:
            raise ValueError("Both 'text' and 'image' fields are required.")

        image = self.process_image_input(request.image)

        task = request.text

        result = self._run_visual_tasks(image, task)

        cleaned_result = None
        if isinstance(result, dict):
            # For simple tasks like <CAPTION> that return string values
            if task in result:
                cleaned_result = result[task]
            # For complex tasks that return structured data (like bounding boxes)
            elif task.strip() in result:
                cleaned_result = result[task.strip()]

        if cleaned_result is None:
            cleaned_result = result

        self.logger.info(f"Cleaned result: {cleaned_result}")

        overlay_image = None
        if (
            isinstance(cleaned_result, dict)
            and "bboxes" in cleaned_result
            and "labels" in cleaned_result
        ):
            bboxes = cleaned_result["bboxes"]
            labels = cleaned_result["labels"]
            if bboxes and labels and len(bboxes) == len(labels):
                overlay_image = self.create_visualization(image, bboxes, labels)

        if overlay_image:
            cleaned_result["overlay.png"] = overlay_image

        return cleaned_result

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process Florence model inference request"""
        try:
            inputs = request.inputs

            input_names = [inp.name for inp in inputs]
            self.logger.info(f"Request inputs: {input_names}")

            image_input = next((inp for inp in inputs if inp.name == "image"), None)
            if image_input is None:
                raise ValueError(f"Task requires an 'image' input")

            image_data = image_input.data[0]
            self.logger.info(
                f"Processing image input (type: {'URL' if image_data.startswith(('http://', 'https://')) else 'base64'})"
            )
            image = self.process_image_input(image_data)

            task_input = next((inp for inp in inputs if inp.name == "text"), None)
            if task_input is None:
                raise ValueError(f"Task input is required")
            task = task_input.data[0]

            complete_result = self._run_visual_tasks(image, task)

            cleaned_result = None
            if isinstance(complete_result, dict):
                # For simple tasks like <CAPTION> that return string values
                if task in complete_result:
                    cleaned_result = complete_result[task]
                # For complex tasks that return structured data (like bounding boxes)
                elif task.strip() in complete_result:
                    cleaned_result = complete_result[task.strip()]

            if cleaned_result is None:
                cleaned_result = complete_result

            self.logger.info(f"Cleaned result: {cleaned_result}")

            overlay_image = None
            if (
                isinstance(cleaned_result, dict)
                and "bboxes" in cleaned_result
                and "labels" in cleaned_result
            ):
                bboxes = cleaned_result["bboxes"]
                labels = cleaned_result["labels"]
                if bboxes and labels and len(bboxes) == len(labels):
                    overlay_image = self.create_visualization(image, bboxes, labels)

            if overlay_image:
                cleaned_result["overlay.png"] = overlay_image

            output_list = [
                InferOutput(
                    name="output",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(cleaned_result)],
                ),
            ]

            return output_list, {}

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
            error_data = {"error": str(e), "task": "error"}
            output_list = [
                InferOutput(
                    name="output",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(error_data)],
                ),
                InferOutput(
                    name="task",
                    datatype="BYTES",
                    shape=[1],
                    data=["error"],
                ),
            ]
            return output_list, {}


if __name__ == "__main__":
    BaseTorchModel.serve(
        model_class=FlorenceModel,
        description="Microsoft Florence Vision-Language Model Server",
        log_level="INFO",
    )

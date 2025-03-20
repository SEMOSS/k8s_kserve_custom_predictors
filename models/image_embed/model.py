import os
import json
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from transformers import AutoImageProcessor, AutoModel
from kserve import InferRequest, InferOutput

from kserve_torch import BaseTorchModel  # type: ignore


class ImageEmbeddingModel(BaseTorchModel):
    """
    KServe predictor for image embedding models like facebook/dinov2-large.

    This class can handle any model that takes an image as input and outputs embeddings,
    making it suitable for various image embedding models from HuggingFace.
    """

    def __init__(self, name: str):
        self.processor = None
        self.model = None
        self.embedding_dim = None
        self.pooling_strategy = os.environ.get("POOLING_STRATEGY", "cls")

        super().__init__(name)

    def load(self) -> None:
        """Load the image embedding model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "facebook/dinov2-large")
            self.logger.info(f"Initializing image embedding model with ID: {model_id}")

            model_dir_exists = self.check_model_dir_exists()
            model_path = None

            if model_dir_exists:
                model_path = self.model_files_base_path
                self.logger.info(f"Found model directory at {model_path}")

            if model_path and os.path.isdir(model_path):
                self.logger.info(f"Loading model from local path: {model_path}")
                try:
                    self.processor = AutoImageProcessor.from_pretrained(
                        model_path, trust_remote_code=True
                    )
                    self.model = AutoModel.from_pretrained(
                        model_path, trust_remote_code=True
                    ).to(self.device)
                    self.logger.info(f"Successfully loaded model from local path")
                except Exception as e:
                    self.logger.warning(f"Failed to load model from local path: {e}")
                    self.logger.info("Falling back to downloading from HuggingFace")
                    model_path = None

            if not model_path or self.model is None or self.processor is None:
                self.logger.info(f"Loading model from HuggingFace Hub: {model_id}")
                self.processor = AutoImageProcessor.from_pretrained(
                    model_id, trust_remote_code=True
                )
                self.model = AutoModel.from_pretrained(
                    model_id, trust_remote_code=True
                ).to(self.device)

            self.model_type = self.model.__class__.__name__

            with torch.no_grad():
                dummy_image = Image.new("RGB", (224, 224), color="white")
                inputs = self.processor(images=dummy_image, return_tensors="pt").to(
                    self.device
                )
                outputs = self.model(**inputs)

                if hasattr(outputs, "pooler_output"):
                    self.embedding_dim = outputs.pooler_output.shape[-1]
                elif hasattr(outputs, "last_hidden_state"):
                    self.embedding_dim = outputs.last_hidden_state.shape[-1]
                else:
                    for key, value in outputs.items():
                        if isinstance(value, torch.Tensor):
                            self.embedding_dim = value.shape[-1]
                            break

            self.logger.info(
                f"Loaded model type: {self.model_type} with embedding dimension: {self.embedding_dim}"
            )
            self.logger.info(f"Using pooling strategy: {self.pooling_strategy}")

            self.ready = True
            self.logger.info(f"Image embedding model loaded successfully.")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def _get_embedding(
        self, image: Image.Image, pooling_strategy: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract embeddings from the image.

        Args:
            image: PIL Image to get embeddings for
            pooling_strategy: Optional override for the pooling strategy
                - 'cls': Use CLS token (first token)
                - 'mean': Mean of all tokens
                - 'max': Max pooling of all tokens

        Returns:
            numpy.ndarray containing the embeddings
        """
        pooling = pooling_strategy or self.pooling_strategy
        self.logger.info(f"Generating embeddings with pooling strategy: {pooling}")

        inputs = self.processor(images=image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if hasattr(outputs, "pooler_output"):
            embeddings = outputs.pooler_output
        elif hasattr(outputs, "last_hidden_state"):
            hidden_states = outputs.last_hidden_state

            if pooling == "cls" and hidden_states.shape[1] > 1:
                embeddings = hidden_states[:, 0]
            elif pooling == "mean":
                embeddings = torch.mean(hidden_states, dim=1)
            elif pooling == "max":
                embeddings = torch.max(hidden_states, dim=1)[0]
            else:
                embeddings = hidden_states[:, 0]
        else:
            for key, value in outputs.items():
                if isinstance(value, torch.Tensor):
                    if value.dim() > 2:
                        if pooling == "mean":
                            embeddings = torch.mean(value, dim=1)
                        elif pooling == "max":
                            embeddings = torch.max(value, dim=1)[0]
                        else:
                            embeddings = value[:, 0]  # Default to first token
                    else:
                        embeddings = value
                    break

        embeddings_np = embeddings.cpu().numpy()

        if os.environ.get("NORMALIZE_EMBEDDINGS", "false").lower() == "true":
            embeddings_np = embeddings_np / np.linalg.norm(
                embeddings_np, axis=1, keepdims=True
            )

        return embeddings_np

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process image embedding model inference request"""
        try:
            inputs = request.inputs
            input_names = [inp.name for inp in inputs]
            self.logger.info(f"Request inputs: {input_names}")

            image_input = next((inp for inp in inputs if inp.name == "image"), None)
            if image_input is None:
                raise ValueError("Request requires an 'image' input")

            embeddings_list = []
            for img_data in image_input.data:
                self.logger.info(
                    f"Processing image input (type: {'URL' if img_data.startswith(('http://', 'https://')) else 'base64'})"
                )
                image = self.process_image_input(img_data)

                pooling_strategy = None
                pooling_input = next(
                    (inp for inp in inputs if inp.name == "pooling_strategy"), None
                )
                if pooling_input and pooling_input.data:
                    pooling_strategy = pooling_input.data[0]

                embeddings = self._get_embedding(image, pooling_strategy)
                embeddings_list.append(embeddings)

            if len(embeddings_list) == 1:
                final_embeddings = embeddings_list[0]
            else:
                final_embeddings = np.vstack(embeddings_list)

            result = {
                "embedding_dim": self.embedding_dim,
                "embeddings": final_embeddings.tolist(),
                "model_id": os.environ.get("MODEL_ID", "facebook/dinov2-large"),
                "shape": final_embeddings.shape,
            }

            include_raw = False
            raw_input = next((inp for inp in inputs if inp.name == "include_raw"), None)
            if raw_input and raw_input.data and raw_input.data[0].lower() == "true":
                include_raw = True

            output_list = [
                InferOutput(
                    name="output",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(result)],
                ),
            ]

            if include_raw:
                raw_bytes = final_embeddings.tobytes()
                output_list.append(
                    InferOutput(
                        name="raw_embeddings",
                        datatype="BYTES",
                        shape=list(final_embeddings.shape),
                        data=[raw_bytes],
                    )
                )

            self.logger.info(
                f"Returning embeddings with shape: {final_embeddings.shape}"
            )
            return output_list, {}

        except Exception as e:
            self.logger.error(f"Error processing request: {str(e)}", exc_info=True)
            error_data = {"error": str(e)}
            output_list = [
                InferOutput(
                    name="output",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(error_data)],
                ),
            ]
            return output_list, {}


if __name__ == "__main__":
    BaseTorchModel.serve(
        model_class=ImageEmbeddingModel,
        description="Image Embedding Model Server (supports models like DinoV2, CLIP, etc.)",
        log_level="INFO",
    )

import abc
import logging
import uuid
import argparse
import os
import re
from typing import Dict, List, Any, Tuple, Type
import base64
from io import BytesIO
from PIL import Image
import requests
import torch
import kserve
from kserve import InferRequest, InferResponse, Model, InferOutput, ModelServer
from kserve import logging as kserve_logging
from fastapi.middleware.cors import CORSMiddleware


class BaseTorchModel(Model, abc.ABC):
    """
    Abstract base class for PyTorch-based KServe models.

    This class provides a common interface and shared functionality for all
    PyTorch model implementations.
    """

    def __init__(self, name: str):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.ready = False
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"Using device: {self.device}")

        self.model_files_base_path = os.environ.get("MODEL_FILES_PATH", "/model-files")
        self.logger.info(f"Model files base path: {self.model_files_base_path}")

        self.load()

    @staticmethod
    def get_safe_model_path(model_id: str) -> str:
        """
        Convert a model ID to a path-safe directory name.

        Args:
            model_id: The full model ID (e.g., "microsoft/Florence-2-large")

        Returns:
            A path-safe directory name for the model
        """
        short_name = model_id.split("/")[-1] if "/" in model_id else model_id

        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "-", short_name).lower()

        return safe_name

    def get_model_dir(self, model_id: str) -> str:
        """
        Get the full path for model directory in the mounted volume.

        Args:
            model_id: The full model ID (e.g., "microsoft/Florence-2-large")

        Returns:
            The full path to the model directory
        """
        safe_name = self.get_safe_model_path(model_id)
        return os.path.join(self.model_files_base_path, safe_name)

    def check_model_dir_exists(self) -> bool:
        """
        Check if the model directory exists in the mounted volume.

        Args:
            model_id: The model ID to check

        Returns:
            True if the directory exists, False otherwise
        """
        # For local development with volumes
        if self.model_files_base_path == "/model-files":
            self.logger.warning(
                "MODEL_FILES_PATH environment variable is not set. Defaulting to /model-files"
            )
            model_id = os.environ.get("MODEL_ID")
            self.model_files_base_path = self.get_model_dir(model_id)

        exists = os.path.isdir(self.model_files_base_path)
        self.logger.info(
            f"Checking if model directory exists at {self.model_files_base_path}: {exists}"
        )
        return exists

    @abc.abstractmethod
    def load(self) -> None:
        """
        Load the model from a source (e.g., HuggingFace Hub, local file, etc.).

        This method must be implemented by subclasses to initialize self.model.
        Upon successful loading, set self.ready = True.
        """
        pass

    @abc.abstractmethod
    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """
        Process the inference request and return the processed outputs.

        Args:
            request: The KServe InferRequest object

        Returns:
            Tuple containing:
            - List of InferOutput objects
            - Optional metadata or additional context as a dictionary
        """
        pass

    def decode_base64_image(self, base64_string: str) -> Image.Image:
        """Decode a base64 string into a PIL Image."""
        try:
            if "base64," in base64_string:
                base64_string = base64_string.split("base64,")[1]

            img_data = base64.b64decode(base64_string)
            img = Image.open(BytesIO(img_data))
            return img
        except Exception as e:
            self.logger.error(f"Error decoding base64 image: {e}")
            raise ValueError(f"Failed to decode base64 image: {e}")

    def process_image_input(self, image_data: str) -> Image.Image:
        """
        Process image input which can be either a URL or base64-encoded data.
        Args:
            image_data: Either a URL (starting with http) or base64-encoded image data

        Returns:
            PIL Image object
        """
        if image_data.startswith(("http://", "https://")):
            self.logger.info(f"Processing image from URL: {image_data[:50]}...")
            try:
                response = requests.get(image_data, timeout=10)
                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                img = Image.open(BytesIO(response.content))
                return img
            except Exception as e:
                self.logger.error(f"Error fetching image from URL: {e}")
                raise ValueError(f"Failed to fetch image from URL: {e}")
        else:
            return self.decode_base64_image(image_data)

    async def predict(
        self, request, headers=None, response_headers=None
    ) -> InferResponse:
        """
        Handle prediction request and return formatted response.

        This implementation provides error handling and common response formatting.
        Specific processing is delegated to the process_request method.
        """
        try:
            if not self.ready:
                self.logger.warning("Model is not ready yet")
                return InferResponse(
                    response_id=str(uuid.uuid4()),
                    model_name=self.name,
                    infer_outputs=[],
                )

            self.logger.debug("Processing inference request")

            if not isinstance(request, InferRequest):
                request = InferRequest.from_json(request)

            # Process request through the model-specific implementation
            output_list, metadata = await self.process_request(request)

            if response_headers is not None:
                response_headers["Content-Type"] = "application/json"
                if metadata and "headers" in metadata:
                    for key, value in metadata["headers"].items():
                        response_headers[key] = value

            self.logger.info("Returning inference response")
            return InferResponse(
                response_id=str(uuid.uuid4()),
                model_name=self.name,
                infer_outputs=output_list,
            )

        except Exception as e:
            self.logger.error(f"Prediction error: {str(e)}", exc_info=True)
            return InferResponse(
                response_id=str(uuid.uuid4()),
                model_name=self.name,
                infer_outputs=[],
            )

    @staticmethod
    def serve(
        model_class: Type["BaseTorchModel"],
        description: str = "KServe PyTorch Model Server",
        log_level: str = "INFO",
    ) -> None:
        """
        Set up and start a KServe model server for the given model class.

        Args:
            model_class: The BaseTorchModel subclass to serve
            description: Description of the model server
            log_level: Logging level
        """
        parser = argparse.ArgumentParser(
            parents=[kserve.model_server.parser], description=description
        )
        args, _ = parser.parse_known_args()

        numeric_level = getattr(logging, log_level.upper(), None)
        if not isinstance(numeric_level, int):
            raise ValueError(f"Invalid log level: {log_level}")

        logging.basicConfig(
            level=numeric_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()],
        )

        logger = logging.getLogger(__name__)
        logger.info(f"Starting {model_class.__name__} server")

        if args.configure_logging:
            logger.info(
                f"Configuring KServe logging with config file: {args.log_config_file}"
            )
            kserve_logging.configure_logging(args.log_config_file)

        # Initialize the model
        model = model_class(args.model_name)

        app = kserve.model_server.app
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        logger.info(f"Starting ModelServer with {model_class.__name__}")
        ModelServer().start([model])

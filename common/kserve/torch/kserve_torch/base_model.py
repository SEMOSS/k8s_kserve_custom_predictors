import abc
import logging
import uuid
import argparse
from typing import Dict, List, Any, Tuple, Type

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
        self.load()

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

import argparse
import os
import random
import string
import json
from typing import Dict, List, Any
import uuid
import logging

from fastapi.middleware.cors import CORSMiddleware
import kserve
from kserve import InferRequest, InferResponse, Model, ModelServer, InferOutput
from kserve import logging as kserve_logging
from gliner import GLiNER
import torch


class GLiNERModel(Model):
    def __init__(self, name: str):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.ready = False
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Using device: {self.device}")
        self.load()

    def load(self):
        """Load the model from HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "urchade/gliner_multi-v2.1")
            self.logger.info(f"Loading GLiNER model from {model_id}")

            self.model = GLiNER.from_pretrained(model_id, device=self.device)
            self.ready = True
            self.logger.info("GLiNER model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise

    def _generate_mask(self, length: int = 6) -> str:
        """Generate a random mask string."""
        random_str = "".join(
            random.choices(string.ascii_uppercase + string.digits, k=length)
        )
        return f"m_{random_str}"

    def _mask_entities(
        self, text: str, entities: List[Dict[str, Any]], mask_entities: List[str]
    ) -> Dict[str, Any]:
        """
        Mask entities in the text based on the mask_entities list.

        Args:
            text (str): Original text
            entities (List[Dict[str, Any]]): Detected entities
            mask_entities (List[str]): List of entity types to mask

        Returns:
            Dict[str, Any]: Dictionary containing masked text and mapping
        """
        entities = sorted(entities, key=lambda x: x["start"], reverse=True)

        mask_values = {}
        new_text = text

        for entity in entities:
            if entity["label"] in mask_entities:
                orig_text = entity["text"]
                start = entity["start"]
                end = entity["end"]

                if orig_text in mask_values:
                    mask_text = mask_values[orig_text]
                else:
                    mask_text = self._generate_mask()
                    mask_values[orig_text] = mask_text
                    mask_values[mask_text] = orig_text

                new_text = new_text[:start] + mask_text + new_text[end:]

        return {"masked_text": new_text, "mask_values": mask_values}

    async def predict(
        self, request, headers=None, response_headers=None
    ) -> InferResponse:
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

            inputs = request.inputs
            text_input = next(inp for inp in inputs if inp.name == "text")
            text = text_input.data[0]

            entities_input = next(inp for inp in inputs if inp.name == "labels")
            entities_schema = entities_input.data

            mask_entities_input = next(
                (inp for inp in inputs if inp.name == "mask_entities"), None
            )
            mask_entities = mask_entities_input.data if mask_entities_input else []

            self.logger.info("Predicting entities with GLiNER model")
            entities = self.model.predict_entities(text, entities_schema)
            self.logger.debug(f"Found {len(entities)} entities")

            masked_output = (
                self._mask_entities(text, entities, mask_entities)
                if mask_entities
                else {"masked_text": text, "mask_values": {}}
            )

            if mask_entities:
                self.logger.info(f"Masked {len(masked_output['mask_values'])} entities")

            output_list = [
                InferOutput(
                    name="output",
                    datatype="BYTES",
                    shape=[1],
                    data=[masked_output["masked_text"]],
                ),
                InferOutput(
                    name="raw_output",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(entities)],
                ),
                InferOutput(
                    name="mask_values",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(masked_output["mask_values"])],
                ),
                InferOutput(
                    name="input",
                    datatype="BYTES",
                    shape=[1],
                    data=[text],
                ),
                InferOutput(
                    name="entities",
                    datatype="BYTES",
                    shape=[len(entities_schema)],
                    data=entities_schema,
                ),
            ]

            if response_headers is not None:
                response_headers["Content-Type"] = "application/json"

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


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info("Starting GLiNER model server")

    if args.configure_logging:
        logger.info(
            f"Configuring KServe logging with config file: {args.log_config_file}"
        )
        kserve_logging.configure_logging(args.log_config_file)

    model = GLiNERModel(args.model_name)

    app = kserve.model_server.app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    logger.info(f"Starting ModelServer with model: {args.model_name}")
    ModelServer().start([model])

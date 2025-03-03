import argparse
import os
import logging
import random
import string
from typing import Dict, List, Any

from fastapi.middleware.cors import CORSMiddleware
import kserve
from kserve import Model, ModelServer
from gliner import GLiNER
import torch


class GLiNERModel(Model):
    def __init__(self, name: str):
        super().__init__(name, return_response_headers=True)
        self.name = name
        self.ready = False
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {self.device}")
        self.load()

    def load(self):
        """Load the model from HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "urchade/gliner_multi-v2.1")
            logging.info(f"Loading GLiNER model from {model_id}")

            self.model = GLiNER.from_pretrained(model_id, device=self.device)
            self.ready = True
            logging.info("GLiNER model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading model: {e}")
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
            if entity["type"] in mask_entities:
                orig_text = text[entity["start"] : entity["end"]]
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
        self,
        payload: Dict,
        headers: Dict[str, str] = None,
        response_headers: Dict[str, str] = None,
    ) -> Dict:
        """
        Predict method for GLiNER model

        Expected payload format:
        {
            "text": "sample text to process",
            "entities" or "labels": ["PERSON", "ORGANIZATION", ...],  # Entity types to extract
            "mask_entities": ["PERSON", ...] # Optional, entities to mask
        }
        """
        try:
            if not self.ready:
                return {"status": "error", "message": "Model is not ready yet"}

            # Debug logging
            logging.info(f"Received payload type: {type(payload)}")
            if isinstance(payload, bytes):
                logging.info(f"Payload is bytes, trying to decode and parse as JSON")
                try:
                    import json

                    payload = json.loads(payload.decode("utf-8"))
                except Exception as e:
                    logging.error(f"Failed to decode bytes payload: {e}")
                    return {
                        "status": "error",
                        "message": f"Failed to decode payload: {str(e)}",
                    }

            logging.info(f"Processing payload: {payload}")

            text = payload.get("text", "")
            entities_schema = payload.get("entities") or payload.get("labels")
            mask_entities = payload.get("mask_entities", [])

            if not text:
                return {
                    "status": "error",
                    "message": "No text provided",
                    "output": [],
                    "raw_output": [],
                    "mask_values": {},
                    "input": "",
                    "entities": [],
                }

            try:
                entities = (
                    self.model.predict_entities(text, entities_schema)
                    if hasattr(self.model, "predict_entities")
                    else self.model(text, schema=entities_schema)
                )

                normalized_entities = []
                for entity in entities:
                    normalized = dict(entity)
                    if "label" in normalized and "type" not in normalized:
                        normalized["type"] = normalized["label"]
                    normalized_entities.append(normalized)

                entities = normalized_entities

            except Exception as e:
                logging.error(f"Error during prediction: {e}")
                return {
                    "status": "error",
                    "message": str(e),
                    "output": [],
                    "raw_output": [],
                    "mask_values": {},
                    "input": text,
                    "entities": entities_schema or [],
                }

            masked_output = (
                self._mask_entities(text, entities, mask_entities)
                if mask_entities
                else {"masked_text": text, "mask_values": {}}
            )

            return {
                "status": "success",
                "output": masked_output["masked_text"],
                "raw_output": entities,
                "mask_values": masked_output["mask_values"],
                "input": text,
                "entities": entities_schema or [],
            }

        except Exception as e:
            logging.error(f"Error during prediction processing: {e}")
            return {
                "status": "error",
                "message": str(e),
                "output": [],
                "raw_output": [],
                "mask_values": {},
                "input": payload.get("text", "") if isinstance(payload, dict) else "",
                "entities": (
                    payload.get("entities", []) if isinstance(payload, dict) else []
                ),
            }


parser = argparse.ArgumentParser(parents=[kserve.model_server.parser])
args, _ = parser.parse_known_args()

if __name__ == "__main__":
    model = GLiNERModel(args.model_name)

    app = kserve.model_server.app
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ModelServer().start([model])

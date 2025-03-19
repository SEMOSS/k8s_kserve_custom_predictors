import os
import random
import string
import json
from typing import Dict, List, Any, Tuple

from kserve import InferRequest, InferOutput
from gliner import GLiNER

from kserve_torch import BaseTorchModel  # type: ignore


class GLiNERModel(BaseTorchModel):
    def __init__(self, name: str):
        super().__init__(name)

    def load(self) -> None:
        """Load the GLiNER model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "urchade/gliner_multi-v2.1")
            self.logger.info(f"Initializing GLiNER model with ID: {model_id}")

            model_dir_exists = self.check_model_dir_exists()
            model_path = None

            if model_dir_exists:
                # Use the local path if it exists
                model_path = self.model_files_base_path
                self.logger.info(f"Found model directory at {model_path}")

            if model_path and os.path.isdir(model_path):
                self.logger.info(f"Loading model from local path: {model_path}")
                try:
                    self.model = GLiNER.from_pretrained(model_path, device=self.device)
                    self.logger.info(f"Successfully loaded model from local path")
                except Exception as e:
                    self.logger.warning(f"Failed to load model from local path: {e}")
                    self.logger.info("Falling back to downloading from HuggingFace")
                    model_path = None

            if not model_path or not hasattr(self, "model") or self.model is None:
                self.logger.info(f"Loading model from HuggingFace Hub: {model_id}")
                self.model = GLiNER.from_pretrained(model_id, device=self.device)

            self.ready = True
            self.logger.info("GLiNER model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
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

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process GLiNER specific inference request"""
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
        self.logger.info(f"output list: {output_list}")
        # Return outputs and empty metadata
        return output_list, {}


# Entry point - much simpler now!
if __name__ == "__main__":
    BaseTorchModel.serve(
        model_class=GLiNERModel, description="GLiNER NER Model Server", log_level="INFO"
    )

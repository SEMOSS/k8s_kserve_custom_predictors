import os
import json
import base64
from typing import Dict, List, Any, Tuple
from kokoro import KPipeline
import soundfile as sf
import numpy as np
from io import BytesIO
from kserve import InferRequest, InferOutput

from kserve_torch import BaseTorchModel  # type: ignore


class KokoroModel(BaseTorchModel):
    def __init__(self, name: str):
        self.pipeline = None
        super().__init__(name)

    def load(self) -> None:
        """Load the Kokoro model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "hexgrad/Kokoro-82M")
            lang_code = os.environ.get("LANG_CODE", "a")
            self.logger.info(
                f"Initializing Kokoro model with ID: {model_id}, lang_code: {lang_code}"
            )

            model_dir_exists = self.check_model_dir_exists()

            if model_dir_exists:
                model_specific_path = os.path.join(
                    self.model_files_base_path, "kokoro-82m"
                )

                config_path = os.path.join(model_specific_path, "config.json")
                model_path_v1 = os.path.join(model_specific_path, "kokoro-v1_0.pth")
                model_path_v1_1 = os.path.join(
                    model_specific_path, "kokoro-v1_1-zh.pth"
                )

                model_path = None
                if os.path.exists(model_path_v1):
                    model_path = model_path_v1
                elif os.path.exists(model_path_v1_1):
                    model_path = model_path_v1_1

                if os.path.exists(config_path) and model_path:
                    self.logger.info(f"Found local config at: {config_path}")
                    self.logger.info(f"Found local model at: {model_path}")

                    try:
                        from kokoro import KModel

                        local_model = (
                            KModel(
                                repo_id=model_id,
                                config=config_path,
                                model=model_path,
                            )
                            .to(self.device)
                            .eval()
                        )

                        self.pipeline = KPipeline(
                            lang_code=lang_code,
                            repo_id=model_id,
                            model=local_model,
                        )

                        self.logger.info(f"Successfully loaded model from local files")
                        self.ready = True
                        return

                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load model from local files: {e}"
                        )
                        self.logger.info("Falling back to downloading from HuggingFace")
                else:
                    missing_files = []
                    if not os.path.exists(config_path):
                        missing_files.append("config.json")
                    if not model_path:
                        missing_files.append("kokoro-v1_0.pth or kokoro-v1_1-zh.pth")

                    self.logger.warning(
                        f"Local directory exists but missing required files: {missing_files}"
                    )

            # Fallback to HuggingFace download
            self.logger.info(f"Loading model from HuggingFace Hub: {model_id}")
            self.pipeline = KPipeline(repo_id=model_id, lang_code=lang_code)
            self.ready = True
            self.logger.info("Kokoro model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def audio_to_base64(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Convert audio numpy array to base64 encoded wav data"""
        try:
            buffer = BytesIO()

            sf.write(buffer, audio_data, sample_rate, format="WAV")

            buffer.seek(0)
            wav_bytes = buffer.read()
            base64_audio = base64.b64encode(wav_bytes).decode("utf-8")

            return f"data:audio/wav;base64,{base64_audio}"

        except Exception as e:
            self.logger.error(f"Error converting audio to base64: {e}")
            raise ValueError(f"Failed to convert audio to base64: {e}")

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process Kokoro TTS inference request"""
        try:
            inputs = request.inputs

            input_names = [inp.name for inp in inputs]
            self.logger.info(f"Request inputs: {input_names}")

            text_input = next((inp for inp in inputs if inp.name == "text"), None)
            if text_input is None:
                raise ValueError("Text input is required")

            text = text_input.data[0]
            self.logger.info(f"Processing text: '{text}'")

            voice_input = next((inp for inp in inputs if inp.name == "voice"), None)
            voice = voice_input.data[0] if voice_input else "af_bella"

            speed_input = next((inp for inp in inputs if inp.name == "speed"), None)
            speed = float(speed_input.data[0]) if speed_input else 1.0

            self.logger.info(f"Using voice: {voice}, speed: {speed}")

            generator = self.pipeline(text, voice=voice, speed=speed)

            audio_chunks = []
            for i, (gs, ps, audio) in enumerate(generator):
                self.logger.debug(
                    f"Generated chunk {i}: gs={gs}, ps={ps}, audio_shape={audio.shape}"
                )
                audio_chunks.append(audio)

            if audio_chunks:
                audio_data = np.concatenate(audio_chunks)
            else:
                raise ValueError("No audio data generated")

            sample_rate = 24000

            self.logger.info(
                f"Generated audio with {len(audio_chunks)} chunks, total shape: {audio_data.shape}, sample_rate: {sample_rate}"
            )

            base64_audio = self.audio_to_base64(audio_data, sample_rate)

            result = {
                "audio": base64_audio,
                "sample_rate": sample_rate,
                "duration": len(audio_data) / sample_rate,
                "voice": voice,
                "speed": speed,
                "text": text,
            }

            output_list = [
                InferOutput(
                    name="output",
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
                    name="output",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(error_data)],
                ),
            ]
            return output_list, {}


if __name__ == "__main__":
    BaseTorchModel.serve(
        model_class=KokoroModel,
        description="Kokoro Text-to-Speech Model Server",
        log_level="INFO",
    )

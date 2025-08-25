import os
import json
import base64
import tempfile
from typing import Dict, List, Any, Tuple, Optional
import torch
import whisper
import librosa
import soundfile as sf
from io import BytesIO
from kserve import InferRequest, InferOutput
from pydantic import BaseModel
import kserve
from fastapi import Depends, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse

from kserve_torch import BaseTorchModel  # type: ignore

app = kserve.model_server.app


def get_whisper_model():
    if not hasattr(app.state, "whisper_model") or app.state.whisper_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return app.state.whisper_model


class TranscriptionRequest(BaseModel):
    file: str  # base64 encoded audio file
    model: Optional[str] = "whisper-large-v3"
    language: Optional[str] = None
    prompt: Optional[str] = None
    response_format: Optional[str] = "json"
    temperature: Optional[float] = 0.0
    timestamp_granularities: Optional[List[str]] = None


@app.post("/v1/audio/transcriptions")
async def transcribe_audio(
    file: UploadFile = File(...),
    model: str = Form("whisper-large-v3"),
    language: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
    timestamp_granularities: Optional[str] = Form(None),
    whisper_model=Depends(get_whisper_model),
):
    """OpenAI-compatible transcription endpoint"""
    try:
        # Read the uploaded file
        audio_content = await file.read()

        # Create a temporary file to store the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_file.write(audio_content)
            temp_file_path = temp_file.name

        try:
            result = await whisper_model.process_file_transcription(
                file_path=temp_file_path,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
                timestamp_granularities=(
                    timestamp_granularities.split(",")
                    if timestamp_granularities
                    else None
                ),
            )
            return result
        finally:
            # Clean up temporary file
            os.unlink(temp_file_path)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "transcription_error"}},
        )


@app.post("/v1/audio/transcriptions/base64")
async def transcribe_audio_base64(
    request: TranscriptionRequest, whisper_model=Depends(get_whisper_model)
):
    """Base64 transcription endpoint"""
    try:
        result = await whisper_model.process_base64_transcription(
            base64_audio=request.file,
            language=request.language,
            prompt=request.prompt,
            response_format=request.response_format,
            temperature=request.temperature,
            timestamp_granularities=request.timestamp_granularities,
        )
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": {"message": str(e), "type": "transcription_error"}},
        )


class WhisperModel(BaseTorchModel):
    def __init__(self, name: str):
        self.whisper_model = None
        super().__init__(name)
        app.state.whisper_model = self

    def load(self) -> None:
        """Load the Whisper model from local storage or download"""
        try:
            model_id = os.environ.get("MODEL_ID", "openai/whisper-large-v3")
            model_size = os.environ.get("MODEL_SIZE", "large-v3")
            self.logger.info(f"Initializing Whisper model with ID: {model_id}")

            # Check if model directory exists in the mounted volume
            model_dir_exists = self.check_model_dir_exists()
            model_path = None

            if model_dir_exists:
                # Look for whisper model files in the mounted volume
                model_specific_path = os.path.join(
                    self.model_files_base_path, "whisper-large-v3"
                )
                if os.path.isdir(model_specific_path):
                    model_path = model_specific_path
                    self.logger.info(f"Found model directory at {model_path}")

            # Try to load from local path first
            if model_path and os.path.isdir(model_path):
                self.logger.info(f"Loading model from local path: {model_path}")
                try:
                    # For local models, we need to load using the transformers library
                    from transformers import (
                        WhisperForConditionalGeneration,
                        WhisperProcessor,
                    )

                    self.processor = WhisperProcessor.from_pretrained(model_path)
                    self.whisper_model = (
                        WhisperForConditionalGeneration.from_pretrained(model_path).to(
                            self.device
                        )
                    )

                    self.logger.info("Successfully loaded model from local path")
                    self.ready = True
                    return

                except Exception as e:
                    self.logger.warning(f"Failed to load model from local path: {e}")
                    self.logger.info("Falling back to downloading from OpenAI")

            # Fallback to downloading the model using OpenAI Whisper
            self.logger.info(f"Loading model using OpenAI Whisper: {model_size}")
            self.whisper_model = whisper.load_model(model_size, device=self.device)

            self.ready = True
            self.logger.info("Whisper model loaded successfully")

        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def decode_base64_audio(self, base64_string: str) -> Tuple[BytesIO, str]:
        """Decode base64 audio string to BytesIO object"""
        try:
            if base64_string.startswith("data:"):
                # Extract the MIME type and base64 data
                header, base64_data = base64_string.split(",", 1)
                mime_type = header.split(":")[1].split(";")[0]
                file_extension = mime_type.split("/")[1]
            else:
                base64_data = base64_string
                file_extension = "wav"  # Default to wav

            # Add padding if necessary
            padding_needed = len(base64_data) % 4
            if padding_needed:
                base64_data += "=" * (4 - padding_needed)

            audio_bytes = base64.b64decode(base64_data)
            audio_buffer = BytesIO(audio_bytes)

            self.logger.info(
                f"Successfully decoded base64 audio of size {len(audio_bytes)} bytes"
            )
            return audio_buffer, file_extension

        except Exception as e:
            self.logger.error(f"Error decoding base64 audio: {e}")
            raise ValueError(f"Failed to decode base64 audio: {e}")

    def load_audio_from_buffer(
        self, audio_buffer: BytesIO, file_extension: str
    ) -> torch.Tensor:
        """Load audio from BytesIO buffer and convert to format expected by Whisper"""
        try:
            audio_buffer.seek(0)

            # Create temporary file to work with librosa
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as temp_file:
                temp_file.write(audio_buffer.read())
                temp_file_path = temp_file.name

            try:
                # Load audio using librosa (Whisper expects 16kHz sample rate)
                audio_data, sr = librosa.load(temp_file_path, sr=16000, mono=True)
                return torch.from_numpy(audio_data).float()
            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            self.logger.error(f"Error loading audio from buffer: {e}")
            raise ValueError(f"Failed to load audio: {e}")

    def format_transcription_result(
        self,
        result: Dict[str, Any],
        response_format: str = "json",
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Format the transcription result according to OpenAI API format"""

        if response_format == "text":
            return {"text": result["text"]}

        # Default JSON format
        formatted_result = {"text": result["text"]}

        # Add segments if available and requested
        if "segments" in result and timestamp_granularities:
            if "segment" in timestamp_granularities:
                formatted_result["segments"] = []
                for segment in result["segments"]:
                    formatted_segment = {
                        "id": segment.get("id", 0),
                        "seek": segment.get("seek", 0),
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "text": segment.get("text", ""),
                        "tokens": segment.get("tokens", []),
                        "temperature": segment.get("temperature", 0.0),
                        "avg_logprob": segment.get("avg_logprob", 0.0),
                        "compression_ratio": segment.get("compression_ratio", 0.0),
                        "no_speech_prob": segment.get("no_speech_prob", 0.0),
                    }
                    formatted_result["segments"].append(formatted_segment)

        return formatted_result

    async def process_file_transcription(
        self,
        file_path: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process file transcription"""
        try:
            # Load audio using whisper's built-in audio loading
            audio = whisper.load_audio(file_path)

            # Transcribe
            options = {
                "language": language,
                "task": "transcribe",
                "temperature": temperature,
                "word_timestamps": "word" in (timestamp_granularities or []),
            }

            if prompt:
                options["prompt"] = prompt

            result = whisper.transcribe(self.whisper_model, audio, **options)

            return self.format_transcription_result(
                result, response_format, timestamp_granularities
            )

        except Exception as e:
            self.logger.error(f"Error during file transcription: {e}")
            raise

    async def process_base64_transcription(
        self,
        base64_audio: str,
        language: Optional[str] = None,
        prompt: Optional[str] = None,
        response_format: str = "json",
        temperature: float = 0.0,
        timestamp_granularities: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Process base64 audio transcription"""
        try:
            # Decode base64 audio
            audio_buffer, file_extension = self.decode_base64_audio(base64_audio)

            # Create temporary file for whisper processing
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=f".{file_extension}"
            ) as temp_file:
                audio_buffer.seek(0)
                temp_file.write(audio_buffer.read())
                temp_file_path = temp_file.name

            try:
                return await self.process_file_transcription(
                    file_path=temp_file_path,
                    language=language,
                    prompt=prompt,
                    response_format=response_format,
                    temperature=temperature,
                    timestamp_granularities=timestamp_granularities,
                )
            finally:
                os.unlink(temp_file_path)

        except Exception as e:
            self.logger.error(f"Error during base64 transcription: {e}")
            raise

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process Whisper inference request through KServe interface"""
        try:
            inputs = request.inputs
            input_names = [inp.name for inp in inputs]
            self.logger.info(f"Request inputs: {input_names}")

            # Get audio input
            audio_input = next((inp for inp in inputs if inp.name == "audio"), None)
            if audio_input is None:
                raise ValueError("Audio input is required")

            audio_data = audio_input.data[0]
            self.logger.info("Processing audio input")

            # Get optional parameters
            language_input = next(
                (inp for inp in inputs if inp.name == "language"), None
            )
            language = language_input.data[0] if language_input else None

            prompt_input = next((inp for inp in inputs if inp.name == "prompt"), None)
            prompt = prompt_input.data[0] if prompt_input else None

            response_format_input = next(
                (inp for inp in inputs if inp.name == "response_format"), None
            )
            response_format = (
                response_format_input.data[0] if response_format_input else "json"
            )

            temperature_input = next(
                (inp for inp in inputs if inp.name == "temperature"), None
            )
            temperature = float(temperature_input.data[0]) if temperature_input else 0.0

            # Process transcription
            result = await self.process_base64_transcription(
                base64_audio=audio_data,
                language=language,
                prompt=prompt,
                response_format=response_format,
                temperature=temperature,
            )

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
        model_class=WhisperModel,
        description="OpenAI Whisper Speech-to-Text Model Server",
        log_level="INFO",
    )

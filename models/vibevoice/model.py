# /app/model.py
import os, json, torch, numpy as np, soundfile as sf, base64
from io import BytesIO
from typing import Dict, List, Any, Tuple
from kserve import InferRequest, InferOutput
from kserve_torch import BaseTorchModel


class VibeVoiceModel(BaseTorchModel):
    def __init__(self, name: str):
        self.processor = None
        self.model = None
        self.sample_rate = 24000
        self.use_pipeline = False  # <-- force off to avoid TorchVision / pipelines
        super().__init__(name)

    def _demo_voices(self) -> List[str]:
        # Use the voices shipped by the repo (installed into /tmp/VibeVoice during docker build)
        base = "/tmp/VibeVoice/demo/voices"
        candidates = [
            "en-Alice_woman.wav",
            "en-Carter_man.wav",
            "en-Frank_man.wav",
            "en-Maya_woman.wav",
        ]
        return [
            os.path.join(base, f)
            for f in candidates
            if os.path.exists(os.path.join(base, f))
        ]

    def load(self) -> None:
        try:
            model_id = os.environ.get("MODEL_ID", "vibevoice/VibeVoice-7B")
            self.logger.info(f"Initializing VibeVoice model with ID: {model_id}")

            # Prefer mounted model dir if present
            model_path = (
                self.model_files_base_path
                if self.check_model_dir_exists()
                else model_id
            )
            if model_path != model_id:
                self.logger.info(f"Found model directory at {model_path}")

            # Import community fork classes
            from vibevoice.modular.modeling_vibevoice_inference import (
                VibeVoiceForConditionalGenerationInference,
            )
            from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

            self.processor = VibeVoiceProcessor.from_pretrained(model_path)

            # Pick dtype/attn based on device (mirror the demo)
            if self.device == "cuda":
                load_dtype = (
                    torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                attn_impl = "flash_attention_2"
                device_map = "cuda"
            else:
                load_dtype = torch.float32
                attn_impl = "sdpa"
                device_map = self.device  # "cpu" or "mps"

            self.logger.info(
                f"Loading model (dtype={load_dtype}, attn={attn_impl}, device_map={device_map})"
            )
            try:
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype,
                    device_map=device_map,
                    attn_implementation=attn_impl,
                )
            except Exception as e:
                # If anything related to flash-attn fails, retry with SDPA
                self.logger.warning(
                    f"Flash-attention load failed ({type(e).__name__}: {e}); retrying with SDPA"
                )
                self.model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                    model_path,
                    torch_dtype=load_dtype if self.device == "cuda" else torch.float32,
                    device_map=device_map,
                    attn_implementation="sdpa",
                )

            self.model.eval()

            # Use SDE solver + set inference steps similarly to the demo
            try:
                self.model.model.noise_scheduler = (
                    self.model.model.noise_scheduler.from_config(
                        self.model.model.noise_scheduler.config,
                        algorithm_type="sde-dpmsolver++",
                        beta_schedule="squaredcos_cap_v2",
                    )
                )
            except Exception:
                pass
            self.model.set_ddpm_inference_steps(num_steps=10)

            self.ready = True
            self.logger.info(f"VibeVoice loaded on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def _encode_audio_b64(self, audio: np.ndarray, sr: int) -> str:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        m = np.max(np.abs(audio)) if audio.size else 1.0
        if m > 1.0:
            audio = audio / m
        with BytesIO() as buf:
            sf.write(buf, audio, sr, format="WAV")
            return base64.b64encode(buf.getvalue()).decode("utf-8")

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        try:
            # Required: text
            text_inp = next((i for i in request.inputs if i.name == "text"), None)
            if not text_inp:
                raise ValueError("Missing required input 'text'")
            text = text_inp.data[0]

            # Optional: cfg_scale (default 1.3 like demo)
            cfg_inp = next((i for i in request.inputs if i.name == "cfg_scale"), None)
            cfg_scale = float(cfg_inp.data[0]) if cfg_inp else 1.3

            # Optional: voice_samples (base64 wavs). If absent, use demo voices.
            voice_inp = [i for i in request.inputs if i.name == "voice_samples"]
            voice_paths: List[str] = []
            if voice_inp:
                # Each element should be base64-encoded WAV bytes
                tmp_dir = "/tmp/voice_uploads"
                os.makedirs(tmp_dir, exist_ok=True)
                for idx, item in enumerate(voice_inp[0].data):
                    wav_bytes = base64.b64decode(item)
                    p = os.path.join(tmp_dir, f"voice_{idx}.wav")
                    with open(p, "wb") as f:
                        f.write(wav_bytes)
                    voice_paths.append(p)
            else:
                voice_paths = self._demo_voices()

            if not voice_paths:
                raise ValueError(
                    "No voice samples available; provide 'voice_samples' or ship demo voices."
                )

            # Build inputs like the demo (batch size 1)
            inputs = self.processor(
                text=[text.replace("’", "'")],
                voice_samples=[voice_paths],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            # Move tensors
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(
                        self.device if self.device in ("cuda", "mps") else "cpu"
                    )

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=cfg_scale,
                    tokenizer=self.processor.tokenizer,
                    generation_config={"do_sample": False},
                    verbose=False,
                )

            # Extract audio (first item of batch)
            # Community inference outputs store speech in `speech_outputs`
            if (
                hasattr(outputs, "speech_outputs")
                and outputs.speech_outputs
                and outputs.speech_outputs[0] is not None
            ):
                audio = outputs.speech_outputs[0]
                if torch.is_tensor(audio):
                    audio = audio.float().cpu().numpy()
            elif hasattr(outputs, "speech"):
                audio = outputs.speech
                if torch.is_tensor(audio):
                    audio = audio.float().cpu().numpy()
            else:
                raise RuntimeError("No audio found in model outputs")

            # Flatten and encode
            audio = np.array(audio).squeeze()
            audio_b64 = self._encode_audio_b64(audio, self.sample_rate)

            payload = {
                "audio_base64": audio_b64,
                "sample_rate": self.sample_rate,
                "duration_seconds": float(len(audio) / self.sample_rate),
                "cfg_scale": cfg_scale,
                "attn": getattr(self.model.model, "language_model", None)
                and getattr(
                    self.model.model.language_model.config,
                    "_attn_implementation",
                    "unknown",
                ),
            }

            return [
                InferOutput(
                    name="audio",
                    datatype="BYTES",
                    shape=[1],
                    data=[json.dumps(payload)],
                )
            ], {}

        except Exception as e:
            self.logger.error(f"Error processing request: {e}", exc_info=True)
            err = {"error": str(e)}
            return [
                InferOutput(
                    name="audio", datatype="BYTES", shape=[1], data=[json.dumps(err)]
                )
            ], {}


if __name__ == "__main__":
    BaseTorchModel.serve(
        model_class=VibeVoiceModel,
        description="VibeVoice Text-to-Speech Model Server",
        log_level="INFO",
    )

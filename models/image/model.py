import os
import json
import base64
from io import BytesIO
from typing import Dict, List, Any, Tuple, Optional
from PIL import Image
import torch
from diffusers import (
    StableDiffusionPipeline,
    DiffusionPipeline,
    AutoPipelineForText2Image,
)
from kserve import InferRequest, InferOutput

from kserve_torch import BaseTorchModel  # type: ignore


class ImageGenerationModel(BaseTorchModel):
    """
    KServe predictor for image generation models like stable-diffusion-v1-5.

    This class can handle various image generation models from HuggingFace,
    focusing primarily on text-to-image generation models.
    """

    def __init__(self, name: str):
        self.tokenizer = None
        self.pipeline = None
        self.model_type = None
        self.default_height = int(os.environ.get("DEFAULT_HEIGHT", "512"))
        self.default_width = int(os.environ.get("DEFAULT_WIDTH", "512"))
        self.default_num_inference_steps = int(
            os.environ.get("DEFAULT_NUM_INFERENCE_STEPS", "50")
        )
        self.default_guidance_scale = float(
            os.environ.get("DEFAULT_GUIDANCE_SCALE", "7.5")
        )
        self.default_negative_prompt = os.environ.get("DEFAULT_NEGATIVE_PROMPT", "")

        super().__init__(name)

    def load(self) -> None:
        """Load the image generation model from local storage or HuggingFace Hub"""
        try:
            model_id = os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5")
            self.logger.info(f"Initializing image generation model with ID: {model_id}")

            model_dir_exists = self.check_model_dir_exists()
            model_path = None

            if model_dir_exists:
                model_path = self.model_files_base_path
                self.logger.info(f"Found model directory at {model_path}")

            if model_path and os.path.isdir(model_path):
                self.logger.info(f"Loading model from local path: {model_path}")
                try:
                    # Try using AutoPipelineForText2Image for maximum compatibility
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_path,
                        torch_dtype=(
                            torch.float16 if self.device == "cuda" else torch.float32
                        ),
                        use_safetensors=True,
                        trust_remote_code=True,
                    ).to(self.device)
                    self.logger.info(
                        f"Successfully loaded model from local path using AutoPipelineForText2Image"
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load with AutoPipelineForText2Image: {e}"
                    )
                    self.logger.info("Falling back to StableDiffusionPipeline")
                    try:
                        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            model_path,
                            torch_dtype=(
                                torch.float16
                                if self.device == "cuda"
                                else torch.float32
                            ),
                            use_safetensors=True,
                        ).to(self.device)
                        self.logger.info(
                            f"Successfully loaded model from local path using StableDiffusionPipeline"
                        )
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load with StableDiffusionPipeline: {e}"
                        )
                        self.logger.info("Falling back to generic DiffusionPipeline")
                        try:
                            self.pipeline = DiffusionPipeline.from_pretrained(
                                model_path,
                                torch_dtype=(
                                    torch.float16
                                    if self.device == "cuda"
                                    else torch.float32
                                ),
                                use_safetensors=True,
                            ).to(self.device)
                            self.logger.info(
                                f"Successfully loaded model from local path using DiffusionPipeline"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to load from local path with all methods: {e}"
                            )
                            self.logger.info(
                                "Falling back to downloading from HuggingFace"
                            )
                            model_path = None

            if not model_path or self.pipeline is None:
                self.logger.info(f"Loading model from HuggingFace Hub: {model_id}")
                # Try using AutoPipelineForText2Image for maximum compatibility
                try:
                    self.pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_id,
                        torch_dtype=(
                            torch.float16 if self.device == "cuda" else torch.float32
                        ),
                        use_safetensors=True,
                        trust_remote_code=True,
                    ).to(self.device)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load with AutoPipelineForText2Image: {e}"
                    )
                    self.logger.info("Falling back to StableDiffusionPipeline")
                    try:
                        self.pipeline = StableDiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=(
                                torch.float16
                                if self.device == "cuda"
                                else torch.float32
                            ),
                            use_safetensors=True,
                        ).to(self.device)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to load with StableDiffusionPipeline: {e}"
                        )
                        self.logger.info("Falling back to generic DiffusionPipeline")
                        self.pipeline = DiffusionPipeline.from_pretrained(
                            model_id,
                            torch_dtype=(
                                torch.float16
                                if self.device == "cuda"
                                else torch.float32
                            ),
                            use_safetensors=True,
                        ).to(self.device)

            # Try to enable memory optimization if on CUDA
            if self.device == "cuda":
                try:
                    self.pipeline.enable_attention_slicing()
                    self.logger.info(
                        "Enabled attention slicing for memory optimization"
                    )
                except:
                    self.logger.warning("Could not enable attention slicing")

                try:
                    # Only use xformers if available and on CUDA
                    if torch.cuda.is_available():
                        try:
                            import xformers

                            self.pipeline.enable_xformers_memory_efficient_attention()
                            self.logger.info(
                                "Enabled xformers for memory efficient attention"
                            )
                        except ImportError:
                            self.logger.warning(
                                "xformers is not installed, skipping memory optimization"
                            )
                except:
                    self.logger.warning("Could not enable xformers memory optimization")

            self.model_type = self.pipeline.__class__.__name__
            self.logger.info(f"Loaded model type: {self.model_type}")

            self.logger.info(
                f"Default image dimensions: {self.default_width}x{self.default_height}"
            )
            self.logger.info(
                f"Default inference steps: {self.default_num_inference_steps}"
            )
            self.logger.info(f"Default guidance scale: {self.default_guidance_scale}")

            self.ready = True
            self.logger.info(f"Image generation model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}", exc_info=True)
            raise

    def _encode_image_to_base64(self, image: Image.Image, format="PNG") -> str:
        """
        Encode a PIL Image to base64 string

        Args:
            image: PIL Image to encode
            format: Image format (PNG, JPEG, etc.)

        Returns:
            base64 encoded string with format prefix
        """
        buffered = BytesIO()
        image.save(buffered, format=format)
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/{format.lower()};base64,{img_str}"

    def _generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        num_images: int = 1,
        return_dict: bool = True,
    ) -> Dict[str, Any]:
        """
        Generate images based on text prompts

        Args:
            prompt: Text prompt to generate image from
            negative_prompt: Text prompt for things to avoid in image
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility
            num_images: Number of images to generate
            return_dict: Whether to return a dict or raw images

        Returns:
            Dictionary with generated images and metadata or list of PIL Images
        """
        self.logger.info(f"Generating image with prompt: {prompt}")

        height = height or self.default_height
        width = width or self.default_width
        num_inference_steps = num_inference_steps or self.default_num_inference_steps
        guidance_scale = guidance_scale or self.default_guidance_scale
        negative_prompt = negative_prompt or self.default_negative_prompt

        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
            self.logger.info(f"Using seed: {seed}")

        try:
            # First check if the pipeline has an expected signature
            if hasattr(self.pipeline, "text2img") and callable(
                getattr(self.pipeline, "text2img")
            ):
                # Handle SDXL Turbo and similar models
                output = self.pipeline.text2img(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=num_images,
                )
            else:
                # Standard StableDiffusionPipeline call
                output = self.pipeline(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,
                    width=width,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    generator=generator,
                    num_images_per_prompt=num_images,
                )

        except Exception as e:
            self.logger.error(f"Error during image generation: {e}", exc_info=True)
            try:
                # Try with minimal parameters as fallback
                self.logger.info("Trying with minimal parameters")
                output = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=num_inference_steps,
                )
            except Exception as e2:
                self.logger.error(
                    f"Fallback generation also failed: {e2}", exc_info=True
                )
                raise ValueError(f"Failed to generate image: {str(e2)}")

        if not return_dict:
            return output.images

        encoded_images = []
        for img in output.images:
            encoded_images.append(self._encode_image_to_base64(img))

        result = {
            "images": encoded_images,
            "parameters": {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
            },
            "model_id": os.environ.get("MODEL_ID", "runwayml/stable-diffusion-v1-5"),
        }

        # Add nsfw content detected if available
        if hasattr(output, "nsfw_content_detected") and output.nsfw_content_detected:
            result["nsfw_content_detected"] = output.nsfw_content_detected

        return result

    async def process_request(
        self, request: InferRequest
    ) -> Tuple[List[InferOutput], Dict[str, Any]]:
        """Process image generation model inference request"""
        try:
            inputs = request.inputs
            input_names = [inp.name for inp in inputs]
            self.logger.info(f"Request inputs: {input_names}")

            prompt_input = next((inp for inp in inputs if inp.name == "prompt"), None)
            if prompt_input is None:
                raise ValueError("Request requires a 'prompt' input")

            prompt = prompt_input.data[0]
            self.logger.info(f"Processing prompt: {prompt}")

            # Get optional parameters
            negative_prompt = None
            negative_prompt_input = next(
                (inp for inp in inputs if inp.name == "negative_prompt"), None
            )
            if negative_prompt_input and negative_prompt_input.data:
                negative_prompt = negative_prompt_input.data[0]

            height = None
            height_input = next((inp for inp in inputs if inp.name == "height"), None)
            if height_input and height_input.data:
                height = int(height_input.data[0])

            width = None
            width_input = next((inp for inp in inputs if inp.name == "width"), None)
            if width_input and width_input.data:
                width = int(width_input.data[0])

            num_inference_steps = None
            steps_input = next(
                (inp for inp in inputs if inp.name == "num_inference_steps"), None
            )
            if steps_input and steps_input.data:
                num_inference_steps = int(steps_input.data[0])

            guidance_scale = None
            guidance_input = next(
                (inp for inp in inputs if inp.name == "guidance_scale"), None
            )
            if guidance_input and guidance_input.data:
                guidance_scale = float(guidance_input.data[0])

            seed = None
            seed_input = next((inp for inp in inputs if inp.name == "seed"), None)
            if seed_input and seed_input.data:
                seed = int(seed_input.data[0])

            num_images = 1
            num_images_input = next(
                (inp for inp in inputs if inp.name == "num_images"), None
            )
            if num_images_input and num_images_input.data:
                num_images = min(
                    int(num_images_input.data[0]), 4
                )  # Limit to 4 images max

            result = self._generate_image(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                seed=seed,
                num_images=num_images,
            )

            self.logger.info(f"Generated {len(result['images'])} images")

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
        model_class=ImageGenerationModel,
        description="Image Generation Model Server (supports models like Stable Diffusion)",
        log_level="INFO",
    )

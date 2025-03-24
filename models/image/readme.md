# Image Generation Model for KServe

This directory contains a KServe implementation of text-to-image generation models like Stable Diffusion.

## Features

- Supports various text-to-image generation models from HuggingFace
- Compatible with Stable Diffusion and similar architectures
- Configurable image dimensions, inference steps, and guidance scale
- Supports negative prompts for better image quality control
- Includes seed control for reproducible results
- Can generate multiple images in a single request
- Outputs base64-encoded images for easy integration

## Docker Build

```bash
# Build the model image (GPU version)
docker build -f models/image/Dockerfile --build-arg MODE=gpu -t image-generation-predictor:gpu .

# Build the model image (CPU version) NOT RECOMMENDED
docker build -f models/image/Dockerfile --build-arg MODE=cpu -t image-generation-predictor:cpu .
```

## Base Image

This model defaults to the GPU version of the PyTorch KServe image (`cfg-ms-torch-gpu`). Using the GPU version is strongly recommended for reasonable inference times.

## Docker Run

```bash
# Run the model locally (GPU)
docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/stable-diffusion-v1-5:/mnt/models image-generation-predictor:gpu

# Run the model locally (CPU) NOT RECOMMENDED
docker run -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/stable-diffusion-v1-5:/mnt/models image-generation-predictor:cpu
```

## Python Packages Added

The following packages are added in the model level Docker image:
- diffusers==0.25.0
- safetensors==0.4.0
- xformers==0.0.22.post7 (for GPU optimization)

## Environment Variables

You can customize the model behavior with these environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| MODEL_ID | stable-diffusion-v1-5/stable-diffusion-v1-5 | HuggingFace model ID |
| DEFAULT_HEIGHT | 512 | Default image height |
| DEFAULT_WIDTH | 512 | Default image width |
| DEFAULT_NUM_INFERENCE_STEPS | 50 | Default denoising steps |
| DEFAULT_GUIDANCE_SCALE | 7.5 | Default guidance scale |
| DEFAULT_NEGATIVE_PROMPT | "low quality, blurry..." | Default negative prompt |

## Example Usage

### Endpoint
```
POST /v2/models/stable-diffusion-v1-5/infer
```

### Basic Payload Example
```json
{
  "inputs": [
    {
      "name": "prompt",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["A photo of a cat in space, wearing an astronaut helmet"]
    }
  ]
}
```

### Advanced Payload Example
```json
{
  "inputs": [
    {
      "name": "prompt",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["A highly detailed digital painting of a magical forest with glowing mushrooms, 4k resolution"]
    },
    {
      "name": "negative_prompt",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["blurry, low quality, distorted, text, watermark"]
    },
    {
      "name": "height",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["768"]
    },
    {
      "name": "width",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["768"]
    },
    {
      "name": "num_inference_steps",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["30"]
    },
    {
      "name": "guidance_scale",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["7.5"]
    },
    {
      "name": "seed",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["42"]
    },
    {
      "name": "num_images",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["2"]
    }
  ]
}
```

### Response Example
```json
{
  "model_name": "image-generation",
  "model_version": null,
  "id": "cb7c1c36-e5d4-4d82-9e1f-c238d7d08dfc",
  "parameters": null,
  "outputs": [
    {
      "name": "output",
      "shape": [1],
      "datatype": "BYTES",
      "parameters": null,
      "data": [
        "{\"images\":[\"data:image/png;base64,iVBORw0KGgoAAAANSUh...\", \"data:image/png;base64,iVBORw0KGgoAAAANSUh...\"],\"parameters\":{\"prompt\":\"A highly detailed digital painting of a magical forest with glowing mushrooms, 4k resolution\",\"negative_prompt\":\"blurry, low quality, distorted, text, watermark\",\"height\":768,\"width\":768,\"num_inference_steps\":30,\"guidance_scale\":7.5,\"seed\":42},\"model_id\":\"runwayml/stable-diffusion-v1-5\"}"
      ]
    }
  ]
}
```

## Supported Models

This implementation has been tested with the following models.. If you test with a different model that works, please add it here:

- stable-diffusion-v1-5/stable-diffusion-v1-5

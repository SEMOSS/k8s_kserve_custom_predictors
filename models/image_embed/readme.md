# Image Embedding Model for KServe

This KServe predictor supports various image embedding models from HuggingFace, including:
- facebook/dinov2-large
- facebook/dinov2-base
- facebook/dinov2-small
- facebook/dinov2-giant
- openai/clip-vit-base-patch32
- microsoft/swinv2-base-patch4-window12-192-22k
- google/vit-base-patch16-224
- ... and many other vision encoder models

## Features

- Extracts image embeddings using pre-trained models
- Supports multiple pooling strategies (cls, mean, max)
- Option to normalize embeddings
- Handles both URL and base64-encoded images
- Supports batch processing of multiple images
- Can return raw embeddings for efficient data transfer

## Docker Build

### GPU Image (Default)
```bash
docker build -f models/image_embed/Dockerfile --build-arg MODE=gpu -t image-embedding-predictor:gpu .
```

### CPU Image
```bash
docker build -f models/image_embed/Dockerfile --build-arg MODE=cpu -t image-embedding-predictor:cpu .
```

## Base Image

This model defaults to the GPU base image (`cfg-ms-torch-gpu:latest`).

## Docker Run

### Run locally with GPU
```bash
docker run --gpus all -p 8080:8080 -e MODEL_ID="facebook/dinov2-large" -e MODEL_NAME="dinov2-large" -v /var/lib/docker/volumes/model-volume/_data/dinov2-large:/mnt/models image-embedding-predictor:gpu
```

### Run locally with CPU
```bash
docker run -p 8080:8080 -e MODEL_ID="facebook/dinov2-base" -e POOLING_STRATEGY="mean" -e NORMALIZE_EMBEDDINGS="true" -v /var/lib/docker/volumes/model-volume/_data/dinov2-large:/mnt/models image-embedding-predictor:cpu
```

## Example Endpoint

```
POST http://localhost:8080/v2/models/dinov2-large/infer`
```

## Example Payload

### Single Image (URL)
```json
{
  "inputs": [
    {
      "name": "image",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"]
    },
    {
      "name": "pooling_strategy", 
      "shape": [1],
      "datatype": "BYTES",
      "data": ["mean"]
    }
  ]
}
```

### Single Image (Base64)
```json
{
  "inputs": [
    {
      "name": "image",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEA..."]
    }
  ]
}
```

### Multiple Images Batch
```json
{
  "inputs": [
    {
      "name": "image",
      "shape": [2],
      "datatype": "BYTES",
      "data": [
        "https://example.com/path/to/image1.jpg",
        "https://example.com/path/to/image2.jpg"
      ]
    },
    {
      "name": "include_raw",
      "shape": [1],
      "datatype": "BYTES",
      "data": ["true"]
    }
  ]
}
```

## Example Response

```json
{
  "id": "8b20be37-5c73-4183-a5a2-3aca1719e1a1",
  "model_name": "image-embedding",
  "outputs": [
    {
      "name": "output",
      "shape": [1],
      "datatype": "BYTES",
      "data": [
        "{\"embedding_dim\": 1024, \"embeddings\": [[0.1, 0.2, 0.3, ...]], \"model_id\": \"facebook/dinov2-large\", \"shape\": [1, 1024]}"
      ]
    }
  ]
}
```

With `include_raw` set to true, a second output containing the raw binary embedding data will be included.

# Florence-2-large Vision-Language Model for KServe

This directory contains a KServe implementation of Microsoft's Florence-2-large vision-language model.

## Features

Florence-2-large is a powerful multimodal model that can perform various tasks:

- `<CAPTION>`: Generates a concise description of the entire image, summarizing its main content
- `<DETAILED_CAPTION>`: Produces a more elaborate description of the image, including finer details and contextual information
- `<MORE_DETAILED_CAPTION>`: Creates an exceptionally detailed caption, capturing intricate attributes and relationships within the image
- `<OD>`: Detects objects in an image and provides their bounding box coordinates along with labels
- `<DENSE_REGION_CAPTION>`: Generates captions for densely packed regions within an image, identifying multiple objects or areas simultaneously
- `<REGION_PROPOSAL>`: Suggests specific regions in an image that may contain objects or areas of interest for further analysis
- `<CAPTION_TO_PHRASE_GROUNDING> your_text_input`: Aligns phrases from a generated caption with specific regions in the image, enabling precise visual-textual mapping
- `<REFERRING_EXPRESSION_SEGMENTATION> your_text_input`: Segments parts of an image based on textual descriptions of specific objects or regions
- `<REGION_TO_SEGMENTATION> your_text_input`: Converts bounding boxes into segmentation masks to outline specific objects or areas within an image
- `<OCR>`: Extracts text from an image as a single string, useful for reading printed or handwritten text
- `<OCR_WITH_REGION>`: Retrieves text from an image along with its location, providing bounding boxes for each piece of text


## Docker Build

```bash
# Build the model image (GPU version)
docker build -f models/florence/Dockerfile --build-arg MODE=gpu -t florence-predictor:gpu .

# Build the model image (CPU version) NOT RECOMMENDED
docker build -f models/florence/Dockerfile --build-arg MODE=cpu -t florence-predictor:cpu .
```

## Base Image

This model defaults to the GPU version of the PyTorch KServe image (`cfg-ms-torch-gpu`).

```bash
# Run the model locally (GPU)
docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/florence-2-large:/mnt/models florence-predictor:gpu

# Run the model locally (CPU) NOT RECOMMENDED
docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/florence-2-large:/mnt/models florence-predictor:cpu
```

## Example Usage

### Image Input Support

The model supports two types of image inputs:

1. **Base64-encoded images**: Send the base64-encoded string of an image
2. **Image URLs**: Send a direct URL to an image (must start with `http://` or `https://`)

#### Endpoint
```
POST /v2/models/florence-2-large/infer
```

### Object Detection Payload Example
```json
{
  "inputs": [
    {
      "name": "image",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"]
    },
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["<OD>"]
    }
  ]
}
```

#### Object Detection Response Example
```json
{
    "model_name": "florence-2-large",
    "model_version": null,
    "id": "715d8559-bc62-4555-9c20-e61da4dd1a43",
    "parameters": null,
    "outputs": [
        {
            "name": "output",
            "shape": [
                1
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "{\"bboxes\": [[33.599998474121094, 160.55999755859375, 596.7999877929688, 371.7599792480469], [271.67999267578125, 242.1599884033203, 302.3999938964844, 246.95999145507812]], 
                \"labels\": [\"turquoise Volkswagen Beetle\", \"door handle\"]",
                \"overlay.png\": \"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASwAAAEsCAYAAAB5fY51AAAgAElEQVR4nOzdeXxU1f7/8e9f...rest of the base64 string"\}"
            ]
        }
    ]
}
```

### Caption Payload Example
```json
{
  "inputs": [
    {
      "name": "image",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"]
    },
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["<CAPTION>"]
    }
  ]
}
```

### Caption Response Example
```json
{
    "model_name": "florence-2-large",
    "model_version": null,
    "id": "9d5fc306-e186-4129-a149-de0859377e9f",
    "parameters": null,
    "outputs": [
        {
            "name": "output",
            "shape": [
                1
            ],
            "datatype": "BYTES",
            "parameters": null,
            "data": [
                "{\"a green volkswagen beetle parked in front of a yellow building\"}"
            ]
        }
    ]
}
```
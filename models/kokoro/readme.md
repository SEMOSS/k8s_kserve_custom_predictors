# Kokoro-82M TTS Model for KServe

This directory contains a KServe implementation of the Kokoro-82M text-to-speech model.

# Docker Build

```bash
# Build the model image (GPU version)
docker build -f models/kokoro/Dockerfile --build-arg MODE=gpu -t kokoro-predictor:gpu .

# Build the model image (CPU version)
docker build -f models/kokoro/Dockerfile --build-arg MODE=cpu -t kokoro-predictor:cpu .
```

## Base Image

This model defaults to the GPU version of the PyTorch KServe image (`cfg-ms-torch-gpu`).

```bash
# Run the model locally (GPU)
docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/kokoro-82m:/mnt/models kokoro-predictor:gpu

# Run the model locally (CPU)
docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/kokoro-82m:/mnt/models kokoro-predictor:cpu
```

## Python Packages Added
The following packages are added in the model-level Docker image:

- kokoro-tts - The main Kokoro TTS library
- soundfile - For audio file I/O
- librosa - For audio processing utilities

## Example Dev Endpoint
- "http://localhost:8080/v2/models/kokoro-82m/infer"

## Example Payload
```json
{
  "inputs": [
    {
      "name": "text",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["Hello, this is a test of the Kokoro text-to-speech system."]
    },
    {
      "name": "voice",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["af_bella"]
    },
    {
      "name": "speed",
      "datatype": "FP32",
      "shape": [1],
      "data": [1.0]
    }
  ]
}
```

## Example Response
```json
{
    "model_name": "kokoro-82m",
    "model_version": null,
    "id": "b9c75ae8-d092-414b-8653-642d3bfe1e24",
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
                {
                "audio": "data:audio/wav;base64,Uk....AD//wAA//8=", 
                "sample_rate": 24000, 
                "duration": 4.375, 
                "voice": "af_bella", 
                "speed": 1.0, 
                "text": "Hello, this is a test of the Kokoro text-to-speech system."
                }
            ]
        }
    ]
}
```

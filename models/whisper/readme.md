# Whisper Large v3 ASR Model for KServe

This directory contains a KServe implementation of OpenAI's Whisper Large v3 automatic speech recognition model.

## Docker Build

```bash
# Build the model image (GPU version)
docker build -f models/whisper/Dockerfile --build-arg MODE=gpu -t whisper-predictor:gpu .

# Build the model image (CPU version)
docker build -f models/whisper/Dockerfile --build-arg MODE=cpu -t whisper-predictor:cpu .
```

## Base Image

This model defaults to the GPU version of the PyTorch KServe image (`cfg-ms-torch-gpu`).

## Docker Run Commands

```bash
# Run the model locally (GPU)
docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/whisper-large-v3:/mnt/models whisper-predictor:gpu

# Run the model locally (CPU)
docker run -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/whisper-large-v3:/mnt/models whisper-predictor:cpu
```

## Python Packages Added

The following packages are added in the model-level Docker image:

- `openai-whisper` - The official OpenAI Whisper library
- `librosa` - For audio processing and format conversion
- `soundfile` - For audio file I/O operations

## Example Endpoints

### OpenAI Compatible Endpoint
- **Standard**: `POST http://localhost:8080/v1/audio/transcriptions`
- **Base64**: `POST http://localhost:8080/v1/audio/transcriptions/base64`

### KServe Native Endpoint
- **Inference**: `POST http://localhost:8080/v2/models/whisper-large-v3/infer`

## OpenAI SDK Usage

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy-key"
)

with open("audio.wav", "rb") as audio_file:
    transcription = client.audio.transcriptions.create(
        model="whisper-large-v3",
        file=audio_file,
        language="en",
        response_format="json"
    )
    
print(transcription.text)
```

## Example Payloads

### OpenAI Compatible API (Form Data)
```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@audio.wav" \
  -F "model=whisper-large-v3" \
  -F "language=en" \
  -F "response_format=json" \
  -F "temperature=0.0"
```

### Base64 API (JSON)
```json
{
  "file": "data:audio/wav;base64,UklGRiQAAABXQVZFZm10...",
  "model": "whisper-large-v3",
  "language": "en",
  "response_format": "json",
  "temperature": 0.0,
  "timestamp_granularities": ["segment"]
}
```

### KServe Native API (JSON)
```json
{
  "inputs": [
    {
      "name": "audio",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["data:audio/wav;base64,UklGRiQAAABXQVZFZm10..."]
    },
    {
      "name": "language",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["en"]
    },
    {
      "name": "response_format",
      "datatype": "BYTES",
      "shape": [1],
      "data": ["json"]
    },
    {
      "name": "temperature",
      "datatype": "FP32",
      "shape": [1],
      "data": [0.0]
    }
  ]
}
```

## Example Responses

### JSON Response Format
```json
{
  "text": "Hello, this is a test of the Whisper speech recognition system."
}
```

### JSON Response with Segments
```json
{
  "text": "Hello, this is a test of the Whisper speech recognition system.",
  "segments": [
    {
      "id": 0,
      "seek": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Hello, this is a test of the Whisper speech recognition system.",
      "tokens": [50364, 2425, 11, 341, 307, 257, 1500, 295, 264, 45437, 6218, 11150, 1185, 13, 50539],
      "temperature": 0.0,
      "avg_logprob": -0.15,
      "compression_ratio": 1.2,
      "no_speech_prob": 0.01
    }
  ]
}
```

### Text Response Format
```
Hello, this is a test of the Whisper speech recognition system.
```

### KServe Native Response
```json
{
  "model_name": "whisper-large-v3",
  "model_version": null,
  "id": "b9c75ae8-d092-414b-8653-642d3bfe1e24",
  "parameters": null,
  "outputs": [
    {
      "name": "output",
      "shape": [1],
      "datatype": "BYTES",
      "parameters": null,
      "data": [
        "{\"text\": \"Hello, this is a test of the Whisper speech recognition system.\"}"
      ]
    }
  ]
}
```

## Supported Parameters

- **model**: Model name (default: "whisper-large-v3")
- **language**: Source language code (optional, auto-detected if not provided)
- **prompt**: Context prompt to improve accuracy (optional)
- **response_format**: Response format - "json" or "text" (default: "json")
- **temperature**: Sampling temperature 0.0-1.0 (default: 0.0)
- **timestamp_granularities**: Array of ["segment", "word"] for detailed timestamps

## Supported Audio Formats

The model supports various audio formats through FFmpeg:
- WAV
- MP3
- MP4
- M4A
- FLAC
- OGG

Audio is automatically converted to 16kHz mono format required by Whisper.

## Notes

- Model files are automatically downloaded if not present in the mounted volume
- The model supports both CPU and GPU inference
- For production use, mount the model files to `/mnt/models/whisper-large-v3/` to avoid repeated downloads
- Base64 audio input should include the data URI prefix (e.g., "data:audio/wav;base64,...")
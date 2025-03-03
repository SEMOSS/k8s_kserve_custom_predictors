# Gliner Predictor

## Docker Commands

### Build Docker Image

```bash
# CPU
docker build --build-arg MODE=cpu -t gliner-predictor:cpu .
# GPU
docker build --build-arg MODE=gpu -t gliner-predictor:gpu .
```

### Run Docker Container

```bash
docker run -p 8080:8080 gliner-predictor:cpu
```
docker build -f models/phi-4/Dockerfile --build-arg MODE=gpu -t phi-4-predictor:gpu .

docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/phi-4:/mnt/models phi-4-predictor:gpu

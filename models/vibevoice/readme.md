docker build -f models/vibevoice/Dockerfile --build-arg MODE=gpu -t vibevoice:gpu .

docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/vibevoice-7b:/mnt/models vibevoice:gpu

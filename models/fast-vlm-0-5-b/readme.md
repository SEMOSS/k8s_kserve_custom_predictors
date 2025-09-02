docker build -f models/fast-vlm-0-5-b/Dockerfile --build-arg MODE=gpu -t fast-vlm-0-5-b-predictor:gpu .

docker run --gpus all -p 8080:8080 -v /var/lib/docker/volumes/model-volume/_data/fast-vlm-0-5-b:/mnt/models fast-vlm-0-5-b-predictor:gpu

ARG MODE=gpu
ARG BASE_IMAGE=cfg-ms-torch-gpu:latest

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY models/image_embed/model.py ./model.py

COPY common/kserve/torch /app/kserve-torch/
RUN pip install -e /app/kserve-torch/

ENV MODEL_NAME=image-embedding
ENV MODEL_ID=facebook/dinov2-large
ENV POOLING_STRATEGY=cls
ENV NORMALIZE_EMBEDDINGS=true
ENV PYTHONUNBUFFERED=1

RUN mkdir -p /mnt/models

EXPOSE 8080

RUN python -c "import kserve_torch; print(f'kserve_torch package found at {kserve_torch.__file__}')"

CMD python model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True
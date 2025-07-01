ARG MODE=gpu
ARG BASE_IMAGE=cfg-ms-torch-gpu:latest

FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    espeak-ng > /dev/null 2>&1 \
    && rm -rf /var/lib/apt/lists/*

COPY models/kokoro/model.py ./model.py

RUN poetry add kokoro soundfile librosa 

COPY common/kserve/torch /app/kserve-torch/
RUN pip install -e /app/kserve-torch/

ENV MODEL_NAME=kokoro-82m
ENV MODEL_ID=hexgrad/Kokoro-82M
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /mnt/models

ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface/transformers"

EXPOSE 8080

RUN python -c "import kserve_torch; print(f'kserve_torch package found at {kserve_torch.__file__}')"

CMD python model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True
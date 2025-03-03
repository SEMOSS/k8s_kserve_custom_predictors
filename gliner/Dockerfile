# ARG to select CPU or GPU
ARG MODE=cpu

FROM python:3.10-slim AS base-cpu
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS base-gpu
FROM base-${MODE} AS final

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN if [ "${MODE}" = "gpu" ]; then \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && ln -sf /usr/bin/pip3 /usr/bin/pip \
    && rm -rf /var/lib/apt/lists/*; \
    fi

WORKDIR /app

RUN pip install --no-cache-dir poetry && \
    poetry config virtualenvs.create false

COPY pyproject.toml ./pyproject.toml

RUN poetry install --no-interaction --no-ansi --no-root

RUN poetry source add --priority explicit pytorch_cpu https://download.pytorch.org/whl/cpu && \
    poetry add --source pytorch_cpu torch torchvision


COPY model.py /app/

ENV MODEL_NAME=gliner-multi-v2-1
ENV MODEL_ID=urchade/gliner_multi-v2.1

RUN mkdir -p /mnt/models

EXPOSE 8080

CMD python model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True
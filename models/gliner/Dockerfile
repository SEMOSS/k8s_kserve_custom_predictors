ARG MODE=cpu
ARG BASE_IMAGE=cfg-ms-torch-cpu:latest

FROM ${BASE_IMAGE}

WORKDIR /app

COPY models/gliner/model.py ./model.py

RUN poetry add gliner=>0.2.16

RUN poetry install --no-interaction --no-ansi --no-root

COPY common/kserve/torch /app/kserve-torch/
RUN pip install -e /app/kserve-torch/

ENV MODEL_NAME=gliner-multi-v2-1
ENV MODEL_ID=urchade/gliner_multi-v2.1
ENV PYTHONUNBUFFERED=1
RUN mkdir -p /mnt/models

EXPOSE 8080

RUN python -c "import kserve_torch; print(f'kserve_torch package found at {kserve_torch.__file__}')"


CMD python model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True
FROM busybox

ARG MODEL_PATH=model_files
ARG MODEL_NAME=default

LABEL model.name="${MODEL_NAME}"

RUN mkdir -p /models && chmod 775 /models

COPY ${MODEL_PATH}/ /models/
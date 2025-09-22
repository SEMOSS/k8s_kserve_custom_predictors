ARG MODE=gpu
ARG BASE_IMAGE=cfg-ms-torch-flash:latest
FROM ${BASE_IMAGE}

# System deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg libavcodec-extra \
 && rm -rf /var/lib/apt/lists/*

# Python deps (keep lean; no torchvision here)
RUN pip install \
    soundfile librosa scipy \
    diffusers ml-collections absl-py av numba llvmlite

# --- Clone + install community fork (keeps demo voices/assets in the image) ---
RUN git clone https://github.com/vibevoice-community/VibeVoice.git /tmp/VibeVoice \
 && cd /tmp/VibeVoice \
 && pip install -e . \
 && rm -rf /tmp/VibeVoice/.git

# Our server code
COPY models/vibevoice/model.py /app/model.py
COPY common/kserve/torch /app/kserve-torch/
RUN pip install -e /app/kserve-torch/

# Cache + model env
ENV MODEL_NAME=vibevoice-7b
ENV MODEL_ID=vibevoice/VibeVoice-7B
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface/transformers"
RUN mkdir -p ${HF_HOME} ${TRANSFORMERS_CACHE} /mnt/models && chmod -R 777 /app/.cache

# *** CRITICAL: disable flash-attn integration in transformers ***
# This prevents `transformers.modeling_utils` from importing flash_attn at all.
ENV TRANSFORMERS_NO_FLASH_ATTENTION=1
# Also disable xformers if present in base image
ENV XFORMERS_DISABLED=1

# Optional: ensure a clean environment without stray flash-attn wheels
# (If your base image already bundles a working flash-attn, you can omit this.)
RUN pip uninstall -y flash-attn flash_attn || true

EXPOSE 8080

# Sanity checks (won’t fail build if VibeVoice import warns)
RUN python -c "import kserve_torch; print(f'kserve_torch at {kserve_torch.__file__}')"
RUN python -c "import vibevoice; print('VibeVoice import OK') or None" || echo "VibeVoice import warning"

# NVIDIA recommends these flags; you’re already seeing the SHMEM note at runtime.
# Use them on `docker run` rather than here:
#   --ipc=host --ulimit memlock=-1 --ulimit stack=67108864

CMD python /app/model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True

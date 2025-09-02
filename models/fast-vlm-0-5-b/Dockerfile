ARG MODE=gpu
ARG BASE_IMAGE=cfg-ms-torch-gpu:latest

FROM ${BASE_IMAGE}

# Install system dependencies if needed
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the model implementation
COPY models/fast-vlm-0-5-b/model.py ./model.py

# Install additional dependencies for Fast-VLM
# Note: Updated transformers version for better FastVLM support
RUN poetry add \
    bitsandbytes \
    torch==2.6.0 \
    transformers==4.48.2 \
    accelerate==1.3.0 \
    soundfile==0.13.1 \
    pillow==11.1.0 \
    scipy==1.15.2 \
    torchvision==0.21.0 \
    backoff==2.2.1 \
    peft==0.13.2 \
    timm \
    requests

# Reinstall the kserve_torch package to ensure it's available
COPY common/kserve/torch /app/kserve-torch/
RUN pip install -e /app/kserve-torch/

# Set environment variables
ENV MODEL_NAME=fast-vlm-0-5-b
ENV MODEL_ID=apple/FastVLM-0.5B
ENV PYTHONUNBUFFERED=1

# Create model mount directory
RUN mkdir -p /mnt/models

# Set HuggingFace cache directories
ENV HF_HOME="/app/.cache/huggingface"
ENV TRANSFORMERS_CACHE="/app/.cache/huggingface/transformers"

# Ensure cache directories exist with appropriate permissions
RUN mkdir -p ${HF_HOME} ${TRANSFORMERS_CACHE} && chmod -R 777 /app/.cache

# Set CUDA environment variables for better memory management
ENV PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
ENV CUDA_LAUNCH_BLOCKING="1"

# FastVLM specific environment variables
ENV TOKENIZERS_PARALLELISM=false
ENV OMP_NUM_THREADS=1

EXPOSE 8080

# Verify the package is installed
RUN python -c "import kserve_torch; print(f'kserve_torch package found at {kserve_torch.__file__}')"

# Test that we can import transformers and torch
RUN python -c "import torch; import transformers; print(f'torch: {torch.__version__}, transformers: {transformers.__version__}')"

CMD python model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True
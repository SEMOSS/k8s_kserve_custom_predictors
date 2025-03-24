ARG MODE=gpu
ARG BASE_IMAGE=cfg-ms-torch-gpu:latest

FROM ${BASE_IMAGE}

# Install additional dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install diffusers and additional dependencies
RUN pip install diffusers==0.32.2 \
    safetensors==0.5.3 \
    xformers==0.0.29.post3 \
    --no-cache-dir

# Copy the model implementation
COPY models/image/model.py ./model.py

# Make sure the kserve-torch package is installed
COPY common/kserve/torch /app/kserve-torch/
RUN pip install -e /app/kserve-torch/

# Set model defaults
ENV MODEL_NAME=image-generation
ENV MODEL_ID=stable-diffusion-v1-5/stable-diffusion-v1-5
ENV DEFAULT_HEIGHT=512
ENV DEFAULT_WIDTH=512
ENV DEFAULT_NUM_INFERENCE_STEPS=30
ENV DEFAULT_GUIDANCE_SCALE=7.5
ENV DEFAULT_NEGATIVE_PROMPT="low quality, blurry, distorted, disfigured, text, watermark"
ENV PYTHONUNBUFFERED=1

# Create directory for model files
RUN mkdir -p /mnt/models

# Expose the KServe port
EXPOSE 8080

# Verify the package is installed correctly
RUN python -c "import kserve_torch; print(f'kserve_torch package found at {kserve_torch.__file__}')"

# Run the model server
CMD python model.py --model_name=${MODEL_NAME} --http_port=8080 --workers=1 --enable_docs_url=True
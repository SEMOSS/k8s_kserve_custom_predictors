# CFG Model Service KServe PyTorch Image with GPU Support and Flash Attention

## Docker Build
Run this command from the project root directory to build the image:
`docker build -f common/kserve/torch/gpu-flash/Dockerfile.torch.gpu-flash -t cfg-ms-torch-gpu-flash .`

## Specs

- Base Image `cfg-ms-base-gpu`

- Python: 3.10.6

## Python Packages
- `torchvision==0.21.0`
- `torchaudio==2.6.0`
- `accelerate>=1.4.0`
- `paddleocr>=2.9.1`
- `transformers>=4.49.0`
- `flash-attn>=2.6.0`
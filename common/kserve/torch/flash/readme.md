# CFG Model Service KServe PyTorch Image with GPU Support and Flash Attention

## Docker Build
Run this command from the project root directory to build the image:
`docker build -f common/kserve/torch/flash/Dockerfile.torch.flash -t cfg-ms-torch-flash .`

## Inspect

`docker run -it --name cfg-ms-torch-flash-container cfg-ms-torch-flash /bin/bash`

## Specs

- Base Image `cfg-ms-base-flash`

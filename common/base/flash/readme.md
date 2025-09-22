# CFG Model Service Base Image with GPU and Flash Attention Support

## Docker Build

`docker build -f common/base/flash/Dockerfile.flash -t cfg-ms-base-flash .`

## Inspect

`docker run -it --name cfg-ms-base-flash-container cfg-ms-base-flash /bin/bash`

## Specs

- Base Image `nvcr.io/nvidia/pytorch:24.12-py3`

- `torch`: 2.6.0a0+df5bbc09d1.nv24.12
- `torchvision`: 0.20.0a0
- `flash_attn`: 2.4.2

# KServe Custom Predictors
This repository implements a 3-tiered Docker image architecture for deploying ML models with KServe.

## Architecture Overview

```
┌────────────────────┐
│                    │
│   Model Layer      │ (e.g., GLiNER, BERT, etc.)
│   (Level 3)        │ Model-specific implementation
│                    │
├────────────────────┤
│                    │
│   Framework Layer  │ (e.g., PyTorch, TensorFlow)
│   (Level 2)        │ Framework-specific base classes
│                    │
├────────────────────┤
│                    │
│   Base Layer       │ (CPU or GPU)
│   (Level 1)        │ Common dependencies and utilities
│                    │
└────────────────────┘
```

## Layer Descriptions

### Level 1: Base Layer 
The foundation of our image stack provides:
- Python runtime environment
- Common system dependencies
- Basic utilities and libraries
- CPU or GPU support via separate images

**Location:** `common/base/{cpu,gpu}/`

### Level 2: Framework Layer
Built on top of the base layer, specialized for specific ML frameworks:
- Framework-specific dependencies (e.g., PyTorch, TensorFlow)
- Abstracted KServe interfaces as base classes
- Utilities for model loading and inference

**Location:** `common/kserve/{torch,tensorflow}/{cpu,gpu}/`

### Level 3: Model Layer
The application-specific layer that implements:
- Specific model implementations
- Model-specific pre/post-processing logic
- Runtime parameters and configurations

**Location:** `models/{model_name}/`

## Using the Architecture

### Building Images
Images should be built in order from base to model layer:

```bash
# 1. Build the base layer
docker build -f common/base/gpu/Dockerfile.gpu -t cfg-ms-base-gpu .
# OR
docker build -f common/base/cpu/Dockerfile.cpu -t cfg-ms-base-cpu .

# 2. Build the framework layer
docker build -f common/kserve/torch/gpu/Dockerfile.torch.gpu -t torch-gpu:latest .
# OR
docker build -f common/kserve/torch/cpu/Dockerfile.torch.cpu -t cfg-ms-torch-cpu .

# 3. Build the model layer
docker build -f models/gliner/Dockerfile --build-arg MODE=gpu -t gliner-predictor:gpu .
# OR
docker build -f models/gliner/Dockerfile --build-arg MODE=cpu -t gliner-predictor:cpu .
```

### Creating a New Model
To create a new model implementation:

1. Create a new directory under `models/`
2. Implement a class that inherits from the appropriate framework base class
3. Implement the required `load()` and `process_request()` methods
4. Add a simple entry point using the base class's `serve()` method
5. Create a Dockerfile that builds upon the appropriate framework image
6. Add a readme file with the following info:

    - a. Docker build commands
    - b. Which base image it defaults to (CPU or GPU)
    - c. Docker run commands
    - d. The Python packages added in the model level Docker image
    - f. Example endpoint
    - g. Example payload
    - h. Example response

## Package Structure

The `kserve_torch` package (and similar framework packages) provides base classes to standardize model implementations:

```python
# Example model implementation
from kserve_torch import BaseTorchModel # This is imported as a pip packaged installed in the layer 2 image

class MyModel(BaseTorchModel):
    def load(self):
        # Model-specific loading code
        
    async def process_request(self, request):
        # Model-specific inference code
        
if __name__ == "__main__":
    BaseTorchModel.serve(model_class=MyModel)
```

## CICD
Docker images are built and pushed when changes are detected at any of the image levels on merges into the `main` branch.

## OCI Images
See the [OCI_README.md](./oci/OCI_README.md) for instructions on how to build and run OCI images for the models.

## Contributing
Local development should be done using Docker due to the dependency architecture.
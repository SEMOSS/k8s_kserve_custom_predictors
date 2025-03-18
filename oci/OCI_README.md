# OCI Images

## Download Model Files

1. Create a new directory for the model and name it by the model path safe short name
2. In the OCI directory run `uv venv` then `uv pip install -r requirements.txt`
3. Set your Python interpreter to the new virtual environment
4. Update the download script to include the model short name and the model repo ID from HF
5. Run the script with `python download.py` to download the model files into the new directory

## Fill out the .env file
1. Use the `.env.example` file as a template
2. Fill in the required values

## Run the build script
1. Run the build-oci-models.cmd with the name of the model directory as the first argument (e.g. `.\build-oci-models.cmd florence-2-large`)
2. Verify that the model built in the repository under `genai/cfg-ms-models/oci/<model_short_name>`


## Building and Running Local OCI Images
1. Ensure you have Docker installed and running
2. Build the OCI image using the following command from the /oci directory
```cmd
docker build --build-arg MODEL_PATH=florence-2-large/model_files --build-arg MODEL_NAME=florence-2-large -t local/oci-model:florence-2-large -f Dockerfile .
```
3. Run the OCI image using the command
```cmd
docker run -it --entrypoint sh --name florence-2-large-oci local/oci-model:florence-2-large
```
4. This leaves the container running which allows you to inspect that the model files are present in the `/mnt/models` directory

## Notes
- The model files are not committed to this repository
- Refer to the [KServe OCI documentation](https://kserve.github.io/website/latest/modelserving/storage/oci/) for why we use OCI images. 
from huggingface_hub import snapshot_download


def download_model_files():

    model_repo_id = "openai/whisper-large-v3"

    short_name = "whisper-large-v3"

    local_model_dir = f"./oci/{short_name}"

    snapshot_download(repo_id=model_repo_id, local_dir=local_model_dir)


if __name__ == "__main__":
    download_model_files()

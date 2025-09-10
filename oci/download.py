from huggingface_hub import snapshot_download


def download_model_files():

    model_repo_id = "microsoft/Florence-2-large"

    short_name = "florence-2-large"

    local_model_dir = f"./oci/{short_name}"

    snapshot_download(repo_id=model_repo_id, local_dir=local_model_dir)


if __name__ == "__main__":
    download_model_files()

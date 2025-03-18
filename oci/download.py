from huggingface_hub import snapshot_download


def download_model_files():

    model_repo_id = "urchade/gliner_multi-v2.1"

    short_name = "gliner_multi-v2-1"

    local_model_dir = f"./oci/{short_name}/model_files"

    snapshot_download(repo_id=model_repo_id, local_dir=local_model_dir)


if __name__ == "__main__":
    download_model_files()

from huggingface_hub import snapshot_download

# nombre exacto del repo
MODEL_REPO = "meta-llama/Llama-3.2-3B"

# carpeta donde lo guardarás
LOCAL_DIR = "./models/llama32-3b"

def download_model():
    print(f"\nDescargando {MODEL_REPO} ...\n")

    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False
    )

    print(f"\nModelo descargado en: {LOCAL_DIR}\n")

if __name__ == "__main__":
    download_model()

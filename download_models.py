import os
from huggingface_hub import snapshot_download

MODELS = {
    "SDXL": "stabilityai/stable-diffusion-xl-base-1.0",
    "IP-Adapter": "h94/IP-Adapter-FaceID",
    "VAE": "madebyollin/sdxl-vae-fp16-fix"
}

def download_models():
    os.makedirs("model_cache", exist_ok=True)
    for name, repo in MODELS.items():
        print(f"Downloading {name}...")
        snapshot_download(
            repo,
            cache_dir="model_cache",
            local_dir=os.path.join("model_cache", repo.split('/')[-1]),
            local_dir_use_symlinks=False
        )
    print("All models downloaded!")

if __name__ == "__main__":
    download_models()
"""Download the pinned local GGUF model after checking its size and free disk."""

import hashlib
import os
import shutil
from pathlib import Path

from huggingface_hub import get_hf_file_metadata, hf_hub_download, hf_hub_url


DEFAULT_REPO = "bartowski/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_REVISION = "9eadc66189c7641e1ddd226b8267a9119b2ce2d4"
DEFAULT_FILE = "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
MAX_MODEL_BYTES = 2_000_000_000
MINIMUM_FREE_BYTES_AFTER_DOWNLOAD = 500_000_000


def download_model() -> Path:
    repo = os.getenv("LOCAL_MODEL_REPO", DEFAULT_REPO)
    revision = os.getenv("LOCAL_MODEL_REVISION", DEFAULT_REVISION)
    filename = os.getenv("LOCAL_MODEL_FILE", DEFAULT_FILE)
    model_path = Path(
        os.getenv("LOCAL_MODEL_PATH", f"models/{DEFAULT_FILE}")
    ).expanduser()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    if model_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing model file: {model_path}. "
            "Move it aside or configure another LOCAL_MODEL_PATH."
        )

    url = hf_hub_url(repo_id=repo, filename=filename, revision=revision)
    metadata = get_hf_file_metadata(url)
    size = metadata.size
    if size is None:
        raise RuntimeError("The model server did not provide a file size; download cancelled.")
    if size > MAX_MODEL_BYTES:
        raise RuntimeError(
            f"Refusing to download {size:,} bytes; safety limit is {MAX_MODEL_BYTES:,} bytes."
        )
    free = shutil.disk_usage(model_path.parent).free
    if free < size + MINIMUM_FREE_BYTES_AFTER_DOWNLOAD:
        raise OSError(
            f"Not enough free space: model needs {size:,} bytes plus "
            f"{MINIMUM_FREE_BYTES_AFTER_DOWNLOAD:,} bytes of headroom; {free:,} available."
        )

    print(
        f"Downloading {repo}/{filename} at revision {revision} "
        f"({size / 1_000_000:.1f} MB) to {model_path}."
    )
    downloaded = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            revision=revision,
            local_dir=str(model_path.parent),
        )
    )
    if downloaded.resolve() != model_path.resolve():
        raise RuntimeError(
            f"Downloaded file path {downloaded} differs from configured model path {model_path}."
        )
    digest = hashlib.sha256(downloaded.read_bytes()).hexdigest()
    print(f"Downloaded {downloaded} ({size:,} bytes, SHA-256 {digest}).")
    return downloaded


if __name__ == "__main__":
    download_model()
"""Resolve pinned Hugging Face snapshots from the local cache."""
from pathlib import Path


def require_cached_revision(model_name: str, revision: str) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required for pinned V4 model loading") from exc
    snapshot = Path(snapshot_download(model_name, revision=revision, local_files_only=True))
    if snapshot.name != revision:
        raise ValueError(f"Expected cached model revision {revision}, resolved {snapshot.name}")
    return snapshot

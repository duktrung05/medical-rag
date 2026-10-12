"""Read the isolated V2 pilot configuration."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/vibiomir/retrieval_v2.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config["retrieval"]["fusion_method"] != "weighted_rrf":
        raise ValueError("V2 pilot supports weighted_rrf only")
    if config["dense"]["metric"] != "inner_product" or not config["dense"]["normalized"]:
        raise ValueError("BGE-M3 pilot requires normalized inner-product vectors")
    return config


def cached_model_path(model_name: str, revision: str) -> Path:
    namespace, name = model_name.split("/", 1)
    return ROOT / ".cache/huggingface/hub" / f"models--{namespace}--{name}" / "snapshots" / revision

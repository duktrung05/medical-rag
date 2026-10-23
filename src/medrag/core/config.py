from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    else:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value.get("corpus_version"):
        raise ValueError("V4 configuration requires corpus_version")
    return value


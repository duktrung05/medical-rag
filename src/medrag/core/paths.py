"""Locate repository-mounted configuration in source and installed runtimes."""
from __future__ import annotations

import os
from pathlib import Path


def project_root() -> Path:
    candidates = [Path(os.environ["MEDRAG_ROOT"])] if os.environ.get("MEDRAG_ROOT") else []
    candidates.extend([Path.cwd(), Path(__file__).resolve().parents[3]])
    for root in candidates:
        if (root / "configs/v4_core.yaml").is_file():
            return root.resolve()
    raise FileNotFoundError("V4 config not found; set MEDRAG_ROOT to the project directory")


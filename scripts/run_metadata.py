"""Small helpers for reproducible offline analysis artifacts."""

import hashlib
import json
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_commit(root: Path | None = None) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def write_run_manifest(output: Path, *, task: str, inputs: dict[str, Path], parameters: dict | None = None,
                       root: Path | None = None) -> dict:
    source_root = root or Path.cwd()
    source_files = sorted(
        path for tree in (source_root / "src", source_root / "scripts", source_root / "configs")
        for path in tree.rglob("*") if path.is_file() and path.suffix in (".py", ".yaml", ".yml")
    )
    manifest = {
        "task": task,
        "source_commit_sha": source_commit(source_root),
        "source_files_sha256": {str(path.relative_to(source_root)): sha256(path) for path in source_files},
        "input_sha256": {name: sha256(path) for name, path in inputs.items()},
        "parameters": parameters or {},
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return manifest

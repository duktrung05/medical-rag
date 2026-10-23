"""Validate and record the pinned Tesla T4 runtime before A0_T4_V1 scoring."""
from __future__ import annotations

import argparse
import json
import subprocess
from importlib.metadata import version
from pathlib import Path

from huggingface_hub import snapshot_download

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import external_lock
from src.medrag.data.phase2_batch import sha256_file

EXPECTED = {"torch": "2.6.0+cu124", "sentence-transformers": "4.1.0",
            "transformers": "4.51.3", "huggingface-hub": "0.36.2",
            "tokenizers": "0.21.4"}


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def main(output: Path) -> dict:
    output = output.resolve()
    lock = external_lock(output, "full")
    for package, expected in EXPECTED.items():
        if version(package) != expected:
            raise ValueError(f"Pinned runtime differs: {package}={version(package)} != {expected}")
    import torch

    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or "T4" not in torch.cuda.get_device_name(0)
            or torch.version.cuda != "12.4"):
        raise ValueError("A0_T4_V1 requires exactly one Tesla T4 with CUDA 12.4")
    contract = lock["score_contract"]
    snapshot = Path(snapshot_download(repo_id=contract["model"],
                                      revision=contract["revision"]))
    if snapshot.name != contract["revision"]:
        raise ValueError("Resolved model revision differs")
    snapshot_hashes = {str(path.relative_to(snapshot)): sha256_file(path)
                       for path in sorted(snapshot.rglob("*")) if path.is_file()}
    if not any(name.endswith((".safetensors", ".bin")) for name in snapshot_hashes):
        raise ValueError("Model snapshot contains no weights")
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total",
         "--format=csv,noheader,nounits"], capture_output=True, text=True,
        check=True, timeout=15).stdout.strip()
    report = {
        "format": "medrag-v4-a0-t4-v1-runtime-v1",
        "status": "READY_TO_SCORE",
        "candidate": "A0_T4_V1",
        "gpu": torch.cuda.get_device_name(0),
        "cuda": torch.version.cuda,
        "nvidia_smi": smi,
        "runtime_versions": {name: version(name) for name in EXPECTED},
        "model": contract["model"], "model_revision": contract["revision"],
        "max_length": contract["max_length"], "precision": contract["precision"],
        "model_snapshot_files_sha256": snapshot_hashes,
        "lock_sha256": sha256_file(output / "lock.json"),
        "pair_manifest_sha256": sha256_file(output / "pair_manifest.json"),
    }
    path = output / "provenance/t4_runtime.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes_checked(path, canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(main(args.output), ensure_ascii=False, indent=2))

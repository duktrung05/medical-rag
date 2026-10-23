"""Package the six immutable P5.2 pilot-live pair shards for a fresh Colab scorer."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sqlite3
import zipfile
from pathlib import Path

from scripts.v4.run_full_baseline import _write_bytes_checked
from scripts.v4.run_p5_2_split import (PILOT_IDS, ScoreCache, external_lock,
                                      read, validate_pair_shard)
from src.medrag.data.phase2_batch import sha256_file

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022"
OUTPUT = ROOT / "outputs/v4/p5_3a_colab_canary_20261022"
INPUT_DIR = Path("outputs/v4/p5_3a_canary_input")
EXPECTED_PAYLOAD_SHA256 = "b7802660f332f9a2a773354a30d4983876745e6b7dd63386143b5d9d073906e6"
DEPENDENCIES = ("sentence-transformers==4.1.0\ntransformers==4.51.3\n"
                "huggingface-hub==0.36.2\nnumpy==2.4.6\npyarrow==21.0.0\n"
                "safetensors==0.8.0\ntokenizers==0.21.4\n")
SECRET_PATTERNS = (rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
                   rb"AKIA[0-9A-Z]{16}", rb"AIza[0-9A-Za-z_-]{35}",
                   rb"(?<![A-Za-z0-9])hf_[A-Za-z0-9]{30,}",
                   rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9]{20,}")
SCORER_MODULES = (
    "src/medrag/__init__.py",
    "src/medrag/adapters/__init__.py",
    "src/medrag/adapters/validation.py",
    "src/medrag/adapters/vibiomir_submission.py",
    "src/medrag/core/__init__.py",
    "src/medrag/core/config.py",
    "src/medrag/core/contracts.py",
    "src/medrag/core/disk_pipeline.py",
    "src/medrag/core/model_revision.py",
    "src/medrag/core/pipeline.py",
    "src/medrag/data/__init__.py",
    "src/medrag/data/disk_store.py",
    "src/medrag/data/phase2/__init__.py",
    "src/medrag/data/phase2/provenance.py",
    "src/medrag/data/phase2/schema.py",
    "src/medrag/data/phase2_batch.py",
    "src/medrag/evidence/__init__.py",
    "src/medrag/evidence/context.py",
    "src/medrag/evidence/selector.py",
    "src/medrag/evidence/verifier.py",
    "src/medrag/ranking/__init__.py",
    "src/medrag/ranking/reranker.py",
    "src/medrag/ranking/score_cache.py",
    "src/medrag/retrieval/__init__.py",
    "src/medrag/retrieval/core.py",
    "src/medrag/retrieval/disk_bm25.py",
    "src/medrag/retrieval/disk_dense.py",
)


def canonical(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def archive_entry(name: str, payload: bytes, bundle: zipfile.ZipFile) -> None:
    info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    bundle.writestr(info, payload, compresslevel=6)


def main(source: Path, output: Path) -> dict:
    source, output = source.resolve(), output.resolve()
    lock = external_lock(source, "pilot-live")
    if tuple(lock["query_ids"]) != PILOT_IDS:
        raise ValueError("Canary requires the six locked P2-A0 pilot IDs")
    run = read(source / "run_manifest.json")
    if (run.get("status") != "PASS" or run.get("mode") != "pilot-live"
            or run.get("pair_count") != 1200 or run.get("query_count") != 6
            or run.get("predictions_sha256") != EXPECTED_PAYLOAD_SHA256):
        raise ValueError("Source live pilot did not pass exact P2-A0 parity")
    for name, expected in lock["implementation_sha256"].items():
        if sha256_file(ROOT / name) != expected:
            raise ValueError(f"Frozen scorer implementation changed: {name}")
    cache = ScoreCache(source / "key_contract.sqlite", lock["score_contract"])
    with sqlite3.connect(source / "key_contract.sqlite") as db:
        if db.execute("SELECT count(*) FROM scores").fetchone()[0] != 0:
            raise ValueError("Input key contract must contain zero scores")
    for qid in PILOT_IDS:
        shard = read(source / "pairs" / f"{qid}.json")
        validate_pair_shard(shard, lock, qid, shard["query"], cache)

    files: dict[str, bytes] = {}
    for name in ("lock.json", "pair_manifest.json", "key_contract.sqlite"):
        files[str(INPUT_DIR / name)] = (source / name).read_bytes()
    for qid in PILOT_IDS:
        files[str(INPUT_DIR / "pairs" / f"{qid}.json")] = (
            source / "pairs" / f"{qid}.json").read_bytes()
    code_files = [ROOT / name for name in SCORER_MODULES]
    code_files += [ROOT / "scripts/v4/run_full_baseline.py",
                   ROOT / "scripts/v4/run_p5_2_split.py",
                   ROOT / "scripts/v4/p5_3a_colab_orchestrator.py",
                   ROOT / "configs/v4_submission_p2_a0.yaml"]
    for path in code_files:
        files[str(path.relative_to(ROOT))] = path.read_bytes()
    files["p5_3a_requirements.txt"] = DEPENDENCIES.encode("ascii")
    if any("scores" in Path(name).parts or name.endswith("scores.sqlite") for name in files):
        raise ValueError("Fresh GPU bundle must not contain score artifacts")
    forbidden = ("corpus.sqlite", "documents.sqlite", "passages.sqlite", "vectors.npy")
    if any(Path(name).name in forbidden or "dense" in Path(name).parts for name in files):
        raise ValueError("Bundle contains a corpus or retrieval index")
    for name, payload in files.items():
        if any(re.search(pattern, payload) for pattern in SECRET_PATTERNS):
            raise ValueError(f"Bundle secret-pattern scan failed: {name}")
    manifest = {"format": "medrag-v4-p5-3a-colab-input-v1",
                "source_mode": "pilot-live", "query_ids": list(PILOT_IDS),
                "query_count": 6, "pair_count": 1200,
                "input_dir": str(INPUT_DIR),
                "model": lock["score_contract"]["model"],
                "model_revision": lock["score_contract"]["revision"],
                "max_length": lock["score_contract"]["max_length"],
                "precision": lock["score_contract"]["precision"],
                "pair_format": lock["score_contract"]["pair_format"],
                "tokenizer_contract": {"implementation": "BGEReranker in src/medrag/ranking/reranker.py",
                                       "class": "AutoTokenizer.from_pretrained",
                                       "revision": lock["score_contract"]["revision"],
                                       "max_length": 512,
                                       "predict_truncation": "CrossEncoder max_length=512",
                                       "diagnostic_truncation": False},
                "torch_cuda_wheel": "torch==2.6.0+cu124",
                "score_contract_fingerprint": cache.fingerprint,
                "source_payload_sha256": EXPECTED_PAYLOAD_SHA256,
                "source_run_manifest_sha256": sha256_file(source / "run_manifest.json"),
                "files_sha256": {name: hashlib.sha256(payload).hexdigest()
                                 for name, payload in sorted(files.items())}}
    files["p5_3a_input_manifest.json"] = canonical(manifest)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", allowZip64=True) as bundle:
        for name, payload in sorted(files.items()):
            archive_entry(name, payload, bundle)
    output.mkdir(parents=True, exist_ok=True)
    archive = output / "p5_3a_colab_bundle.zip"
    _write_bytes_checked(archive, stream.getvalue())
    _write_bytes_checked(output / "input_manifest.json", canonical(manifest))
    handoff = {"format": "medrag-v4-p5-3a-handoff-v1",
               "archive": archive.name, "archive_sha256": sha256_file(archive),
               "archive_bytes": archive.stat().st_size,
               "input_manifest_sha256": sha256_file(output / "input_manifest.json"),
               "query_count": 6, "pair_count": 1200,
               "model_revision": manifest["model_revision"],
               "fresh_gpu_inference_required": True}
    _write_bytes_checked(output / "handoff.json", canonical(handoff))
    return handoff


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(main(args.source, args.output), indent=2))

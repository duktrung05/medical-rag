"""Run the pinned V4 Phase 2 subset through the active CLI/API pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.core.paths import project_root
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.prepared import load_prepared


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(snapshot: Path, config_path: Path, output: Path, limit: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if limit <= 0:
        raise ValueError("limit must be positive")
    root = project_root()
    config = load_config(config_path)
    config_sha256 = _sha(config_path)
    implementation = {
        name: _sha(root / name) for name in (
            "src/medrag/core/pipeline.py",
            "src/medrag/core/factory.py",
            "src/medrag/evidence/selector.py",
            "src/medrag/retrieval/core.py",
            "src/medrag/adapters/vibiomir_submission.py",
        )
    }
    registry = CorpusRegistry.load(root / config["registry"])
    queries = pq.read_table(snapshot / "queries.parquet", columns=["id", "query"]).to_pylist()
    if len({int(row["id"]) for row in queries}) != len(queries):
        raise ValueError("Duplicate query IDs in snapshot")
    pipeline = load_prepared(snapshot, registry, config=config)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    staging.mkdir()
    raw_path = staging / "results.jsonl"
    diagnostics_path = staging / "diagnostics.jsonl"
    predictions = []
    with raw_path.open("w", encoding="utf-8") as stream, diagnostics_path.open("w", encoding="utf-8") as diag_stream:
        for row in queries:
            qid = int(row["id"])
            result, diagnostics = pipeline.retrieve_with_diagnostics(str(row["query"]), limit,
                                                                     include_candidates=True)
            stream.write(json.dumps({"id": qid, "query": row["query"], "results": result},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            diag_stream.write(json.dumps({"id": qid, **diagnostics}, allow_nan=False) + "\n")
            predictions.extend(records([qid], {qid: result}))
    payload = json.dumps(predictions, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    (staging / "predictions.json").write_text(payload + "\n", encoding="utf-8")
    manifest = {
        "status": "complete",
        "scope": "locked_v4_subset",
        "snapshot_manifest_sha256": _sha(snapshot / "manifest.json"),
        "config_sha256": config_sha256,
        "config": config,
        "implementation_sha256": implementation,
        "query_count": len(queries),
        "source_count": len(pipeline.sources),
        "passage_count": len(pipeline.passages),
        "result_limit": limit,
        "results_sha256": _sha(raw_path),
        "diagnostics_sha256": _sha(diagnostics_path),
        "canonical_predictions_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "query_ids": [int(row["id"]) for row in queries],
    }
    (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()
    print(json.dumps(run(args.snapshot.resolve(), args.config.resolve(), args.output.resolve(),
                         args.limit), indent=2))

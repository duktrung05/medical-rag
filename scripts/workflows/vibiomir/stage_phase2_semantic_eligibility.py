"""Materialize a Phase 2-only semantic eligibility ledger without changing Phase 1."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("review_case", pa.string()),
    ("reason", pa.string()),
    ("staging_decision", pa.string()),
])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def materialize(config_path: Path, pilot_manifest: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("phase1_mutation_allowed") is not False:
        raise ValueError("Staging policy must explicitly forbid Phase 1 mutation")
    pilot = pq.read_table(pilot_manifest).to_pylist()
    pilot_ids = {int(row["doc_id"]) for row in pilot}
    documents = config.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Staging document list is empty")
    rows = [{
        "doc_id": int(row["doc_id"]),
        "review_case": row["review_case"],
        "reason": row["reason"],
        "staging_decision": "EXCLUDE_RETRIEVAL",
    } for row in documents]
    ids = [row["doc_id"] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= pilot_ids:
        raise ValueError("Staging IDs are duplicate or outside the locked pilot")
    output.mkdir(parents=True)
    table_path = output / "semantic_eligibility.parquet"
    with tempfile.NamedTemporaryFile(dir=output, suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temporary, compression="zstd")
        os.replace(temporary, table_path)
    finally:
        temporary.unlink(missing_ok=True)
    manifest = {
        "status": "staging_complete",
        "format": config["format"],
        "policy": config["policy"],
        "phase1_mutated": False,
        "documents": len(rows),
        "config_sha256": _sha256(config_path),
        "pilot_manifest_sha256": _sha256(pilot_manifest),
        "ledger_sha256": _sha256(table_path),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pilot-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = materialize(args.config, args.pilot_manifest, args.output)
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Semantic staging failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

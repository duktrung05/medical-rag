"""Sample pinned extracted Parquet by remote row groups without downloading 4.4 GB."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.retrieval_v2.settings import DEFAULT_CONFIG, load_config

REVISION = "bcb5a08fcff00385d7dabd0ac1807441630890c1"
EXPECTED_SOURCE_SHA256 = "7580d46bb839ac694312a7a4e24b7afc00a052cdf64da254a4f11edd4ee98be8"
REPO_PATH = f"datasets/duktrung/med-rag@{REVISION}/data/vibiomir/chunks_crawl_20261007.parquet"
ROOT = Path(__file__).resolve().parents[3]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(output: Path, per_group: int = 64, local_source: Path | None = None) -> dict:
    if not 1 <= per_group <= 512:
        raise ValueError("per_group must be 1..512")
    output.mkdir(parents=True, exist_ok=True)
    pilot_path = output / "pilot_source.parquet"
    if pilot_path.exists():
        raise FileExistsError(pilot_path)
    if local_source:
        source_hash = sha(local_source)
        if source_hash != EXPECTED_SOURCE_SHA256:
            raise ValueError("local extracted Parquet checksum does not match pinned corpus")
        source_file = local_source.open("rb")
        provenance = "local_verified_sha256"
    else:
        from huggingface_hub import HfFileSystem
        source_file = HfFileSystem().open(REPO_PATH, "rb", block_size=1024 * 1024)
        source_hash = EXPECTED_SOURCE_SHA256
        provenance = "remote_pinned_revision_expected_sha256_not_recomputed"
    try:
        parquet = pq.ParquetFile(source_file)
        if parquet.metadata.num_rows != 20523261:
            raise ValueError("source row count differs from audited snapshot")
        groups = sorted({0, parquet.metadata.num_row_groups // 2, parquet.metadata.num_row_groups - 1})
        sampled = []
        for group in groups:
            table = parquet.read_row_group(group)
            indices = np.linspace(0, table.num_rows - 1, min(per_group, table.num_rows), dtype=np.int64)
            doc_ids = table.take(pa.array(indices)).column("doc_id").unique()
            sampled.append(table.filter(pc.is_in(table.column("doc_id"), value_set=doc_ids)))
            del table
        sample = pa.concat_tables(sampled)
    finally:
        source_file.close()
    pq.write_table(sample, pilot_path, compression="zstd")
    links = ROOT / "data/raw/vibiomir/links_corpus.parquet"
    queries = ROOT / "data/raw/vibiomir/query.parquet"
    link_ids = pq.read_table(links, columns=["id"]).column("id").to_numpy()
    sample_ids = np.asarray(sample.column("doc_id").to_numpy())
    missing = sorted(set(sample_ids[~np.isin(sample_ids, link_ids)]))
    records = sample.to_pylist()
    keys = [(x["doc_id"], x["chunk_index"]) for x in records]
    lengths = [len((x.get("text") or "").split()) for x in records]
    counts = {}
    for doc_id, _ in keys:
        counts[doc_id] = counts.get(doc_id, 0) + 1
    report = {
        "source_revision": REVISION, "source_expected_sha256": source_hash,
        "checksum_status": provenance, "source_rows": parquet.metadata.num_rows,
        "source_row_groups": parquet.metadata.num_row_groups, "sample_row_groups": groups,
        "sample_rows": len(records), "sample_sha256": sha(pilot_path),
        "query_rows": pq.ParquetFile(queries).metadata.num_rows, "query_sha256": sha(queries),
        "mapping_rows": len(link_ids), "mapping_sha256": sha(links),
        "missing_sample_doc_ids": missing, "duplicate_sample_keys": len(keys) - len(set(keys)),
        "empty_sample_texts": sum(not (x.get("text") or "").strip() for x in records),
        "sample_unique_documents": len(counts), "sample_max_chunks_per_doc": max(counts.values()),
        "sample_word_length_p10_p50_p90": [float(x) for x in np.percentile(lengths, [10, 50, 90])],
        "sample_late_chunks_gt6": sum(x["chunk_index"] > 6 for x in records),
        "scope": "stratified row-group sample only; no full-corpus uniqueness or content audit",
    }
    (output / "V2_CORPUS_MANIFEST.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    import psutil
    env = {"python": platform.python_version(), "cpu_count": psutil.cpu_count(),
           "ram_total_bytes": psutil.virtual_memory().total,
           "ram_available_bytes": psutil.virtual_memory().available,
           "disk_free_bytes": psutil.disk_usage(str(output)).free}
    (output / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/retrieval_v2/pilot")
    ap.add_argument("--per-group", type=int)
    ap.add_argument("--local-source", type=Path)
    args = ap.parse_args()
    config = load_config(args.config)
    print(json.dumps(run(args.output, args.per_group or config["corpus"]["pilot_rows_per_group"],
                         args.local_source), ensure_ascii=False, indent=2))

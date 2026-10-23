"""Freeze disjoint unlabeled development/holdout query IDs outside the six P0 pilots."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
PILOT = {53, 130, 380, 754, 771, 1144}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(output: Path, development_count: int, holdout_count: int) -> None:
    if development_count <= 0 or holdout_count <= 0 or output.exists():
        raise ValueError("Need positive split sizes and a fresh output directory")
    migration = json.loads((ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
                           .read_text(encoding="utf-8"))
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query file fingerprint changed")
    table = pq.read_table(query_path, columns=["id", "query"]).to_pylist()
    pool = [(int(row["id"]), str(row["query"])) for row in table
            if int(row["id"]) not in PILOT]
    if len(pool) != len({qid for qid, _ in pool}) or len(pool) < development_count + holdout_count:
        raise ValueError("Insufficient distinct non-pilot queries")
    # Balance approximate short/medium/long query groups without looking at judgments.
    lengths = sorted(len(query) for _, query in pool)
    cut1, cut2 = lengths[len(lengths) // 3], lengths[2 * len(lengths) // 3]
    groups = {0: [], 1: [], 2: []}
    for qid, query in pool:
        bucket = 0 if len(query) <= cut1 else 1 if len(query) <= cut2 else 2
        key = hashlib.sha256(f"p3-dev-holdout-v1:{qid}".encode()).hexdigest()
        groups[bucket].append((key, qid))
    for group in groups.values():
        group.sort()

    def take(count: int) -> list[int]:
        result = []
        while len(result) < count:
            for group in groups.values():
                if group and len(result) < count:
                    result.append(group.pop(0)[1])
        return sorted(result)

    development = take(development_count)
    holdout = take(holdout_count)
    if set(development) & set(holdout) or (set(development) | set(holdout)) & PILOT:
        raise ValueError("Query split is not disjoint")
    output.mkdir(parents=True)
    (output / "query_split.json").write_text(json.dumps({
        "development_ids": development, "holdout_ids": holdout,
        "pilot_excluded_ids": sorted(PILOT), "label_status": "unlabeled",
        "holdout_policy": "freeze before training/tuning; do not inspect holdout labels during selection",
    }, indent=2) + "\n", encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps({
        "status": "query_ids_frozen_unlabeled", "source_query_sha256": sha(query_path),
        "split_sha256": sha(output / "query_split.json"),
        "development_count": len(development), "holdout_count": len(holdout),
        "selection": "deterministic SHA-256 order within query-character-length terciles",
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--development-count", type=int, default=60)
    parser.add_argument("--holdout-count", type=int, default=60)
    args = parser.parse_args()
    main(args.output, args.development_count, args.holdout_count)

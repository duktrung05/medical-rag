"""Independently reconcile official, Phase 1 PASS, and Phase 2 passage identities."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.data.phase2_batch import sha256_file


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(root: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    p1_root = root / "outputs/vibiomir_phase1_full_20261014_v1"
    p2_root = root / "outputs/vibiomir_phase2_full_v2"
    p1_path = p1_root / "release/phase1_release_manifest.json"
    p2_path = p2_root / "manifests/phase2_release_manifest.json"
    p1, p2 = _json(p1_path), _json(p2_path)
    if p2.get("phase1_release_manifest_sha256") != sha256_file(p1_path):
        raise ValueError("Phase 2/Phase 1 release linkage mismatch")
    if len(p1["pass_allowlist_parts"]) != 922 or len(p2["passage_parts"]) != 922:
        raise ValueError("Release must contain exactly 922 aligned parts")
    migration = _json(root / "configs/vibiomir/phase2_batch_migration_v1.json")
    official_path = root / migration["official_corpus"]
    if sha256_file(official_path) != migration["official_corpus_sha256"]:
        raise ValueError("Official corpus checksum mismatch")
    official_ids: set[int] = set()
    for batch in pq.ParquetFile(official_path).iter_batches(columns=["id"], batch_size=100_000):
        for value in batch.column(0).to_pylist():
            ident = int(value)
            if ident in official_ids:
                raise ValueError(f"Duplicate official ID: {ident}")
            official_ids.add(ident)
    exclusions_path = root / "configs/vibiomir/phase2_semantic_staging_20261015.json"
    exclusions = {int(row["doc_id"]) for row in _json(exclusions_path)["documents"]}
    pass_ids: set[int] = set()
    passage_ids: set[str] = set()
    passage_parents: set[int] = set()
    excluded_passage_count = 0
    passage_count = 0
    for index, (p1_part, p2_part) in enumerate(zip(p1["pass_allowlist_parts"],
                                                 p2["passage_parts"], strict=True)):
        name = f"part-{index:05d}"
        if p1_part["part"] != name or p2_part["part"] != name:
            raise ValueError(f"Non-aligned release part: {name}")
        pass_path = p1_root / "release/pass_document_ids" / f"{name}.parquet"
        passage_path = p2_root / "passages/strategy_C" / f"{name}.parquet"
        if sha256_file(pass_path) != p1_part["sha256"] or sha256_file(passage_path) != p2_part["sha256"]:
            raise ValueError(f"Part checksum mismatch: {name}")
        local_pass = set(map(int, pq.read_table(pass_path, columns=["doc_id"])
                                 .column(0).to_pylist()))
        if len(local_pass) != int(p1_part["rows"]) or pass_ids.intersection(local_pass):
            raise ValueError(f"Duplicate Phase 1 PASS ID or count mismatch: {name}")
        pass_ids.update(local_pass)
        local_parents: set[int] = set()
        local_passages = 0
        for batch in pq.ParquetFile(passage_path).iter_batches(
                columns=["passage_id", "doc_id"], batch_size=25_000):
            for passage_id, parent in zip(batch.column(0).to_pylist(),
                                          batch.column(1).to_pylist(), strict=True):
                parent = int(parent)
                if passage_id in passage_ids:
                    raise ValueError(f"Duplicate Phase 2 passage ID: {passage_id}")
                passage_ids.add(passage_id)
                local_parents.add(parent)
                if parent in exclusions:
                    excluded_passage_count += 1
                else:
                    passage_parents.add(parent)
                    passage_count += 1
                local_passages += 1
        if local_passages != int(p2_part["rows"]) or local_parents != local_pass:
            raise ValueError(f"Phase 1/2 parent coverage mismatch: {name}")
        if (index + 1) % 50 == 0 or index + 1 == 922:
            print(f"audited {index + 1}/922 parts", flush=True)
    eligible = pass_ids - exclusions
    unknown_pass = pass_ids - official_ids
    missing_passage_parents = eligible - passage_parents
    unknown_passage_parents = passage_parents - eligible
    orphan_exclusions = exclusions - pass_ids
    report = {
        "format": "medrag-v4-release-coverage-audit-v1",
        "status": "pass" if not (unknown_pass or missing_passage_parents or
                                 unknown_passage_parents or orphan_exclusions) and
                                 len(official_ids) == 1_220_598 and len(pass_ids) == 1_073_531 and
                                 len(eligible) == 1_073_525 and passage_count == 2_951_224 and
                                 excluded_passage_count == 49 else "fail",
        "scope": "922_release_parts",
        "official_source_count": len(official_ids),
        "phase1_pass_source_count": len(pass_ids),
        "semantic_excluded_source_count": len(exclusions),
        "eligible_source_count": len(eligible),
        "phase2_passage_count_before_exclusion": len(passage_ids),
        "eligible_passage_count": passage_count,
        "excluded_passage_count": excluded_passage_count,
        "official_not_phase1_pass_count": len(official_ids - pass_ids),
        "unknown_pass_source_ids": sorted(unknown_pass)[:20],
        "eligible_without_passage_ids": sorted(missing_passage_parents)[:20],
        "passage_parent_not_eligible_ids": sorted(unknown_passage_parents)[:20],
        "exclusion_not_phase1_pass_ids": sorted(orphan_exclusions)[:20],
        "phase1_release_manifest_sha256": sha256_file(p1_path),
        "phase2_release_manifest_sha256": sha256_file(p2_path),
        "official_corpus_sha256": migration["official_corpus_sha256"],
        "semantic_exclusions_sha256": sha256_file(exclusions_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                         encoding="utf-8")
    os.rename(temporary, output)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve()), indent=2))

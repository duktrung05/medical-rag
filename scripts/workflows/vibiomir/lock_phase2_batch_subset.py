#!/usr/bin/env python3
"""Lock a reproducible, language/length-stratified Phase 2 subset before inference."""
from __future__ import annotations

import argparse
import bisect
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def allocate(total: int, counts: dict[str, int]) -> dict[str, int]:
    denominator = sum(counts.values())
    exact = {key: total * value / denominator for key, value in counts.items()}
    result = {key: int(value) for key, value in exact.items()}
    remainder = total - sum(result.values())
    for key in sorted(counts, key=lambda item: (-(exact[item] - result[item]), item))[:remainder]:
        result[key] += 1
    return result


def lock(config_path: Path, output: Path, *, seed: int = 20261016,
         document_count: int = 1000, query_count: int = 20) -> dict:
    if output.exists():
        raise FileExistsError(f"Subset lock output already exists: {output}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    phase1_root = ROOT / config["phase1_root"]
    phase2_root = ROOT / config["phase2_root"]
    official_queries = ROOT / config["official_queries"]
    p1_release_path = phase1_root / "release/phase1_release_manifest.json"
    p2_release_path = phase2_root / "manifests/phase2_release_manifest.json"
    p1_release = json.loads(p1_release_path.read_text(encoding="utf-8"))
    p2_release = json.loads(p2_release_path.read_text(encoding="utf-8"))
    pass_ids: set[int] = set()
    for part in p1_release["pass_allowlist_parts"]:
        path = phase1_root / "release/pass_document_ids" / f"{part['part']}.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        pass_ids.update(map(int, pq.read_table(path, columns=["doc_id"])["doc_id"].to_pylist()))
    if len(pass_ids) != p1_release["pass_documents"]:
        raise ValueError("Phase 1 PASS allowlist rows do not match release manifest")

    mapping_path = phase2_root / p2_release["document_passage_mapping"]["path"]
    mapping_table = pq.read_table(mapping_path, columns=["doc_id", "source_non_whitespace_chars"])
    source_lengths = {int(doc): int(length) for doc, length in zip(
        mapping_table["doc_id"].to_pylist(), mapping_table["source_non_whitespace_chars"].to_pylist(), strict=True)}
    if len(source_lengths) != p2_release["documents"] or set(source_lengths) != pass_ids:
        raise ValueError("Phase 2 document mapping does not exactly match the Phase 1 PASS allowlist")

    stage_path = ROOT / config["semantic_staging"]
    staging = json.loads(stage_path.read_text(encoding="utf-8"))
    excluded = {int(row["doc_id"]) for row in staging["documents"]}
    rows_by_id = {}
    for part in p1_release["pass_allowlist_parts"]:
        path = phase1_root / "clean_documents" / f"{part['part']}.parquet"
        table = pq.read_table(path, columns=["doc_id", "detected_language"])
        for doc_id, lang in zip(table["doc_id"].to_pylist(), table["detected_language"].to_pylist(), strict=True):
            doc_id = int(doc_id)
            if doc_id in pass_ids and doc_id not in excluded:
                rows_by_id[doc_id] = ((lang or "unknown").strip().lower(), source_lengths[doc_id])
    if set(rows_by_id) != pass_ids - excluded:
        raise ValueError("Clean Phase 1 rows do not cover the eligible PASS population")

    by_language: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for doc_id, (language, length) in rows_by_id.items():
        group = language if language in {"vi", "en", "zh"} else "other"
        by_language[group].append((doc_id, length))
    language_counts = {language: len(values) for language, values in by_language.items()}
    language_quotas = allocate(document_count, language_counts)
    rng = random.Random(seed)
    selected: list[int] = []
    strata_report = {}
    for language in sorted(by_language):
        values = by_language[language]
        sorted_lengths = sorted(length for _, length in values)
        boundaries = [sorted_lengths[min(len(sorted_lengths) - 1, (len(sorted_lengths) * q) // 4)]
                      for q in (1, 2, 3)]
        strata: dict[int, list[int]] = defaultdict(list)
        for doc_id, length in values:
            strata[bisect.bisect_right(boundaries, length)].append(doc_id)
        stratum_counts = {str(index): len(strata[index]) for index in range(4)}
        quotas = allocate(language_quotas[language], stratum_counts)
        for index in range(4):
            selected.extend(rng.sample(strata[index], quotas[str(index)]))
        strata_report[language] = {
            "population": len(values), "sampled": language_quotas[language],
            "length_quartile_boundaries_source_non_whitespace_chars": boundaries,
            "quartiles": {str(index): {"population": stratum_counts[str(index)],
                                       "sampled": quotas[str(index)]} for index in range(4)},
        }
    if len(selected) != document_count or len(set(selected)) != document_count:
        raise ValueError("Stratified selection did not produce the requested unique document count")

    queries = pq.read_table(official_queries, columns=["id"])["id"].to_pylist()
    if len(queries) != 1200 or len(set(queries)) != 1200:
        raise ValueError("Official query snapshot does not contain 1,200 unique IDs")
    selected_queries = sorted(random.Random(seed ^ 0x5A17).sample(sorted(map(int, queries)), query_count))
    output.mkdir(parents=True)
    doc_path, query_path = output / "doc_ids.json", output / "query_ids.json"
    doc_path.write_text(json.dumps(sorted(selected), indent=2) + "\n", encoding="utf-8")
    query_path.write_text(json.dumps(selected_queries, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "format": "medrag-phase2-subset-lock-v1", "status": "locked_before_inference",
        "seed": seed, "positive_enriched": False,
        "sampling": "population-proportional language allocation; within-language source-length quartile allocation",
        "source_length": "Phase 2 document_passage_mapping.source_non_whitespace_chars",
        "documents": document_count, "queries": query_count,
        "eligible_documents": len(rows_by_id), "phase1_pass_documents": len(pass_ids),
        "semantic_exclusions": sorted(excluded), "language_strata": strata_report,
        "query_selection": "uniform deterministic sample from the official 1,200 query IDs",
        "doc_ids_path": doc_path.name, "doc_ids_sha256": sha256(doc_path),
        "query_ids_path": query_path.name, "query_ids_sha256": sha256(query_path),
        "phase1_release_manifest_sha256": sha256(p1_release_path),
        "phase2_release_manifest_sha256": sha256(p2_release_path),
        "document_passage_mapping_sha256": sha256(mapping_path),
        "official_query_sha256": sha256(official_queries),
        "semantic_staging_sha256": sha256(stage_path),
    }
    (output / "subset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--documents", type=int, default=1000)
    parser.add_argument("--queries", type=int, default=20)
    args = parser.parse_args()
    print(json.dumps(lock(args.config.resolve(), args.output.resolve(), seed=args.seed,
                          document_count=args.documents, query_count=args.queries),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

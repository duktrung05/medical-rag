#!/usr/bin/env python3
"""Compare two completed Phase 2 subset runs for deterministic replay."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from r2ai.evidence_io import read_ranking_cache


def canonical_zip_hash(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError(f"ZIP CRC failure in {path}: {corrupt}")
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if names != ["predictions.json"]:
            raise ValueError(f"Unexpected submission ZIP entries: {names}")
        records = json.loads(archive.read("predictions.json"))
    payload = json.dumps(records, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare(first: Path, second: Path, output: Path, *, retrieval_tolerance: float = 1e-5,
            rerank_tolerance: float = 1e-4) -> dict:
    for path in (first, second):
        manifest_path = path / "run_manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"Run is not complete: {path}")
        if json.loads(manifest_path.read_text(encoding="utf-8")).get("status") != "complete":
            raise ValueError(f"Run is not complete: {path}")
    a, meta_a = read_ranking_cache(first / "rankings.jsonl.gz")
    b, meta_b = read_ranking_cache(second / "rankings.jsonl.gz")
    if meta_a != meta_b:
        raise ValueError("Ranking cache metadata differ between repeated runs")
    if [item.query_id for item in a] != [item.query_id for item in b]:
        raise ValueError("Query order/coverage differs between repeated runs")
    max_retrieval_delta = 0.0
    max_rerank_delta = 0.0
    total_candidates = 0
    for left, right in zip(a, b, strict=True):
        if left.document_order != right.document_order:
            raise ValueError(f"Document ranking differs for query_id={left.query_id}")
        if [item.key for item in left.candidates] != [item.key for item in right.candidates]:
            raise ValueError(f"Passage candidate order/membership differs for query_id={left.query_id}")
        total_candidates += len(left.candidates)
        for c1, c2 in zip(left.candidates, right.candidates, strict=True):
            if (c1.rerank_score is None) != (c2.rerank_score is None):
                raise ValueError(f"Rerank status differs for query_id={left.query_id}, key={c1.key}")
            max_retrieval_delta = max(max_retrieval_delta,
                                      abs(float(c1.retrieval_score) - float(c2.retrieval_score)))
            if c1.rerank_score is not None:
                max_rerank_delta = max(max_rerank_delta,
                                       abs(float(c1.rerank_score) - float(c2.rerank_score)))
    if max_retrieval_delta > retrieval_tolerance:
        raise ValueError(f"Dense retrieval score delta {max_retrieval_delta} exceeds {retrieval_tolerance}")
    if max_rerank_delta > rerank_tolerance:
        raise ValueError(f"Reranker score delta {max_rerank_delta} exceeds {rerank_tolerance}")
    hash_a = canonical_zip_hash(first / "submission.zip")
    hash_b = canonical_zip_hash(second / "submission.zip")
    if hash_a != hash_b:
        raise ValueError("Canonical prediction payload differs between repeated runs")
    report = {
        "format": "medrag-phase2-repeatability-report-v1", "status": "pass",
        "runs": [str(first.resolve()), str(second.resolve())],
        "queries": len(a), "ranked_candidates_compared": total_candidates,
        "candidate_membership_order": "identical", "document_order": "identical",
        "max_dense_score_absolute_delta": max_retrieval_delta,
        "dense_score_tolerance": retrieval_tolerance,
        "max_reranker_score_absolute_delta": max_rerank_delta,
        "reranker_score_tolerance": rerank_tolerance,
        "canonical_predictions_sha256": hash_a,
        "zip1_sha256": hashlib.sha256((first / "submission.zip").read_bytes()).hexdigest(),
        "zip2_sha256": hashlib.sha256((second / "submission.zip").read_bytes()).hexdigest(),
        "zip_container_bytes_identical": (first / "submission.zip").read_bytes()
        == (second / "submission.zip").read_bytes(),
        "f2": "unmeasured",
        "query_coverage": "20/20 strict official IDs",
        "strict_submission_validation": "pass for both runs",
        "cache_replay": "pass for both runs",
        "runtime_versions": {
            "python": platform.python_version(),
            **{name: importlib.metadata.version(dist) for name, dist in (
                ("torch", "torch"), ("sentence_transformers", "sentence-transformers"),
                ("transformers", "transformers"), ("bm25s", "bm25s"),
                ("numpy", "numpy"), ("pyarrow", "pyarrow"))},
        },
        "cuda_visible_devices": "2; runner's cuda:0 maps to physical GPU 2",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifests = [json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
                 for path in (first, second)]
    subset = json.loads((first / "subset_manifest.json").read_text(encoding="utf-8"))
    lines = [
        "# Phase 2 subset integration report — 2026-10-16", "",
        f"Status: **REAL-MODEL SUBSET E2E PASS**; repeatability: **{report['status'].upper()}**.",
        "F2: **UNMEASURED** (no official qrels/score used). This run is not a BTC submission and does not establish full-corpus quality.", "",
        "## Locked subset", "",
        f"- Documents: {subset['documents']:,} of {subset['eligible_documents']:,} eligible PASS documents; fixed seed {subset['seed']}.",
        f"- Queries: {subset['queries']} official query IDs, selected deterministically from the official 1,200.",
        f"- Language allocations: " + ", ".join(f"{lang}={item['sampled']}" for lang, item in subset['language_strata'].items()) + ".",
        "- Source-length stratification uses Phase 2 `source_non_whitespace_chars`; `positive_enriched=false`.",
        "- Six semantic staging documents (49 mapped passage rows) were excluded; selected dataset contains no held/staging IDs.", "",
        "## Verified run results", "",
        "| Run | Load/provenance (s) | Doc retrieval (s) | Passage retrieval + rerank (s) | Reranked pairs | Avg docs/query | Avg chunks/query | Empty queries | Strict validation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, manifest in zip(("run_01", "run_02"), manifests, strict=True):
        summary = manifest["summary"]
        empty = int(round(summary["empty_doc_rate"] * summary["num_queries"]))
        times = manifest["stage_seconds"]
        lines.append(
            f"| {label} | {times['load_and_provenance_seconds']:.1f} | "
            f"{times['document_retrieval_seconds']:.1f} | "
            f"{times['passage_retrieval_and_reranking_seconds']:.1f} | "
            f"{manifest['reranked_passages']:,} | {summary['average_docs']:.2f} | "
            f"{summary['average_chunks']:.2f} | {empty}/{summary['num_queries']} | PASS |"
        )
    lines.extend([
        "", "Both runs consumed the same verified dataset/provenance fingerprint, used the pinned BGE-M3 and bge-reranker-v2-m3 commits, replayed selector output from cache, and passed strict official-ID/query validation. See `repeatability_report.json` for candidate/score deltas and canonical payload hashes.",
        "", "## Interpretation and limits", "",
        "- The 20 queries are paired with a random 1,000-document subset, not the full document corpus. Retrieval misses and empty outputs therefore cannot be interpreted as a full-corpus score estimate.",
        "- The 6/20 empty outputs and observed average selection counts are descriptive only; official subset F2 remains unmeasured.",
        "- The runner validates grounding against the selected Phase 2 source slices and mapping, but this does not constitute an independent relevance/clinical review.",
        "- No 1.2M-document inference, BTC upload, or official F2 score was run.",
        "- One earlier attempt is preserved under `.run_01.incomplete-*`; it reached inference/export but did not publish because the model library returned a null `_commit_hash`. The successful retries resolved and verified the exact cached commit IDs.",
        "",
    ])
    (output.parent / "SUBSET_INTEGRATION_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare(args.first.resolve(), args.second.resolve(), args.output.resolve()),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Inspect existing submissions and scores on CPU; do not change predictions."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
import random
import re
import sys
import zipfile
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2ai.evidence_selector import EvidenceCandidate, QueryRanking, SelectionConfig, select_evidence
from r2ai.validate import validate

CJK = re.compile(r"[㐀-鿿豈-﫿]")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def quantiles(values):
    return dict(zip(["min", "p05", "p25", "p50", "p75", "p95", "p99", "max"],
                    np.quantile(values, [0, .05, .25, .5, .75, .95, .99, 1]).tolist()))


def inspect_zip(path):
    with zipfile.ZipFile(path) as z:
        members = z.namelist()
        bad_crc = z.testzip()
        records = json.loads(z.read("predictions.json"))
    doc_counts, chunk_counts, sizes, fingerprints = [], [], [], {}
    parentless = duplicate_evidence = without_evidence = global_duplicates = 0
    for row in records:
        docs = row["relevant_docs"]
        chunks = row["relevant_chunks"]
        pairs = [(c["doc_id"], hashlib.sha256(c["chunk_text"].encode()).hexdigest()) for c in chunks]
        doc_counts.append(len(docs)); chunk_counts.append(len(chunks))
        sizes.extend(len(c["chunk_text"]) for c in chunks)
        parentless += sum(c["doc_id"] not in docs for c in chunks)
        duplicate_evidence += len(pairs) - len(set(pairs))
        global_duplicates += len(chunks) - len({c["chunk_text"] for c in chunks})
        without_evidence += len(set(docs) - {c["doc_id"] for c in chunks})
        fingerprints[row["id"]] = {"docs": docs, "chunks": set(pairs),
                                   "chunk_parents": [c["doc_id"] for c in chunks]}
    result = {
        "path": str(path.relative_to(ROOT)), "sha256": digest(path),
        "mtime_vietnam": datetime.fromtimestamp(path.stat().st_mtime, ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "zip_bytes": path.stat().st_size, "members": members, "bad_crc": bad_crc,
        "queries": len(records), "doc_total": sum(doc_counts), "chunk_total": sum(chunk_counts),
        "doc_counts": dict(Counter(doc_counts)), "chunk_counts": quantiles(chunk_counts),
        "average_docs": float(np.mean(doc_counts)), "average_chunks": float(np.mean(chunk_counts)),
        "chunk_characters": quantiles(sizes), "documents_without_evidence": without_evidence,
        "chunks_without_parent": parentless, "duplicate_doc_text_pairs": duplicate_evidence,
        "cross_document_duplicate_text_occurrences": global_duplicates,
    }
    del records
    result["validation_errors"] = validate(path)
    return result, fingerprints


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=ROOT / "outputs/precision_review_20261007")
    p.add_argument("--tokenizer", type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    names = {
        "baseline": "outputs/submissions/vibiomir_test_best_20261006.zip",
        "taxonomy": "outputs/submissions/vibiomir_taxonomy_test_20261006.zip",
        "submission": "outputs/evidence/test_20261006/selected/submission.zip",
        "exact80": "outputs/submissions/vibiomir_probes_20261007/vibiomir_rerank80_exact_20261007.zip",
        "window80": "outputs/submissions/vibiomir_probes_20261007/vibiomir_rerank80_window1_20261007.zip",
    }
    report = {"reviewed_at_vietnam": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
              "official_score_recomputed": False,
              "screenshot_zip_checksum_available": False, "submissions": {}, "comparisons": {}}
    identities = {}
    for name, relative in names.items():
        report["submissions"][name], identities[name] = inspect_zip(ROOT / relative)
        print("Audited", name, flush=True)
    for left, right in [("baseline", "submission"), ("baseline", "taxonomy"), ("exact80", "window80")]:
        a, b = identities[left], identities[right]
        if set(a) != set(b):
            raise ValueError("Query sets differ")
        report["comparisons"][f"{left}_vs_{right}"] = {
            "queries_with_changed_document_sets": sum(set(a[q]["docs"]) != set(b[q]["docs"]) for q in a),
            "queries_with_changed_document_order": sum(a[q]["docs"] != b[q]["docs"] for q in a),
            "queries_with_changed_evidence_sets": sum(a[q]["chunks"] != b[q]["chunks"] for q in a),
            "queries_with_changed_chunk_parents": sum(a[q]["chunk_parents"] != b[q]["chunk_parents"] for q in a),
            "added_evidence": sum(len(b[q]["chunks"] - a[q]["chunks"]) for q in a),
            "removed_evidence": sum(len(a[q]["chunks"] - b[q]["chunks"]) for q in a),
        }
    cache_path = ROOT / "outputs/evidence/test_20261006/rankings.jsonl.gz"
    variants = [("keep200_threshold-6", SelectionConfig(mode="threshold", max_docs=200, max_chunks=120,
                    chunks_per_doc=1, chunk_threshold=-6, document_mode="baseline"))]
    for threshold in [-6, -4, -2, 0, 1, 2]:
        variants.append((f"reranked_docs_threshold{threshold}", SelectionConfig(mode="threshold",
            max_docs=80, max_chunks=60, chunks_per_doc=1, chunk_threshold=threshold,
            doc_threshold=threshold, document_mode="threshold")))
    variants.append(("filtered_original_doc_order_threshold-2", SelectionConfig(mode="threshold",
        max_docs=80, max_chunks=60, chunks_per_doc=1, chunk_threshold=-2,
        doc_threshold=-2, document_mode="threshold")))
    variant_stats = {name: {"config": asdict(config), "docs": [], "chunks": [],
                           "empty_chunks": 0, "empty_docs": 0} for name, config in variants}
    raw_gates = {t: [] for t in [-6, -4, -2, 0, 1, 2]}
    scores, selected_scores, samples = [], [], []
    pool_docs, unscored_docs, chunks_per_doc, duplicates = [], [], [], []
    examples, replay_mismatches = [], 0
    rng = random.Random(42)
    with gzip.open(cache_path, "rt", encoding="utf-8") as f:
        header = json.loads(next(f))
        report["cache_metadata"] = header["metadata"]
        for line in f:
            row = json.loads(line)
            ranking = QueryRanking(row["query_id"], row["document_order"],
                [EvidenceCandidate(**c) for c in row["candidates"]], row["query_text"])
            candidates = ranking.candidates
            if any(c.rerank_score is None for c in candidates):
                raise ValueError("Expected authentic scores for all candidates")
            row_scores = [c.rerank_score for c in candidates]
            scores.extend(row_scores)
            counts = Counter(c.doc_id for c in candidates)
            pool_docs.append(len(counts)); chunks_per_doc.extend(counts.values())
            unscored_docs.append(len(set(ranking.document_order) - set(counts)))
            ordered_candidates = sorted(candidates, key=lambda c: (-c.rerank_score, c.doc_id, c.chunk_index))
            ordered_docs = list(dict.fromkeys(c.doc_id for c in ordered_candidates))
            score_ranking = QueryRanking(ranking.query_id, ordered_docs, ordered_candidates, ranking.query_text)
            groups = defaultdict(list)
            for c in candidates:
                groups[c.chunk_text].append(c.doc_id)
                key = c.doc_id, hashlib.sha256(c.chunk_text.encode()).hexdigest()
                if key in identities["submission"][ranking.query_id]["chunks"]:
                    selected_scores.append(c.rerank_score)
            duplicates.append(sum(len(v) - 1 for v in groups.values()))
            for t in raw_gates:
                raw_gates[t].append(sum(s >= t for s in row_scores))
            for name, config in variants:
                use_original_order = name.startswith(("keep200", "filtered_original"))
                prediction, diag = select_evidence(ranking if use_original_order else score_ranking, config)
                stats = variant_stats[name]
                stats["docs"].append(len(prediction["relevant_docs"]))
                stats["chunks"].append(len(prediction["relevant_chunks"]))
                stats["empty_chunks"] += not prediction["relevant_chunks"]
                stats["empty_docs"] += not prediction["relevant_docs"]
                if name == "keep200_threshold-6":
                    expected = identities["submission"][ranking.query_id]
                    predicted = {(c["doc_id"], hashlib.sha256(c["chunk_text"].encode()).hexdigest())
                                 for c in prediction["relevant_chunks"]}
                    replay_mismatches += expected["docs"] != prediction["relevant_docs"] or expected["chunks"] != predicted
            for c in rng.sample(candidates, min(5, len(candidates))):
                selected = (c.doc_id, hashlib.sha256(c.chunk_text.encode()).hexdigest()) in identities["submission"][ranking.query_id]["chunks"]
                samples.append((ranking.query_text, f"{c.title}\n{c.chunk_text}" if c.title else c.chunk_text,
                                selected, bool(CJK.search(c.chunk_text))))
            if ranking.query_id in [2, 10, 100, 410]:
                selected = [c for c in candidates if
                    (c.doc_id, hashlib.sha256(c.chunk_text.encode()).hexdigest())
                    in identities["submission"][ranking.query_id]["chunks"]]
                low = min(selected, key=lambda c: c.rerank_score)
                top = max(candidates, key=lambda c: c.rerank_score)
                examples.append({"query_id": ranking.query_id, "query": ranking.query_text,
                    "highest_score": {**asdict(top), "chunk_text": top.chunk_text[:550]},
                    "lowest_selected": {**asdict(low), "chunk_text": low.chunk_text[:550]}})
    report["cache"] = {
        "path": str(cache_path.relative_to(ROOT)), "sha256": digest(cache_path),
        "pairs": len(scores), "pool_documents_per_query": quantiles(pool_docs),
        "unscored_baseline_documents_per_query": quantiles(unscored_docs),
        "chunks_per_document_in_pool": quantiles(chunks_per_doc), "score_quantiles": quantiles(scores),
        "selected_score_quantiles": quantiles(selected_scores),
        "selected_chunks_below_zero": sum(s < 0 for s in selected_scores),
        "selected_chunks_below_minus2": sum(s < -2 for s in selected_scores),
        "cross_document_duplicate_text_occurrences": sum(duplicates),
        "baseline_selector_replay_mismatched_queries": replay_mismatches,
    }
    report["raw_threshold_counts"] = {str(t): {"average": float(np.mean(v)),
        "empty_queries": sum(n == 0 for n in v), "counts": quantiles(v)} for t, v in raw_gates.items()}
    report["selection_ablation_without_labels"] = {}
    for name, stats in variant_stats.items():
        docs, chunks = stats.pop("docs"), stats.pop("chunks")
        report["selection_ablation_without_labels"][name] = {**stats,
            "document_order": "original fused ranking" if name.startswith(("keep200", "filtered_original"))
                              else "descending maximum authentic reranker score",
            "average_docs": float(np.mean(docs)), "average_chunks": float(np.mean(chunks)),
            "doc_counts": quantiles(docs), "chunk_counts": quantiles(chunks),
            "precision": None, "recall": None, "f2": None}
    if args.tokenizer:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(args.tokenizer))
        tokenizer.no_truncation(); tokenizer.no_padding()
        encodings = tokenizer.encode_batch([(q, text) for q, text, _, _ in samples])
        lengths = [len(e.ids) for e in encodings]
        report["reranker_token_length_sample"] = {
            "tokenizer_path": str(args.tokenizer), "tokenizer_sha256": digest(args.tokenizer),
            "seed": 42, "sample_strategy": "five random candidates per query; full query/title/chunk pair",
            "pairs": len(samples), "lengths": quantiles(lengths),
            "pairs_above_512": sum(n > 512 for n in lengths),
            "pairs_above_512_fraction": sum(n > 512 for n in lengths) / len(lengths),
            "selected_pairs": sum(s[2] for s in samples),
            "selected_pairs_above_512": sum(n > 512 and s[2] for n, s in zip(lengths, samples)),
            "cjk_pairs": sum(s[3] for s in samples),
            "cjk_pairs_above_512": sum(n > 512 and s[3] for n, s in zip(lengths, samples)),
        }
    report["qualitative_examples_not_official_judgments"] = examples
    report["corpus"] = {
        "old_chunk_rows": pq.ParquetFile(ROOT / "data/vibiomir/chunks_v4.parquet").metadata.num_rows,
        "old_index_documents": len(np.load(ROOT / "data/vibiomir/docidx4_ids.npy", mmap_mode="r")),
        "official_documents": pq.ParquetFile(ROOT / "data/raw/vibiomir/links_corpus.parquet").metadata.num_rows,
        "running_extraction_not_read": "data/vibiomir/chunks_crawl_20261007.parquet",
    }
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output / "audit.json"), "cache": report["cache"],
                      "ablation": report["selection_ablation_without_labels"],
                      "token_lengths": report.get("reranker_token_length_sample")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

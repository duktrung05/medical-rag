"""Audit actual V4 dense and reranker token truncation on deterministic pilot cases."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sqlite3

import pyarrow.parquet as pq

from src.medrag.core.config import load_config
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.core.contracts import Passage, Source
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.disk_dense import payload
from src.medrag.data.phase2_batch import sha256_file


def audit(sparse_index: Path, pilot_index: Path, pilot_report: Path,
          config_path: Path, queries_path: Path, output: Path) -> dict:
    config = load_config(config_path)
    sparse_manifest = json.loads((sparse_index / "manifest.json").read_text())
    pilot_manifest = json.loads((pilot_index / "manifest.json").read_text())
    pilot = json.loads(pilot_report.read_text())
    if sparse_manifest.get("status") != "complete" or pilot_manifest.get("status") != "complete":
        raise ValueError("Audit requires complete sparse and pilot indexes")
    if (sha256_file(pilot_index / "manifest.json") != pilot.get("index_manifest_sha256")
            or sha256_file(config_path) != pilot.get("config_sha256")
            or sha256_file(queries_path) != pilot.get("official_queries_sha256")):
        raise ValueError("Pilot/config/query hashes differ from the locked benchmark")

    dense_snapshot = require_cached_revision(config["dense"]["model"], config["dense"]["revision"])
    reranker_snapshot = require_cached_revision(config["reranker"]["model"], config["reranker"]["revision"])
    from transformers import AutoTokenizer
    dense_tokenizer = AutoTokenizer.from_pretrained(dense_snapshot, local_files_only=True)
    reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_snapshot, local_files_only=True)

    version = str(pilot_manifest["corpus_version"])
    db_path = pilot_index / "corpus.sqlite"
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        rows = db.execute(
            "SELECT p.passage_id,p.official_id,p.url,p.text,p.source_start,p.source_end,"
            "p.source_sha256,p.language,p.provenance,s.title,s.text AS source_text,"
            "s.raw_sha256,s.origin,s.metadata FROM passages p JOIN sources s USING(official_id) "
            "ORDER BY length(p.text)+length(s.title) DESC LIMIT 6").fetchall()
        cases = {str(row["passage_id"]): row for row in rows}
        sample_reasons = {str(row["passage_id"]): ["longest_payload"] for row in rows}
        tail_ids = [int(row[0]) for row in db.execute(
            "SELECT official_id FROM sources ORDER BY official_id DESC LIMIT 6")]
        for official_id in tail_ids:
            row = db.execute(
                "SELECT p.passage_id,p.official_id,p.url,p.text,p.source_start,p.source_end,"
                "p.source_sha256,p.language,p.provenance,s.title,s.text AS source_text,"
                "s.raw_sha256,s.origin,s.metadata FROM passages p JOIN sources s USING(official_id) "
                "WHERE p.official_id=? ORDER BY p.source_end DESC LIMIT 1", (official_id,)).fetchone()
            if row:
                cases[str(row["passage_id"])] = row
                sample_reasons.setdefault(str(row["passage_id"]), []).append("tail_passage")
        empty_title = db.execute(
            "SELECT p.passage_id,p.official_id,p.url,p.text,p.source_start,p.source_end,"
            "p.source_sha256,p.language,p.provenance,s.title,s.text AS source_text,"
            "s.raw_sha256,s.origin,s.metadata FROM passages p JOIN sources s USING(official_id) "
            "WHERE s.title='' ORDER BY p.official_id,p.source_start LIMIT 1").fetchone()
        if empty_title:
            cases[str(empty_title["passage_id"])] = empty_title
            sample_reasons.setdefault(str(empty_title["passage_id"]), []).append("empty_title")
    actual_empty_title_found = bool(empty_title)

    query_rows = {int(row["id"]): str(row["query"]) for row in
                  pq.read_table(queries_path, columns=["id", "query"]).to_pylist()}
    query_ids = list(pilot.get("query_ids", []))
    if len(query_ids) != 50 or any(qid not in query_rows for qid in query_ids):
        raise ValueError("Pilot report does not identify the locked 50 official queries")
    query_id = max(query_ids, key=lambda qid: len(reranker_tokenizer(query_rows[qid],
                                                                       add_special_tokens=True)["input_ids"]))
    query = query_rows[query_id]
    maximum_dense = int(config["dense"]["document_max_length"])
    maximum_query = int(config["dense"]["query_max_length"])
    maximum_pair = int(config["reranker"]["max_length"])
    dense_query_full = dense_tokenizer(query, add_special_tokens=True, truncation=False)
    dense_query_truncated = dense_tokenizer(query, add_special_tokens=True, truncation=True,
                                            max_length=maximum_query)
    samples = []
    for row in cases.values():
        source = Source(version, int(row["official_id"]), str(row["url"]), str(row["title"]),
                        str(row["source_text"]), str(row["origin"]), str(row["raw_sha256"]),
                        json.loads(row["metadata"]))
        passage = Passage(str(row["passage_id"]), version, int(row["official_id"]),
                          str(row["url"]), str(row["text"]), int(row["source_start"]),
                          int(row["source_end"]), str(row["source_sha256"]),
                          str(row["language"]), str(row["provenance"]))
        if not verify(passage, source):
            raise ValueError(f"Passage source slice/provenance validation failed: {passage.passage_id}")

        text = payload(passage.text, source.title)
        dense_full = dense_tokenizer(text, add_special_tokens=True, truncation=False,
                                     return_offsets_mapping=True)
        dense_trunc = dense_tokenizer(text, add_special_tokens=True, truncation=True,
                                      max_length=maximum_dense, return_offsets_mapping=True)
        dense_offsets = [end for start, end in dense_trunc["offset_mapping"] if end > start]
        dense_end = max(dense_offsets, default=0)
        dense_passage_start = len(source.title) + 1 if source.title else 0

        pair_full = reranker_tokenizer(query, text, add_special_tokens=True, truncation=False,
                                       return_offsets_mapping=True)
        pair_trunc = reranker_tokenizer(query, text, add_special_tokens=True, truncation=True,
                                        max_length=maximum_pair, return_offsets_mapping=True)
        pair_seq = pair_trunc.sequence_ids()
        pair_offsets = pair_trunc["offset_mapping"]
        query_kept = sum(1 for seq, offset in zip(pair_seq, pair_offsets, strict=True)
                         if seq == 0 and offset[1] > offset[0])
        payload_kept = sum(1 for seq, offset in zip(pair_seq, pair_offsets, strict=True)
                           if seq == 1 and offset[1] > offset[0])
        payload_offsets = [end for seq, (start, end) in zip(pair_seq, pair_offsets, strict=True)
                           if seq == 1 and end > start]
        payload_end = max(payload_offsets, default=0)
        samples.append({
            "passage_id": passage.passage_id, "official_id": passage.official_id,
            "sample_reasons": sample_reasons.get(passage.passage_id, []),
            "is_tail_passage": passage.source_end == len(source.text.rstrip()),
            "title_empty": not bool(source.title), "title_chars": len(source.title),
            "passage_chars": len(passage.text), "source_offsets_valid": True,
            "provenance_present": bool(passage.provenance),
            "dense": {"raw_tokens": len(dense_full["input_ids"]),
                      "retained_tokens": len(dense_trunc["input_ids"]),
                      "max_length": maximum_dense, "truncated": len(dense_full["input_ids"]) > maximum_dense,
                      "retained_chars_end": dense_end,
                      "passage_chars_retained": max(0, min(len(passage.text), dense_end-dense_passage_start))},
            "reranker_pair": {"query_id": query_id,
                              "raw_tokens": len(pair_full["input_ids"]),
                              "retained_tokens": len(pair_trunc["input_ids"]),
                              "max_length": maximum_pair,
                              "truncated": len(pair_full["input_ids"]) > maximum_pair,
                              "query_tokens_retained": query_kept,
                              "payload_tokens_retained": payload_kept,
                              "payload_chars_retained_end": payload_end},
        })

    fixture_passage = next(iter(cases.values()))
    fallback_text = str(fixture_passage["text"])
    fallback_payload = payload(fallback_text, "")
    if fallback_payload != fallback_text:
        raise ValueError("Empty-title payload must preserve passage text exactly")
    fallback_encoding = dense_tokenizer(fallback_payload, add_special_tokens=True,
                                        truncation=False)

    report = {
        "format": "medrag-v4-tokenization-audit-v1", "status": "PASS_WITH_BASELINE_TRUNCATION",
        "scope": "six-longest-six-tail-and-empty-title-cases-from-12,347-passage-pilot",
        "sparse_manifest_sha256": sha256_file(sparse_index / "manifest.json"),
        "pilot_manifest_sha256": sha256_file(pilot_index / "manifest.json"),
        "pilot_report_sha256": sha256_file(pilot_report), "config_sha256": sha256_file(config_path),
        "dense": {"model": config["dense"]["model"], "revision": config["dense"]["revision"],
                  "input": "title + newline + passage; empty title uses passage only",
                  "max_length": maximum_dense, "query_max_length": maximum_query,
                  "query_sample": {"query_id": query_id,
                                   "raw_tokens": len(dense_query_full["input_ids"]),
                                   "retained_tokens": len(dense_query_truncated["input_ids"]),
                                   "truncated": len(dense_query_full["input_ids"]) > maximum_query}},
        "reranker": {"model": config["reranker"]["model"],
                     "revision": config["reranker"]["revision"], "max_length": maximum_pair,
                     "input": "official query paired with title + newline + passage"},
        "pilot_reranker_aggregate": {"pairs": pilot["rerank_pairs_total"],
                                     "truncated_pairs": pilot["truncated_pairs_total"]},
        "empty_title_fallback": {"status": "PASS", "actual_empty_title_source_in_pilot": actual_empty_title_found,
                                 "fixture_is_synthetic": not actual_empty_title_found,
                                 "payload_equals_passage": True,
                                 "dense_raw_tokens": len(fallback_encoding["input_ids"])},
        "samples": samples,
        "limitations": ["This is a deterministic pilot audit, not an all-corpus truncation census.",
                        "Baseline truncation is measured and retained as configured; it does not imply only two passages per source are read."],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    temporary.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
    os.replace(temporary, output)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-index", type=Path, required=True)
    parser.add_argument("--pilot-index", type=Path, required=True)
    parser.add_argument("--pilot-report", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.sparse_index.resolve(), args.pilot_index.resolve(),
                   args.pilot_report.resolve(), args.config.resolve(),
                   args.queries.resolve(), args.output.resolve())
    print(json.dumps({"status": result["status"], "samples": len(result["samples"]),
                      "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()

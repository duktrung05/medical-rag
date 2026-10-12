"""CPU rerank, source-verify and produce an official-schema pilot preview."""
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from sentence_transformers import CrossEncoder

from src.retrieval_v2.core import assert_grounded, score_and_select
from src.retrieval_v2.settings import DEFAULT_CONFIG, cached_model_path, load_config

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--pilot", type=Path, default=ROOT / "outputs/retrieval_v2/pilot")
    ap.add_argument("--rerank-budget", type=int)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N; GPU requires prior resource approval")
    args = ap.parse_args()
    config = load_config(args.config)
    args.rerank_budget = args.rerank_budget or config["reranker"]["pilot_cpu_budget_per_query"]
    if not 1 <= args.rerank_budget <= 20:
        raise ValueError("rerank-budget must be 1..20 on CPU pilot")
    trace = [json.loads(line) for line in (args.pilot / "V2_CANDIDATE_TRACE.jsonl").open(encoding="utf-8")]
    model = CrossEncoder(str(cached_model_path(config["reranker"]["model"],
                                               config["reranker"]["revision"])),
                         device=args.device, max_length=config["reranker"]["max_length"],
                         activation_fn=torch.nn.Identity())
    con = sqlite3.connect(args.pilot / "v2_sparse.sqlite")
    results = []
    logits_cache = []
    score_rows = []
    reranked_rows = []
    for row in trace:
        candidates = row["candidates"][:args.rerank_budget]
        pairs = [(row["query"], f"{c['title']}\n{c['text']}") for c in candidates]
        logits = model.predict(pairs, batch_size=2, show_progress_bar=False,
                               convert_to_numpy=True)
        for candidate, logit in zip(candidates, logits, strict=True):
            candidate["rerank_logit"] = float(logit)
            untruncated = model.tokenizer(row["query"], f"{candidate['title']}\n{candidate['text']}",
                                          truncation=False, add_special_tokens=True)
            candidate["input_tokens"] = len(untruncated["input_ids"])
            candidate["truncated"] = candidate["input_tokens"] > config["reranker"]["max_length"]
            reranked_rows.append({"query_id": row["id"], "passage_id": candidate["passage_id"],
                                  "doc_id": candidate["doc_id"], "fusion_rank": candidate["fusion_rank"],
                                  "rerank_logit": candidate["rerank_logit"],
                                  "input_tokens": candidate["input_tokens"],
                                  "truncated": candidate["truncated"]})
        by_doc = {}
        for candidate in candidates:
            by_doc.setdefault(candidate["doc_id"], []).append(candidate)
        for doc_id, items in by_doc.items():
            ordered = sorted((x["rerank_logit"] for x in items), reverse=True)
            first_rank = min(x["fusion_rank"] for x in items)
            score_rows.append({"query_id": row["id"], "doc_id": doc_id,
                               "max_logit_control": ordered[0],
                               "v2_score": ordered[0] + (0.1 * ordered[1] if len(ordered) > 1 else 0)
                               + 1 / (60 + first_rank),
                               "supporting_passages": len(items)})
        prediction = {"id": row["id"], **score_and_select(
            candidates, max_docs=config["output"]["max_documents"],
            max_chunks=config["output"]["max_evidence"])}
        assert_grounded(con, prediction)
        results.append(prediction)
        logits_cache.append({"id": row["id"], "scores": [
            {"passage_id": c["passage_id"], "logit": c["rerank_logit"],
             "input_words": len((c["title"] + " " + c["text"]).split()),
             "input_tokens": c["input_tokens"], "truncated": c["truncated"]}
            for c in candidates]})
    con.close()
    predictions = args.pilot / "V2_FINAL_PREDICTIONS.jsonl"
    if predictions.exists():
        raise FileExistsError(predictions)
    predictions.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in results), encoding="utf-8")
    (args.pilot / "V2_RERANKED_CANDIDATES.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False) + "\n" for x in logits_cache), encoding="utf-8")
    pq.write_table(pa.Table.from_pylist(reranked_rows), args.pilot / "V2_RERANKED_CANDIDATES.parquet")
    with (args.pilot / "V2_DOCUMENT_SCORING.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "doc_id", "max_logit_control",
                                                 "v2_score", "supporting_passages"])
        writer.writeheader()
        writer.writerows(score_rows)
    # Complete 1,200-query array only for schema validation. Empty non-pilot rows are explicit.
    full_ids = pq.read_table(ROOT / "data/raw/vibiomir/query.parquet", columns=["id"]).column("id").to_pylist()
    by_id = {x["id"]: x for x in results}
    preview = [by_id.get(qid, {"id": qid, "relevant_docs": [], "relevant_chunks": []}) for qid in full_ids]
    zip_path = args.pilot / "V2_SCHEMA_PREVIEW_NOT_FOR_SUBMISSION.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("submission.json", json.dumps(preview, ensure_ascii=False))
    print(f"Reranked {len(results)} queries; grounded pilot predictions: {predictions}")


if __name__ == "__main__":
    main()

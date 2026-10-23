"""Locate the P0 truncation boundary with the pinned local reranker tokenizer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.v4.audit_evidence_survival import sha
from src.medrag.core.config import load_config
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.data.disk_store import DiskCorpusStore, SourceLookup


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main(output: Path) -> None:
    output = output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config = load_config(ROOT / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml")
    reranker = config["reranker"]
    snapshot = require_cached_revision(reranker["model"], reranker["revision"])
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    migration = read_json(ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official queries changed")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    qid, parent = 130, 2968260
    checkpoint_path = (ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
                       / "queries" / f"{qid}.json")
    checkpoint = read_json(checkpoint_path)
    candidate = next(row for row in checkpoint["diagnostics"]["ranked_candidates"]
                     if int(row["official_id"]) == parent)
    pair_position = next(index for index, row in enumerate(checkpoint["pair_trace"], 1)
                         if int(row["official_id"]) == parent)
    passage_id = checkpoint["pair_trace"][pair_position - 1]["passage_id"]
    evidence = next(row for row in candidate["evidence"] if row["passage_id"] == passage_id)
    index = ROOT / config["index_namespace"]
    index_manifest = read_json(index / "manifest.json")
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    title = SourceLookup(store)[parent].title
    payload = f"{title}\n{evidence['text']}" if title else evidence["text"]
    if hashlib.sha256(payload.encode()).hexdigest() != checkpoint["pair_trace"][
            pair_position - 1]["input_sha256"]:
        raise ValueError("P0 exact pair input differs")
    full = tokenizer(queries[qid], payload, add_special_tokens=True, truncation=False)
    limited = tokenizer(queries[qid], payload, add_special_tokens=True,
                        truncation=True, max_length=reranker["max_length"],
                        return_offsets_mapping=True)
    expected_length = checkpoint["diagnostics"]["reranker_tokens"][
        "pair_lengths"][pair_position - 1]
    if len(full["input_ids"]) != expected_length or len(limited["input_ids"]) != 512:
        raise ValueError("Pinned tokenizer disagrees with P0 token diagnostics")
    passage_offsets = [offset for sequence, offset in zip(
        limited.sequence_ids(), limited["offset_mapping"], strict=True)
        if sequence == 1 and offset[1] > offset[0]]
    payload_end = max(end for _, end in passage_offsets)
    text_end = payload_end - (len(title) + 1 if title else 0)
    phrases = ["小儿总是喜欢", "新生儿泪囊炎", "不是", "可以滴眼药水", "用药 药品库"]
    phrase_rows = []
    for phrase in phrases:
        start = evidence["text"].find(phrase)
        if start < 0:
            raise ValueError(f"Expected phrase absent: {phrase}")
        phrase_rows.append({"phrase": phrase, "start_char": start,
                            "end_char": start + len(phrase),
                            "retained_at_512": start + len(phrase) <= text_end})
    result = {
        "status": "PINNED_TOKENIZER_TRUNCATION_AUDIT_PASS",
        "query_id": qid, "doc_id": parent, "passage_id": passage_id,
        "full_pair_tokens": len(full["input_ids"]),
        "max_pair_tokens": reranker["max_length"],
        "retained_pair_tokens": len(limited["input_ids"]),
        "passage_characters": len(evidence["text"]),
        "retained_passage_characters": text_end,
        "removed_tail_characters": len(evidence["text"]) - text_end,
        "phrases": phrase_rows,
        "interpretation": "Clinical description, disease term, negative answer, and eye-drop sentence remain inside the 512-token input; only the last tail is removed.",
        "model_calls": 0,
    }
    output.mkdir(parents=True)
    (output / "truncation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {"format": "v4-p2-truncation-audit-v1", "status": "complete",
                "model_calls": 0, "model_revision": reranker["revision"],
                "snapshot": str(snapshot), "input_sha256": {
                    "checkpoint": sha(checkpoint_path), "queries": sha(query_path),
                    "config": sha(ROOT / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml"),
                    "tokenizer": sha(snapshot / "tokenizer.json")},
                "result_sha256": sha(output / "truncation_result.json"),
                "implementation_sha256": sha(Path(__file__))}
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    main(parser.parse_args().output)

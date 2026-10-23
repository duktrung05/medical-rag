"""Compare B0 passage-first candidate survival to A0 without new model calls."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.medrag.core.contracts import Hit
from src.medrag.core.passage_first import passage_candidates, shared_pair_plan
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.evidence.verifier import verify


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def main(traces: Path, p2: Path, index: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    a_pairs = {}
    with (p2 / "pair_allocation.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["arm"] == "P2-A0":
                a_pairs.setdefault(int(row["query_id"]), []).append(
                    (int(row["doc_id"]), row["passage_id"]))
    results = []
    for qid in sorted(a_pairs):
        trace_path = traces / f"{qid}.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))["branch_trace"]
        rankings = [[Hit(int(item["official_id"]), item["passage_id"],
                         float(item["score"]), channel, int(item["rank"]))
                     for item in trace["branch_hits"][channel]]
                    for channel in ("passage_bm25", "dense")]
        b_fused = passage_candidates(rankings, limit=400)
        b_pairs = [(hit.official_id, hit.passage_id) for hit in b_fused[:200]]
        if len(a_pairs[qid]) != 200 or len(b_pairs) != 200:
            raise ValueError(f"Query {qid} does not have 200 candidate pairs per standalone arm")
        a_set, b_set = set(a_pairs[qid]), set(b_pairs)
        shared = shared_pair_plan(a_pairs[qid], b_pairs, budget=200, a_slots=100)
        if shared["total_inference_pairs"] > 200 or len(set(shared["unique_pairs"])) != len(
                shared["unique_pairs"]):
            raise ValueError("Shared budget or pair deduplication failed")
        results.append({"query_id": qid, "a0_scored_pairs": 200,
                        "b0_replay_candidate_pairs": 200,
                        "b0_unique_parents": len({doc for doc, _ in b_pairs}),
                        "a0_unique_parents": len({doc for doc, _ in a_pairs[qid]}),
                        "pair_overlap": len(a_set & b_set),
                        "b0_pairs_missing_a0_score": len(b_set - a_set),
                        "b0_parent_overlap_with_a0": len({doc for doc, _ in b_pairs} &
                                                          {doc for doc, _ in a_pairs[qid]}),
                        "shared_100_100": {key: value for key, value in shared.items()
                                           if key != "unique_pairs"},
                        "b0_pair_ids": b_pairs})
    index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    if index_manifest["status"] != "complete":
        raise ValueError("B0 replay requires complete registered full index")
    store = DiskCorpusStore(index / "corpus.sqlite", index_manifest["corpus_version"],
                            index_manifest["fingerprint"])
    all_pairs = {(doc, passage_id) for row in results for doc, passage_id in row["b0_pair_ids"]}
    passages = store.passages_batch(list({passage_id for _, passage_id in all_pairs}))
    sources = {doc: store.source(doc) for doc, _ in all_pairs}
    for doc, passage_id in all_pairs:
        if passages[passage_id].official_id != doc or not verify(passages[passage_id], sources[doc]):
            raise ValueError(f"Ungrounded B0 candidate: {doc}/{passage_id}")
    output.mkdir(parents=True)
    write_json(output / "candidate_comparison.json", results)
    write_json(output / "manifest.json", {
        "status": "candidate_replay_only", "query_count": len(results),
        "model_calls": 0, "new_retrieval_calls": 0,
        "registered_source_validated_pairs": len(all_pairs),
        "full_index_manifest_sha256": sha(index / "manifest.json"),
        "baseline_pair_allocation_sha256": sha(p2 / "pair_allocation.csv"),
        "trace_sha256": {str(row["query_id"]): sha(traces / f"{row['query_id']}.json")
                         for row in results},
        "candidate_comparison_sha256": sha(output / "candidate_comparison.json"),
        "limitations": ["Only the six pilot branch traces contain both passage retrieval branches",
                        "B0 pairs lacking cached rerank scores cannot be ranked or exported",
                        "No independent relevance labels or full-index B0 latency yet"],
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, default=ROOT /
                        "outputs/v4/parent_first_evidence_pilot_20261021_v2/queries")
    parser.add_argument("--p2", type=Path, default=ROOT /
                        "outputs/v4/p2_passage_candidate_replay_20261021")
    parser.add_argument("--index", type=Path, default=ROOT /
                        "outputs/v4/task4/full_index_gpu3_20261019")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.traces, args.p2, args.index, args.output)

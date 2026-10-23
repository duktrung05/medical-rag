"""Prepare an arm-blinded passage relevance review packet from the P2 pilot replay."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.selector import SelectionPolicy


DIAGNOSTIC_PARENTS = {
    (53, 128866), (1144, 142195),
    (53, 756661), (130, 553119), (130, 2968260),
    (754, 2713777), (771, 484565),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, data) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")


def main(replay: Path, output: Path) -> None:
    replay, output = replay.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    validation = json.loads((replay / "validation.json").read_text(encoding="utf-8"))
    if validation["status"] != "PASS_FULL_REPLAY":
        raise ValueError("Review packet requires complete P2 replay")
    migration = json.loads((ROOT / "configs/vibiomir/phase2_batch_migration_v1.json")
                           .read_text(encoding="utf-8"))
    query_path = ROOT / migration["official_queries"]
    if sha(query_path) != migration["official_queries_sha256"]:
        raise ValueError("Official query fingerprint changed")
    queries = {int(row["id"]): str(row["query"]) for row in pq.read_table(
        query_path, columns=["id", "query"]).to_pylist()}
    by_item = {}
    for line in (replay / "replay_selected_chunks.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        key = (int(row["query_id"]), int(row["doc_id"]),
               row["seed_passage_id"], row["context_text"])
        by_item.setdefault(key, set()).add(row["arm"])
    config = load_config(ROOT / "configs/v4_phase2_full_independent_d200_context_gpu3.yaml")
    index = ROOT / config["index_namespace"]
    index_manifest = json.loads((index / "manifest.json").read_text(encoding="utf-8"))
    store = DiskCorpusStore(index / "corpus.sqlite", config["corpus_version"],
                            index_manifest["fingerprint"])
    sources, passages = SourceLookup(store), PassageLookup(store)
    policy = SelectionPolicy(**config["selection"])
    for qid, doc_id in sorted(DIAGNOSTIC_PARENTS):
        checkpoint = json.loads((ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2"
                                 / "queries" / f"{qid}.json").read_text(encoding="utf-8"))
        candidate = next(row for row in checkpoint["diagnostics"]["ranked_candidates"]
                         if int(row["official_id"]) == doc_id)
        for evidence in candidate["evidence"]:
            seed = {**evidence, "score_status": "reranked", "rerank_score": 0.0}
            contextual, _ = apply_chunk_context(
                [{"official_id": doc_id, "evidence": [seed]}],
                policy, sources, passages)
            context_text = contextual[0]["evidence"][0]["text"]
            key = (qid, doc_id, evidence["passage_id"], context_text)
            by_item.setdefault(key, set()).add("diagnostic_candidate")
    if not by_item:
        raise ValueError("No P2 selected chunks available")
    output.mkdir(parents=True)
    annotations = []
    mapping = []
    with (output / "review_items.jsonl").open("w", encoding="utf-8") as stream:
        for key, arms in sorted(by_item.items()):
            qid, doc_id, passage_id, context_text = key
            item_id = hashlib.sha256(json.dumps(key, ensure_ascii=False,
                                                separators=(",", ":")).encode()).hexdigest()[:20]
            item = {"item_id": item_id, "query": queries[qid],
                    "passage_context": context_text}
            stream.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")) + "\n")
            mapping.append({"item_id": item_id, "query_id": qid, "doc_id": doc_id,
                            "seed_passage_id": passage_id, "arms": sorted(arms)})
            annotations.append({"item_id": item_id, "answers_question_0_1_2": "",
                                "patient_context_match_yes_no_unknown": "",
                                "numeric_match_yes_no_na": "",
                                "contradicts_query_yes_no": "",
                                "source_grounded_yes_no": "",
                                "reviewer_notes": ""})
    with (output / "annotation_template.csv").open("w", encoding="utf-8",
                                                     newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(annotations[0]))
        writer.writeheader()
        writer.writerows(annotations)
    write_json(output / "item_map_for_analysis.json", mapping)
    rubric = {
        "scope": "six-query diagnostic pilot including selected chunks and seven diagnostic parents; review items are blinded to arms, scores, ranks, and historical anchors",
        "answers_question_0_1_2": {
            "0": "Does not answer, or gives contradictory/wrong-subject evidence",
            "1": "Related topic or partial evidence, but missing the requested answer or context",
            "2": "Direct answer in the right patient/value context, grounded in the passage",
        },
        "review_instructions": [
            "Read query and passage_context without opening item_map_for_analysis.json.",
            "Judge patient age/condition and requested numeric value separately from topic overlap.",
            "Fill the annotation template independently; resolve disagreements before unblinding arms.",
            "Do not use historical anchors or pilot scores as relevance labels.",
        ],
        "status": "template_only_unreviewed",
    }
    write_json(output / "rubric.json", rubric)
    names = ["review_items.jsonl", "annotation_template.csv",
             "item_map_for_analysis.json", "rubric.json"]
    write_json(output / "manifest.json", {
        "format": "v4-p2-blinded-relevance-review-packet-v1",
        "status": "template_only_unreviewed",
        "items": len(mapping),
        "source_replay_sha256": sha(replay / "replay_selected_chunks.jsonl"),
        "diagnostic_parents": len(DIAGNOSTIC_PARENTS),
        "query_file_sha256": sha(query_path),
        "artifacts_sha256": {name: sha(output / name) for name in names},
        "implementation_sha256": sha(Path(__file__)),
    })
    print(json.dumps({"status": "template_only_unreviewed", "items": len(mapping),
                      "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    main(args.replay, args.output)

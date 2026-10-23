"""Replay V4 evidence selection from a complete scored subset trace on CPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.evidence.selector import SelectionPolicy, select_results


def replay(source: Path, config_path: Path, output: Path, *, result_limit: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    source_manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if source_manifest.get("status") != "complete" or source_manifest.get("scope") != "locked_v4_subset":
        raise ValueError("Input must be a complete V4 subset inference run")
    actual_diagnostics_sha = hashlib.sha256((source / "diagnostics.jsonl").read_bytes()).hexdigest()
    if actual_diagnostics_sha != source_manifest.get("diagnostics_sha256"):
        raise ValueError("Scored trace checksum differs from inference manifest")
    base_config = source_manifest.get("config")
    if not isinstance(base_config, dict):
        raise ValueError("Inference run did not record its complete configuration")
    config = load_config(config_path)
    for key in ("corpus_version", "registry", "lexical", "dense", "reranker", "retrieval", "fusion"):
        if config.get(key) != base_config.get(key):
            raise ValueError(f"Replay may not change the inference contract: {key}")
    selection = config.get("selection", {})
    policy = SelectionPolicy(max_docs=int(selection.get("max_docs", 200)),
                             max_chunks=int(selection.get("max_chunks", 60)),
                             chunks_per_doc=int(selection.get("chunks_per_doc", 1)),
                             document_selection=str(selection.get("document_selection", "with_chunks")),
                             chunk_context=str(selection.get("chunk_context", "passage")),
                             context_window=int(selection.get("context_window", 1)),
                             context_max_characters=int(selection.get("context_max_characters", 4096)))
    if policy.chunk_context != "passage":
        raise ValueError("This replay trace has no corpus store for adjacent context grounding")
    limit = source_manifest["result_limit"] if result_limit is None else result_limit
    if type(limit) is not int or limit <= 0:
        raise ValueError("Replay result limit must be a positive integer")
    predictions = []
    summaries = []
    seen = set()
    for line in (source / "diagnostics.jsonl").open(encoding="utf-8"):
        row = json.loads(line)
        qid = int(row["id"])
        if qid in seen:
            raise ValueError(f"Duplicate query ID in score trace: {qid}")
        seen.add(qid)
        candidates = row.get("ranked_candidates")
        if not isinstance(candidates, list):
            raise ValueError("Score trace lacks ranked candidates")
        selected, diagnostics = select_results(
            candidates, policy,
            reranker_enabled=bool(config.get("reranker", {}).get("enabled", False)),
            limit=limit,
        )
        predictions.extend(records([qid], {qid: selected}))
        summaries.append({"id": qid, "selection": diagnostics})
    if [row["id"] for row in predictions] != source_manifest["query_ids"]:
        raise ValueError("Score trace query coverage/order differs from inference manifest")
    payload = json.dumps(predictions, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    output.mkdir(parents=True)
    (output / "predictions.json").write_text(payload + "\n", encoding="utf-8")
    (output / "diagnostics.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=False) + "\n" for row in summaries), encoding="utf-8")
    manifest = {"status": "complete", "scope": "cpu_selection_replay",
                "source_manifest_sha256": hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
                "source_diagnostics_sha256": source_manifest["diagnostics_sha256"],
                "selection_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "selection": selection, "result_limit": limit,
                "source_result_limit": source_manifest["result_limit"],
                "selected_documents": sum(len(row["relevant_docs"]) for row in predictions),
                "selected_chunks": sum(len(row["relevant_chunks"]) for row in predictions),
                "implementation_sha256": {name: hashlib.sha256(
                    (Path(__file__).resolve().parents[2] / name).read_bytes()).hexdigest() for name in (
                        "scripts/v4/replay_subset_selection.py", "src/medrag/evidence/selector.py",
                        "src/medrag/adapters/vibiomir_submission.py")},
                "query_count": len(predictions),
                "canonical_predictions_sha256": hashlib.sha256(payload.encode()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--result-limit", type=int,
                        help="Override only the selection limit on the same scored candidates; no inference")
    args = parser.parse_args()
    print(json.dumps(replay(args.source.resolve(), args.config.resolve(), args.output.resolve(),
                            result_limit=args.result_limit), indent=2))

"""Rescore one exact historical batch to diagnose R7 anchor batch effects."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path

from scripts.v4.run_p5_2_split import ROOT, read
from src.medrag.data.phase2_batch import sha256_file


BASE = ROOT / "outputs/v4/a0_fast_a6000_final_20261023"
RUNTIME = ROOT / "outputs/v4/p5_gpu_incremental_a6000_20261023/runtime_lock.json"


def run(query_id: int, batch_start: int) -> dict:
    runtime = read(RUNTIME)
    pair_path = BASE / "pairs" / f"{query_id}.json"
    score_path = BASE / "scores" / f"{query_id}.json"
    pair, score = read(pair_path), read(score_path)
    if not 0 <= batch_start < len(pair["pairs"]) or batch_start % runtime["batch_size"]:
        raise ValueError("batch_start must be an existing historical batch boundary")
    pair_rows = pair["pairs"][batch_start:batch_start + runtime["batch_size"]]
    score_rows = score["scores"][batch_start:batch_start + runtime["batch_size"]]
    if len(pair_rows) != len(score_rows):
        raise ValueError("Historical pair/score batch length differs")
    for left, right in zip(pair_rows, score_rows, strict=True):
        if any(left[key] != right[key] for key in
               ("official_id", "passage_id", "input_sha256", "cache_key")):
            raise ValueError("Historical pair/score identity differs")
        if hashlib.sha256(left["payload"].encode()).hexdigest() != left["input_sha256"]:
            raise ValueError("Historical input payload changed")
    actual_versions = {name: importlib.metadata.version(name) for name in runtime["versions"]}
    if actual_versions != runtime["versions"]:
        raise ValueError("Runtime versions differ")
    import torch
    if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
            or torch.version.cuda != runtime["cuda"]
            or torch.cuda.get_device_name(0) != runtime["gpu_name"]):
        raise ValueError("Pinned single-GPU runtime differs")
    torch.set_num_threads(runtime["num_threads"])
    torch.set_num_interop_threads(runtime["num_interop_threads"])
    torch.backends.cuda.matmul.allow_tf32 = runtime["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = runtime["cudnn_allow_tf32"]
    from src.medrag.core.model_revision import require_cached_revision
    contract = runtime["score_contract"]
    snapshot = require_cached_revision(contract["model"], contract["revision"])
    for name, expected in runtime["model_snapshot_files_sha256"].items():
        if sha256_file(snapshot / name) != expected:
            raise ValueError("Pinned model snapshot differs")
    from src.medrag.ranking.reranker import BGEReranker
    model = BGEReranker(contract["model"], revision=contract["revision"],
                        max_length=contract["max_length"], device="cuda:0")
    values = model.score(pair["query"], [row["payload"] for row in pair_rows])
    differences = [abs(float(value) - float(old["score"]))
                   for value, old in zip(values, score_rows, strict=True)]
    if any(not math.isfinite(value) for value in values):
        raise ValueError("Fresh score is non-finite")
    return {
        "format": "v4-r7-exact-historical-batch-diagnostic-v1",
        "query_id": query_id, "batch_start": batch_start,
        "batch_count": len(pair_rows), "gpu_uuid_selector": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pair_sha256": sha256_file(pair_path), "score_sha256": sha256_file(score_path),
        "maximum_absolute_difference": max(differences),
        "exact_scores": sum(value == 0 for value in differences),
        "within_1e_3": sum(value <= 1e-3 for value in differences),
        "differences": differences,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query_id", type=int)
    parser.add_argument("batch_start", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.query_id, args.batch_start), indent=2))

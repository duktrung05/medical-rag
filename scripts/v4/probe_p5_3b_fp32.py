"""Run a locked 12-pair CPU FP32 diagnostic without changing frozen scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import time
from importlib.metadata import version
from pathlib import Path

from scripts.v4.run_p5_2_split import PILOT_IDS, ROOT, read
from src.medrag.data.phase2_batch import sha256_file

SOURCE = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022"
CANARY = ROOT / "outputs/v4/p5_3a_colab_canary_20261022"
AUDIT = ROOT / "outputs/v4/p5_3b_cpu_gpu_parity_20261022"
REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
SNAPSHOT = ROOT / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots" / REVISION
PINNED = {"sentence-transformers": "4.1.0", "transformers": "4.51.3",
          "huggingface-hub": "0.36.2", "tokenizers": "0.21.4"}


def canonical(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False) + "\n").encode("utf-8")


def main(subset_path: Path, output: Path, snapshot: Path) -> dict:
    subset_path, output, snapshot = subset_path.resolve(), output.resolve(), snapshot.resolve()
    subset = read(subset_path)
    if (subset.get("format") != "medrag-v4-p5-3b-fp32-subset-v1"
            or subset.get("query_ids") != list(PILOT_IDS)
            or subset.get("pair_count") != 12
            or subset.get("model_revision") != REVISION
            or not snapshot.is_dir() or snapshot.name != REVISION):
        raise ValueError("Locked FP32 subset/model snapshot identity mismatch")
    for package, expected in PINNED.items():
        if version(package) != expected:
            raise ValueError(f"Pinned diagnostic package mismatch: {package}")

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from sentence_transformers import CrossEncoder

    torch.set_num_threads(min(4, os.cpu_count() or 1))
    model = CrossEncoder(str(snapshot), max_length=512, device="cpu",
                         model_kwargs={"torch_dtype": torch.float32},
                         activation_fn=torch.nn.Identity())
    if ({parameter.dtype for parameter in model.model.parameters()} != {torch.float32}
            or model.max_length != 512):
        raise ValueError("FP32 diagnostic model dtype/max length differs")

    inputs, metadata = [], []
    for query_row in subset["queries"]:
        qid = int(query_row["query_id"])
        pair_path = SOURCE / "pairs" / f"{qid}.json"
        if sha256_file(pair_path) != query_row["pair_shard_sha256"]:
            raise ValueError(f"FP32 subset pair shard changed: {qid}")
        pair = read(pair_path)
        if hashlib.sha256(pair["query"].encode()).hexdigest() != query_row["query_sha256"]:
            raise ValueError(f"FP32 subset query changed: {qid}")
        baseline = read(SOURCE / "scores" / f"{qid}.json")["scores"]
        gpu = read(CANARY / "verification_0434ce1c30a2/scores" / f"{qid}.json")["scores"]
        for position, index in enumerate(query_row["pair_indices"]):
            item = pair["pairs"][index]
            if (item["cache_key"] != query_row["cache_keys"][position]
                    or item["input_sha256"] != query_row["input_sha256"][position]
                    or baseline[index]["cache_key"] != item["cache_key"]
                    or gpu[index]["cache_key"] != item["cache_key"]):
                raise ValueError(f"FP32 subset pair identity changed: {qid}/{index}")
            inputs.append([pair["query"], item["payload"]])
            full_tokens = model.tokenizer(pair["query"], item["payload"],
                                          add_special_tokens=True, truncation=False)["input_ids"]
            truncated = model.tokenizer(pair["query"], item["payload"], add_special_tokens=True,
                                        truncation=True, max_length=512)["input_ids"]
            metadata.append({"query_id": qid, "pair_index": index,
                             "cache_key": item["cache_key"],
                             "official_id": item["official_id"],
                             "passage_id": item["passage_id"],
                             "input_sha256": item["input_sha256"],
                             "full_token_count": len(full_tokens),
                             "truncated_token_count": len(truncated),
                             "truncated": len(full_tokens) > 512,
                             "baseline_score": float(baseline[index]["score"]),
                             "gpu_t4_score": float(gpu[index]["score"])})

    started = time.perf_counter()
    batch_scores = [float(value) for value in model.predict(
        inputs, batch_size=32, show_progress_bar=False)]
    batch_seconds = time.perf_counter() - started
    started = time.perf_counter()
    single_scores = [float(value) for value in model.predict(
        inputs, batch_size=1, show_progress_bar=False)]
    single_seconds = time.perf_counter() - started
    if len(batch_scores) != 12 or len(single_scores) != 12:
        raise ValueError("FP32 diagnostic score coverage differs")
    rows = []
    for item, batched, single in zip(metadata, batch_scores, single_scores, strict=True):
        if not all(math.isfinite(value) for value in (batched, single)):
            raise ValueError("FP32 diagnostic produced a non-finite score")
        rows.append({**item, "fp32_batch12_score": batched, "fp32_batch1_score": single,
                     "fp32_padding_difference": batched - single,
                     "fp32_vs_baseline": batched - item["baseline_score"],
                     "fp32_vs_gpu_t4": batched - item["gpu_t4_score"]})
    def mean(values) -> float:
        materialized = list(values)
        return sum(materialized) / len(materialized)
    report = {
        "format": "medrag-v4-p5-3b-fp32-probe-v1",
        "status": "COMPLETE_DIAGNOSTIC_ONLY",
        "subset_manifest_sha256": sha256_file(subset_path),
        "model": subset["model"], "model_revision": REVISION,
        "snapshot_model_sha256": sha256_file(snapshot / "model.safetensors"),
        "device": "cpu", "dtype": "torch.float32", "autocast": False,
        "max_length": 512, "truncation": True,
        "batch_modes": [12, 1], "pair_count": 12,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    **{name: version(name) for name in PINNED}},
        "benchmark": {"batch12_seconds": batch_seconds, "batch1_seconds": single_seconds},
        "summary": {
            "max_absolute_padding_difference": max(abs(row["fp32_padding_difference"])
                                                       for row in rows),
            "mean_absolute_fp32_vs_baseline": mean(abs(row["fp32_vs_baseline"])
                                                     for row in rows),
            "mean_absolute_fp32_vs_gpu_t4": mean(abs(row["fp32_vs_gpu_t4"])
                                                   for row in rows),
            "fp32_closer_to_baseline_pairs": sum(abs(row["fp32_vs_baseline"]) <
                                                  abs(row["fp32_vs_gpu_t4"]) for row in rows),
            "fp32_closer_to_gpu_pairs": sum(abs(row["fp32_vs_gpu_t4"]) <
                                             abs(row["fp32_vs_baseline"]) for row in rows),
            "truncated_pairs": sum(row["truncated"] for row in rows),
        },
        "rows": rows,
        "limitations": [
            "This is a locked 12-pair diagnostic, not a replacement score artifact.",
            "The model/tokenizer/inference libraries are pinned, but local torch differs from the T4 torch build.",
            "Batch-12 padding is a diagnostic shape and does not reproduce every original 32-pair batch companion.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical(report))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subset", type=Path, default=AUDIT / "fp32_subset_manifest.json")
    parser.add_argument("--output", type=Path, default=AUDIT / "fp32_probe.json")
    parser.add_argument("--snapshot", type=Path, default=SNAPSHOT)
    args = parser.parse_args()
    print(json.dumps(main(args.subset, args.output, args.snapshot), indent=2, ensure_ascii=False))

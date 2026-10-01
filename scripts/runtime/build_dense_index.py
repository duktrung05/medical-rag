"""Build a cached exact dense passage index from a configured corpus."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from src.config import load_pipeline_config
from src.data.loader import DataLoader
from src.retrieval.dense import DenseRetriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/baseline_dense.yaml")
    parser.add_argument("--chunks", type=str, help="Optional chunks JSONL override")
    parser.add_argument("--output-index", type=str, help="Optional index directory override")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--timeout-seconds", type=int, default=3600,
                        help="Hard timeout for the entire build including model loading (default: 3600)")
    parser.add_argument("--limit", type=int, help="Build only the first N sorted chunks for a resource probe")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error("timeout-seconds and limit must be positive")
    if not args.worker:
        command = [sys.executable, "-m", "scripts.runtime.build_dense_index", *sys.argv[1:], "--worker"]
        started = time.monotonic()
        print(f"Starting supervised build; hard timeout={args.timeout_seconds}s", flush=True)
        try:
            result = subprocess.run(command, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f"BUILD TIMED OUT after {args.timeout_seconds}s; index is not complete", file=sys.stderr)
            raise SystemExit(124)
        print(f"Build subprocess exit={result.returncode}; elapsed={time.monotonic()-started:.1f}s", flush=True)
        raise SystemExit(result.returncode)

    config = load_pipeline_config(args.config)
    if not config.retrieval.dense.enabled:
        raise ValueError("Dense index build requires retrieval.dense.enabled=true")
    dense_config = config.retrieval.dense
    if args.output_index:
        dense_config = dense_config.model_copy(update={"index_path": Path(args.output_index).resolve()})
    chunks = DataLoader.load_chunks(args.chunks or config.corpus)
    if args.limit:
        if not args.output_index:
            parser.error("--limit requires --output-index to protect the full corpus index")
        chunks = sorted(chunks, key=lambda chunk: chunk.chunk_id)[:args.limit]
    target = Path(dense_config.index_path)
    if target.exists() and any(target.iterdir()):
        raise FileExistsError(f"Output index is not empty: {target}; choose a new --output-index")
    staging = target.with_name(target.name + ".building")
    if staging.exists():
        raise FileExistsError(f"Staging directory already exists: {staging}; inspect it before retrying")
    dense_config = dense_config.model_copy(update={"index_path": staging})
    import psutil
    import torch
    if (args.device or dense_config.device) == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA build requested but PyTorch cannot use GPU; refusing a silent CPU build")
    print(f"Resources: available_ram_gb={psutil.virtual_memory().available / 2**30:.2f}; "
          f"cuda={torch.cuda.is_available()}; passages={len(chunks)}; "
          f"batch={dense_config.batch_size}; precision={dense_config.precision}", flush=True)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    manifest = DenseRetriever.build_index(chunks, dense_config, device=args.device, progress=True)
    manifest["build_elapsed_seconds"] = time.monotonic() - started
    manifest["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0
    manifest["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0
    manifest["build_timeout_seconds"] = args.timeout_seconds
    (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    # Publish only after all embeddings and the manifest have been written.
    if target.exists():
        target.rmdir()  # Only the previously checked empty target can be removed.
    staging.rename(target)
    print(
        f"Built exact dense index with {manifest['num_passages']} chunks at "
        f"{target} using model commit {manifest['model_revision']} "
        f"in {manifest['build_elapsed_seconds']:.1f}s"
    )


if __name__ == "__main__":
    main()

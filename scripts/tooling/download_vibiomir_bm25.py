"""Download only a ViBioMIR BM25 index and its lookup tables, pinned to one revision."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.errors import HfHubHTTPError

ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def download(args) -> None:
    info = HfApi().repo_info(args.repo_id, repo_type="dataset", revision=args.revision,
                             files_metadata=True)
    prefix = f"data/vibiomir/{args.index_name}"
    required = {f"{prefix}_ids.npy", f"{prefix}_vocab.json"}
    if not args.index_only:
        required.update({"data/raw/vibiomir/query.parquet",
                         "data/raw/vibiomir/links_corpus.parquet"})
    by_name = {entry.rfilename: entry for entry in info.siblings}
    missing = required - by_name.keys()
    if missing:
        raise FileNotFoundError(f"Missing companion files on Hub: {sorted(missing)}")
    files = sorted(name for name in by_name if name in required or name.startswith(prefix + "/"))
    if not any(name.startswith(prefix + "/") for name in files):
        raise FileNotFoundError(f"No index files at {prefix}")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    pending = 0
    for name in files:
        entry = by_name[name]
        if entry.size is None:
            raise ValueError(f"Unknown remote size: {name}")
        target = (destination / name).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe remote path: {name}")
        checksum = entry.lfs.sha256 if entry.lfs else None
        # Never replace different local data silently.
        exists = target.exists()
        if exists and (target.stat().st_size != entry.size or
                       (checksum and sha256(target) != checksum)):
            raise FileExistsError(f"Local file differs from pinned snapshot: {target}")
        if not exists:
            pending += entry.size
        records.append({"path": name, "size": entry.size, "sha256": checksum})
    gib = 1024 ** 3
    print(f"Revision {info.sha}: {len(files)} files, {pending / gib:.2f} GiB to download", flush=True)
    if pending > args.max_gib * gib:
        raise ValueError(f"Download exceeds --max-gib {args.max_gib}; inspect size before increasing it")
    if shutil.disk_usage(destination).free < pending + gib:
        raise OSError("Insufficient free disk space including 1 GiB reserve")
    for record in records:
        target = destination / record["path"]
        if not target.exists():
            print(f"Downloading {record['path']} ({record['size'] / gib:.3f} GiB)", flush=True)
            hf_hub_download(args.repo_id, record["path"], repo_type="dataset",
                            revision=info.sha, local_dir=destination)
        if target.stat().st_size != record["size"]:
            raise ValueError(f"Downloaded size mismatch: {target}")
        if record["sha256"] and sha256(target) != record["sha256"]:
            raise ValueError(f"Downloaded checksum mismatch: {target}")
    manifest = destination / "data/vibiomir" / f"{args.index_name}_download.json"
    manifest.write_text(json.dumps({"repo_id": args.repo_id, "revision": info.sha,
                                   "files": records}, indent=2) + "\n", encoding="utf-8")
    print(f"Verified download; manifest: {manifest}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="duktrung/med-rag")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--index-name", default="bm25idx_crawl_20261007")
    parser.add_argument("--destination", type=Path, default=ROOT)
    parser.add_argument("--index-only", action="store_true", help="Omit query and URL lookup tables")
    parser.add_argument("--max-gib", type=float, default=64)
    parser.add_argument("--xet-fixed-concurrency", type=int, default=8,
                        help="Pin Xet download streams; use 0 for adaptive concurrency")
    parser.add_argument("--xet-chunk-cache-gib", type=float, default=10,
                        help="Xet chunk cache size so interrupted downloads can reuse chunks")
    args = parser.parse_args()
    if not args.index_name or Path(args.index_name).name != args.index_name or args.index_name in {".", ".."}:
        parser.error("--index-name must be a single directory name")
    if args.xet_fixed_concurrency < 0 or args.xet_chunk_cache_gib < 0:
        parser.error("Xet concurrency and cache size cannot be negative")
    if args.xet_fixed_concurrency:
        os.environ["HF_XET_FIXED_DOWNLOAD_CONCURRENCY"] = str(args.xet_fixed_concurrency)
    os.environ["HF_XET_CHUNK_CACHE_SIZE_BYTES"] = str(
        int(args.xet_chunk_cache_gib * 1024 ** 3)
    )
    try:
        download(args)
    except HfHubHTTPError as error:
        if error.response is not None and error.response.status_code in {401, 403}:
            raise SystemExit(
                "Hugging Face access denied. Log in with Python:\n"
                ' .venv\\Scripts\\python.exe -c "from huggingface_hub import login; login()"\n'
                "Use an account allowed to read the dataset, then retry."
            ) from None
        raise


if __name__ == "__main__":
    main()

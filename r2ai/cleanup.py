"""Remove intermediate artefacts that earlier pipeline generations left behind.

Three kinds of waste accumulate across runs:

* Superseded generations. Each rebuild of the corpus produces its own chunk
  file, document vectors and embedding cache; once a later generation exists the
  earlier ones are only reproducible history.
* Duplicated submissions. Every submission is written as both ``.json`` and the
  ``.zip`` that actually gets uploaded, so the uncompressed copy is dead weight.
* Smoke-test leftovers from one-off experiments.

Crawled pages are never touched: they cost days of network time and several
hosts have since started blocking us, so they cannot be reproduced.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/vibiomir"
SUB = ROOT / "outputs/submissions"

# Superseded by the chunks_v4 / docidx4 / bm25idx generation.
SUPERSEDED = [
    "emb_chunks_full_docidx_candidates.npz",
    "emb_chunks_docidx_small_candidates.npz",
    "emb_chunks.npz",
    "chunks_full.parquet",
    "chunks.parquet",
    "chunks_smoke.parquet",
    "docidx_vecs.npy", "docidx_ids.npy", "docidx_candidates.parquet",
    "docidx_small_vecs.npy", "docidx_small_ids.npy", "docidx_small_candidates.parquet",
    # The day-one slug index, replaced by the real lexical index over page text.
    "url_index_meta.parquet", "url_candidates.parquet",
]
SUPERSEDED_DIRS = ["url_bm25"]

# Needed by the current pipeline; listed so the script documents what it protects.
PROTECTED = {
    "chunks_v4.parquet", "docidx4_vecs.npy", "docidx4_ids.npy",
    "docidx4_candidates.parquet", "bm25idx_ids.npy", "bm25idx_vocab.json",
    "bm25idx_candidates.parquet", "fused_candidates.parquet",
    "emb_chunks_v4_docidx4_candidates.npz",
}


def in_use(path: Path) -> bool:
    """True if any running process holds this file open."""
    for fd_dir in Path("/proc").glob("[0-9]*/fd"):
        try:
            for fd in fd_dir.iterdir():
                try:
                    if fd.resolve() == path.resolve():
                        return True
                except OSError:
                    continue
        except (OSError, PermissionError):
            continue
    return False


def plan() -> tuple[list[Path], int]:
    targets: list[Path] = []
    for name in SUPERSEDED:
        path = DATA / name
        if path.exists() and name not in PROTECTED:
            targets.append(path)
    for name in SUPERSEDED_DIRS:
        path = DATA / name
        if path.is_dir():
            targets.append(path)
    # Submission JSON duplicates the zip that is actually uploaded.
    targets.extend(p for p in sorted(SUB.glob("*.json")) if (p.with_suffix(".zip")).exists())
    total = 0
    for path in targets:
        if path.is_dir():
            total += sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        else:
            total += path.stat().st_size
    return targets, total


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    targets, total = plan()
    print(f"{'size':>10}  path")
    for path in targets:
        size = (sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir() else path.stat().st_size)
        print(f"{size / 1e9:>8.2f} GB  {path.relative_to(ROOT)}")
    print(f"\n{len(targets)} items, {total / 1e9:.2f} GB")

    if not args.apply:
        print("(dry run — pass --apply to delete)")
        raise SystemExit(0)

    freed = skipped = 0
    for path in targets:
        if path.is_file() and in_use(path):
            print(f"SKIP (đang mở): {path.name}")
            skipped += 1
            continue
        size = (sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                if path.is_dir() else path.stat().st_size)
        shutil.rmtree(path) if path.is_dir() else path.unlink()
        freed += size
    print(f"\nfreed {freed / 1e9:.2f} GB, skipped {skipped} in-use files")

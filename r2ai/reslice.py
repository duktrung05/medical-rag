"""Cut an existing submission down to smaller K without re-running the GPU.

`relevant_docs` and `relevant_chunks` are already ranked best-first, so taking a
prefix of each is exactly what emitting a smaller K would have produced. That
makes new leaderboard probes free once one expensive ranking run exists, and it
is the way to bring a submission under the platform's upload cap: documents cost
~8 bytes each while chunks carry their full text, so K_docs can stay large while
K_chunks is trimmed to fit.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from r2ai.rank import write_submission

UPLOAD_CAP_MB = 50.0


def load(path: Path) -> list[dict]:
    if path.suffix == ".zip":
        with zipfile.ZipFile(path) as zf:
            return json.loads(zf.read(zf.namelist()[0]))
    return json.loads(path.read_text(encoding="utf-8"))


def reslice(source: Path, k_docs: int, k_chunks: int, out_name: str) -> float:
    records = load(source)
    sliced = []
    for rec in records:
        docs = rec["relevant_docs"][:k_docs]
        keep = set(docs)
        # A chunk whose parent document fell outside K_docs can never be scored:
        # the metric only compares a predicted chunk against reference chunks in
        # the same document, and that document is no longer claimed.
        chunks = [c for c in rec["relevant_chunks"] if c["doc_id"] in keep][:k_chunks]
        sliced.append({"id": rec["id"], "relevant_docs": docs, "relevant_chunks": chunks})

    zip_path = write_submission(sliced, out_name)
    size_mb = zip_path.stat().st_size / 1e6

    avg_d = sum(len(r["relevant_docs"]) for r in sliced) / len(sliced)
    avg_c = sum(len(r["relevant_chunks"]) for r in sliced) / len(sliced)
    flag = "  <-- OVER CAP" if size_mb > UPLOAD_CAP_MB else ""
    print(f"{zip_path.name:28s} docs~{avg_d:6.1f} chunks~{avg_c:6.1f} "
          f"zip {size_mb:6.1f} MB{flag}")
    return size_mb


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path, help="existing submission .zip or .json")
    ap.add_argument("--k-docs", type=int, required=True)
    ap.add_argument("--k-chunks", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    reslice(a.source, a.k_docs, a.k_chunks, a.out)

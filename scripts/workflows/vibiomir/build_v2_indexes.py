"""Build bounded, isolated V2 pilot indexes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.retrieval_v2.core import build_dense, build_sparse
from src.retrieval_v2.settings import DEFAULT_CONFIG, cached_model_path, load_config

ROOT = Path(__file__).resolve().parents[3]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--pilot", type=Path, default=ROOT / "outputs/retrieval_v2/pilot")
    ap.add_argument("--target-words", type=int)
    ap.add_argument("--dense", action="store_true", help="bounded BGE-M3 pilot; separate process")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu or cuda:N; GPU requires prior resource approval")
    args = ap.parse_args()
    config = load_config(args.config)
    db = args.pilot / "v2_sparse.sqlite"
    manifest = args.pilot / "V2_INDEX_MANIFEST.json"
    result = json.loads(manifest.read_text(encoding="utf-8")) if manifest.exists() else {}
    if args.dense:
        result["dense"] = build_dense(db, args.pilot / "v2_dense.faiss",
                                      cached_model_path(config["dense"]["model"], config["dense"]["revision"]),
                                      args.batch_size, args.device)
    else:
        result["sparse"] = build_sparse(args.pilot / "pilot_source.parquet", db,
                                        target_words=args.target_words or config["chunking"]["target_words"])
    manifest.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

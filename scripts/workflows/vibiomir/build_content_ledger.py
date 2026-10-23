"""Build a revision-bound content ledger from existing ViBioMIR artifacts.

This is an inventory. File presence and legacy chunks are candidates for reuse,
not proof that the saved HTML or extracted text is good. Source files are read only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STEP1 = ROOT / "outputs/vibiomir_step1_20261013/manifest.json"
DEFAULT_PUBMED = (ROOT / "data/vibiomir/pubmed/71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
                  / "run_20261013_2gpu")
DEFAULT_OUTPUT = ROOT / "outputs/vibiomir_step2_20261013"
SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("source_kind", pa.string()), ("content_state", pa.string()),
    ("raw_path", pa.string()), ("raw_verified", pa.bool_()),
    ("legacy_chunks_present", pa.bool_()), ("legacy_bm25_present", pa.bool_()),
    ("legacy_dense_present", pa.bool_()), ("legacy_manifest_status", pa.string()),
    ("pubmed_status", pa.string()), ("pubmed_raw_path", pa.string()),
    ("pubmed_text_sha256", pa.string()),
])
MANIFEST_CODES = {"ok": 1, "circuit_open": 2, "http_error": 3,
                  "robots_denied": 4, "blocked": 5, "timeout": 6,
                  "network_error": 7, "error": 8}
MANIFEST_LABELS = {v: k for k, v in MANIFEST_CODES.items()}
MANIFEST_LABELS.update({0: "no_manifest", 9: "unknown_status"})


def file_sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_pubmed(directory: Path, revision: str) -> dict[int, tuple[str, str, str, str]]:
    release = json.loads((directory / "release_manifest.json").read_text())
    if not release.get("complete") or release["corpus_revision"] != revision:
        raise ValueError("PubMed release is incomplete or uses another corpus revision")
    result = {}
    for part in sorted((directory / "documents").glob("part_*.parquet")):
        for batch in pq.ParquetFile(part).iter_batches(
                batch_size=5000, columns=["id", "official_url", "status", "raw_file", "text_sha256"]):
            cols = batch.to_pydict()
            for ident, url, status, raw, digest in zip(
                    cols["id"], cols["official_url"], cols["status"],
                    cols["raw_file"], cols["text_sha256"], strict=True):
                if ident in result:
                    raise ValueError(f"Duplicate PubMed ID: {ident}")
                result[ident] = (url, status, raw or "", digest or "")
    if len(result) != 240_000 or Counter(v[1] for v in result.values()) != release["states"]:
        raise ValueError("PubMed document export does not match release manifest")
    return result


def load_legacy_flags(bound: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    crawl = ROOT / "data/vibiomir/crawl"
    latest = np.zeros(bound, dtype=np.uint8)
    for manifest in sorted(crawl.glob("manifest_*.parquet")):
        for batch in pq.ParquetFile(manifest).iter_batches(
                batch_size=50_000, columns=["doc_id", "status"]):
            cols = batch.to_pydict()
            for ident, status in zip(cols["doc_id"], cols["status"], strict=True):
                if ident is not None and 0 <= ident < bound:
                    latest[ident] = MANIFEST_CODES.get(status, 9)
    chunks = np.zeros(bound, dtype=np.bool_)
    chunk_path = ROOT / "data/vibiomir/chunks_crawl_20261007.parquet"
    for batch in pq.ParquetFile(chunk_path).iter_batches(
            batch_size=500_000, columns=["doc_id"]):
        ids = batch.column(0).to_numpy(zero_copy_only=False)
        ids = ids[(ids >= 0) & (ids < bound)]
        chunks[ids] = True
    flags = []
    for filename in ("bm25idx_crawl_20261007_ids.npy", "docidx_crawl_20261007_ids.npy"):
        array = np.zeros(bound, dtype=np.bool_)
        ids = np.load(ROOT / "data/vibiomir" / filename, mmap_mode="r", allow_pickle=False)
        if ids.min() < 0 or ids.max() >= bound:
            raise ValueError(f"Legacy index ID outside expected bound: {filename}")
        array[ids] = True
        flags.append(array)
    return latest, chunks, flags[0], flags[1]


def scan_raw_filenames(bound: int) -> np.ndarray:
    pages = np.zeros(bound, dtype=np.bool_)
    root = ROOT / "data/vibiomir/crawl/pages"
    for shard in sorted(root.iterdir()):
        if not shard.is_dir():
            continue
        with os.scandir(shard) as entries:
            for entry in entries:
                if not entry.name.endswith(".gz"):
                    continue
                stem = entry.name[:-3]
                if stem.isdigit() and 0 <= int(stem) < bound:
                    pages[int(stem)] = True
    return pages


def run(step1_path: Path, pubmed_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Choose a new output directory: {output}")
    step1 = json.loads(step1_path.read_text())
    if step1["status"] != "complete":
        raise ValueError("Step 1 snapshot is not complete")
    revision = step1["revision"]
    corpus_path = Path(step1["files"]["links_corpus_vi_en_zh.parquet"]["path"])
    if file_sha(corpus_path) != step1["files"]["links_corpus_vi_en_zh.parquet"]["sha256"]:
        raise ValueError("Official corpus checksum changed")
    official = pq.ParquetFile(corpus_path)
    bound = 1 + pq.read_table(corpus_path, columns=["id"])["id"].to_numpy().max()
    pubmed = load_pubmed(pubmed_dir, revision)
    latest, chunks, bm25, dense = load_legacy_flags(bound)
    pages = scan_raw_filenames(bound)
    output.mkdir(parents=True)
    writers = {}
    count = Counter()
    domains = defaultdict(Counter)
    seen_pubmed = set()
    total = 0
    try:
        for batch in official.iter_batches(batch_size=50_000, columns=["id", "url"]):
            rows = []
            grouped = defaultdict(list)
            cols = batch.to_pydict()
            for ident, url in zip(cols["id"], cols["url"], strict=True):
                ident = int(ident)
                host = (urlsplit(url).hostname or "").lower().removeprefix("www.")
                is_pubmed = host == "pubmed.ncbi.nlm.nih.gov"
                pub = pubmed.get(ident)
                if is_pubmed:
                    if pub is None or pub[0] != url:
                        raise ValueError(f"PubMed ID/URL mismatch: {ident}")
                    seen_pubmed.add(ident)
                    state = {"usable_abstract": "pubmed_usable",
                             "metadata_only": "pubmed_metadata_only",
                             "unavailable": "pubmed_unavailable"}[pub[1]]
                    raw_path = None
                else:
                    if pub is not None:
                        raise ValueError(f"PubMed ID mapped to another host: {ident}")
                    raw_path = (str(ROOT / "data/vibiomir/crawl/pages"
                                    / f"{ident % 1000:03d}" / f"{ident}.gz")
                                if pages[ident] else None)
                    state = ("legacy_chunk_candidate" if chunks[ident] else
                             "needs_extract" if pages[ident] else "needs_fetch")
                row = {"id": ident, "official_url": url, "hostname": host,
                       "source_kind": "pubmed" if is_pubmed else "legacy_web",
                       "content_state": state, "raw_path": raw_path,
                       "raw_verified": False,
                       "legacy_chunks_present": bool(chunks[ident]),
                       "legacy_bm25_present": bool(bm25[ident]),
                       "legacy_dense_present": bool(dense[ident]),
                       "legacy_manifest_status": MANIFEST_LABELS[int(latest[ident])],
                       "pubmed_status": pub[1] if pub else None,
                       "pubmed_raw_path": pub[2] if pub else None,
                       "pubmed_text_sha256": pub[3] if pub else None}
                rows.append(row)
                if state in {"needs_extract", "needs_fetch", "legacy_chunk_candidate"}:
                    grouped[state].append(row)
                count[state] += 1
                domains[host][state] += 1
                total += 1
            grouped["document_state"].extend(rows)
            for name, records in grouped.items():
                if not records:
                    continue
                writer = writers.get(name)
                if writer is None:
                    writer = pq.ParquetWriter(output / f"{name}.parquet", SCHEMA, compression="zstd")
                    writers[name] = writer
                writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
    finally:
        for writer in writers.values():
            writer.close()
    if total != official.metadata.num_rows or total != 1_220_598 or seen_pubmed != set(pubmed):
        raise ValueError("Ledger coverage differs from official corpus or PubMed export")
    report = {"status": "complete", "checked_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
              "corpus_revision": revision, "corpus_sha256": file_sha(corpus_path),
              "step1_manifest": str(step1_path), "pubmed_release": str(pubmed_dir / "release_manifest.json"),
              "rows": total, "states": dict(count),
              "legacy_manifest_order": "filename and row order; historical timestamps unknown",
              "raw_file_presence_is_not_content_validation": True,
              "legacy_chunk_presence_is_not_quality_validation": True,
              "files": {p.name: {"rows": pq.ParquetFile(p).metadata.num_rows, "sha256": file_sha(p)}
                        for p in output.glob("*.parquet")},
              "domains": {host: dict(states) for host, states in domains.items()}}
    (output / "ledger_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step1", type=Path, default=DEFAULT_STEP1)
    parser.add_argument("--pubmed", type=Path, default=DEFAULT_PUBMED)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.step1.resolve(), args.pubmed.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ("status", "rows", "states")}, indent=2))


if __name__ == "__main__":
    main()

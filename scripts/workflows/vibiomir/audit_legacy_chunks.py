"""Audit structural quality of legacy chunks within the pinned official corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LEDGER = ROOT / "outputs/vibiomir_step2_20261013/document_state.parquet"
DEFAULT_OUTPUT = ROOT / "outputs/vibiomir_step2_20261013/legacy_chunk_quality.parquet"
SUSPECT = re.compile(r"<script\b|<style\b|@font-face|document\.cookie|cf-mitigated|captcha", re.I)
SCHEMA = pa.schema([
    ("id", pa.int64()), ("chunk_count", pa.int32()),
    ("chunk_index_min", pa.int32()), ("chunk_index_max", pa.int32()),
    ("chunk_indices_contiguous", pa.bool_()),
    ("total_chunk_chars", pa.int64()), ("short_chunks", pa.int32()),
    ("blank_chunks", pa.int32()), ("suspect_markup_chunks", pa.int32()),
])


def run(ledger: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(output)
    official = pq.read_table(ledger, columns=["id", "source_kind"])
    ids = official["id"].to_numpy()
    bound = int(ids.max()) + 1
    in_scope = np.zeros(bound, dtype=np.bool_)
    in_scope[ids] = True
    counts = np.zeros(bound, dtype=np.int32)
    mins = np.full(bound, np.iinfo(np.int32).max, dtype=np.int32)
    maxs = np.full(bound, -1, dtype=np.int32)
    chars = np.zeros(bound, dtype=np.int64)
    short = np.zeros(bound, dtype=np.int32)
    blank = np.zeros(bound, dtype=np.int32)
    suspect = np.zeros(bound, dtype=np.int32)
    source = ROOT / "data/vibiomir/chunks_crawl_20261007.parquet"
    scanned = kept = 0
    pf = pq.ParquetFile(source)
    for batch in pf.iter_batches(batch_size=20_000,
                                 columns=["doc_id", "chunk_index", "text"]):
        doc = batch.column(0).to_numpy(zero_copy_only=False)
        indices = batch.column(1).to_numpy(zero_copy_only=False)
        valid = (doc >= 0) & (doc < bound)
        valid[valid] &= in_scope[doc[valid]]
        positions = np.flatnonzero(valid)
        selected_ids = doc[positions]
        selected_indices = indices[positions]
        np.add.at(counts, selected_ids, 1)
        np.minimum.at(mins, selected_ids, selected_indices)
        np.maximum.at(maxs, selected_ids, selected_indices)
        texts = batch.column(2)
        for row_index, ident in zip(positions, selected_ids, strict=True):
            value = texts[int(row_index)].as_py()
            length = len(value) if value else 0
            chars[int(ident)] += length
            if not value or not value.strip():
                blank[int(ident)] += 1
            elif length < 30:
                short[int(ident)] += 1
            if value and SUSPECT.search(value):
                suspect[int(ident)] += 1
        kept += len(positions)
        scanned += batch.num_rows
        if scanned % 1_000_000 < batch.num_rows:
            print(f"Scanned {scanned:,}/{pf.metadata.num_rows:,} chunk rows", flush=True)
    candidate_ids = np.flatnonzero(counts)
    if kept != 4_261_536 or len(candidate_ids) != 833_892:
        raise ValueError(f"Unexpected official legacy chunk scope: {kept}, {len(candidate_ids)}")
    rows = []
    writer = pq.ParquetWriter(output, SCHEMA, compression="zstd")
    try:
        for ident in candidate_ids:
            ident = int(ident)
            rows.append({"id": ident, "chunk_count": int(counts[ident]),
                         "chunk_index_min": int(mins[ident]),
                         "chunk_index_max": int(maxs[ident]),
                         "chunk_indices_contiguous": bool(mins[ident] == 0 and maxs[ident] + 1 == counts[ident]),
                         "total_chunk_chars": int(chars[ident]),
                         "short_chunks": int(short[ident]),
                         "blank_chunks": int(blank[ident]),
                         "suspect_markup_chunks": int(suspect[ident])})
            if len(rows) == 50_000:
                writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
                rows.clear()
        if rows:
            writer.write_table(pa.Table.from_pylist(rows, schema=SCHEMA))
    finally:
        writer.close()
    summary = {"checked_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
               "source": str(source), "scanned_rows": scanned,
               "official_chunk_rows": kept, "official_documents_with_chunks": len(candidate_ids),
               "documents_with_blank_chunk": int(np.count_nonzero(blank)),
               "documents_with_short_chunk": int(np.count_nonzero(short)),
               "documents_with_suspect_markup_chunk": int(np.count_nonzero(suspect)),
               "documents_with_noncontiguous_chunk_indices": int(np.count_nonzero(
                   (mins[candidate_ids] != 0) | (maxs[candidate_ids] + 1 != counts[candidate_ids]))),
               "quality_scope": "structural signals; raw HTML and clinical relevance not validated"}
    with output.open("rb") as stream:
        summary["output_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
    output.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.ledger, args.output), indent=2))


if __name__ == "__main__":
    main()

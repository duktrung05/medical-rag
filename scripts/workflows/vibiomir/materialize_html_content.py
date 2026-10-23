"""Checkpointed HTML extraction with raw hashes and exact-offset passages."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import trafilatura
from lxml import html as lxml_html

from src.extraction.html_extractor import decode_html
from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.contracts import Source
from src.medrag.evidence.verifier import verify


ROOT = Path(__file__).resolve().parents[3]
REVISION = "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"
BASE = ROOT / "outputs/vibiomir_step2_20261013"
DOC_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("status", pa.string()), ("title", pa.string()), ("text", pa.string()),
    ("encoding", pa.string()), ("encoding_source", pa.string()),
    ("extraction_method", pa.string()), ("raw_path", pa.string()),
    ("gzip_sha256", pa.string()), ("raw_sha256", pa.string()),
    ("raw_bytes", pa.int64()), ("body_bytes", pa.int64()),
    ("text_sha256", pa.string()), ("error", pa.string()),
])
PASSAGE_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("passage_id", pa.string()),
    ("passage_index", pa.int32()), ("text", pa.string()),
    ("source_start", pa.int64()), ("source_end", pa.int64()),
    ("source_sha256", pa.string()), ("language_hint", pa.string()),
])
CHALLENGE = re.compile(
    r"<title>\s*(?:just a moment|access denied|captcha|verify|security check)|"
    r"cf-mitigated|cloudflare.*challenge|/captcha/|/verify\.html", re.I)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def title_from_html(decoded: str) -> str:
    try:
        tree = lxml_html.fromstring(decoded)
        for query in ("//meta[@property='og:title']/@content", "//article//h1[1]//text()",
                      "//h1[1]//text()", "//title[1]//text()"):
            parts = tree.xpath(query)
            value = " ".join(str(x).strip() for x in parts if str(x).strip())
            if value:
                return re.sub(r"\s+", " ", value).strip()[:500]
    except (TypeError, ValueError, lxml_html.etree.ParserError):
        pass
    return ""


def process(row: dict) -> tuple[dict, list[dict]]:
    ident, url = row["id"], row["official_url"]
    result = {"id": ident, "official_url": url, "hostname": row["hostname"],
              "status": "extract_failed", "title": "", "text": "", "encoding": None,
              "encoding_source": None, "extraction_method": "trafilatura_2_3_1_favor_recall",
              "raw_path": row["raw_path"], "gzip_sha256": None, "raw_sha256": None,
              "raw_bytes": None, "body_bytes": None, "text_sha256": None, "error": None}
    passages = []
    try:
        compressed = Path(row["raw_path"]).read_bytes()
        result["raw_bytes"] = len(compressed)
        is_gzip = compressed.startswith(b"\x1f\x8b")
        result["gzip_sha256"] = hashlib.sha256(compressed).hexdigest() if is_gzip else None
        raw = gzip.decompress(compressed) if is_gzip else compressed
        result["raw_sha256"] = hashlib.sha256(raw).hexdigest()
        result["body_bytes"] = len(raw)
        decoded, encoding, encoding_source = decode_html(raw)
        result["encoding"] = encoding
        result["encoding_source"] = encoding_source
        result["title"] = title_from_html(decoded)
        if CHALLENGE.search(decoded[:100_000]):
            result["status"] = "quality_review_challenge"
            return result, passages
        text = trafilatura.extract(decoded, include_tables=True, favor_recall=True,
                                   include_comments=False, no_fallback=False)
        result["text"] = (text or "").strip()
        if not result["text"]:
            result["status"] = "quality_review_empty"
            return result, passages
        result["text_sha256"] = hashlib.sha256(result["text"].encode()).hexdigest()
        if len(result["text"]) < 80:
            result["status"] = "quality_review_short"
            return result, passages
        result["status"] = "usable"
        source = Source("official-" + REVISION, ident, url, result["title"], result["text"],
                        row["raw_path"], result["raw_sha256"])
        for index, passage in enumerate(build_passages(source)):
            if not verify(passage, source):
                raise ValueError("passage offset mismatch")
            passages.append({"id": ident, "official_url": url,
                             "passage_id": passage.passage_id, "passage_index": index,
                             "text": passage.text, "source_start": passage.source_start,
                             "source_end": passage.source_end,
                             "source_sha256": passage.source_sha256,
                             "language_hint": passage.language})
    except (OSError, EOFError, ValueError, UnicodeError, TypeError,
            lxml_html.etree.ParserError) as exc:
        result["status"] = "extract_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"[:500]
        passages = []
    return result, passages


def write_atomic(path: Path, records: list[dict], schema: pa.Schema) -> None:
    temporary = path.with_suffix(".partial")
    if temporary.exists() or path.exists():
        raise FileExistsError(path)
    pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary, compression="zstd")
    os.replace(temporary, path)


def run(source: str, max_parts: int | None = None, *, shard_index: int = 0,
        num_shards: int = 1) -> dict:
    if source not in {"needs_extract", "legacy_chunk_candidate"}:
        raise ValueError(source)
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid shard assignment")
    input_path = BASE / f"{source}.parquet"
    root = BASE / "content" / source
    dirs = [root / name for name in ("documents", "passages", "parts")]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
    reader = pq.ParquetFile(input_path)
    counts = Counter()
    part_count = 0
    selected_total = 0
    for batch_index, batch in enumerate(reader.iter_batches(batch_size=1000)):
        if batch_index % num_shards != shard_index:
            continue
        if max_parts is not None and part_count >= max_parts:
            break
        selected_total += batch.num_rows
        name = f"part_{batch_index:05d}"
        doc_path = dirs[0] / f"{name}.parquet"
        passage_path = dirs[1] / f"{name}.parquet"
        sidecar = dirs[2] / f"{name}.json"
        source_rows = batch.to_pylist()
        if sidecar.exists():
            saved = json.loads(sidecar.read_text())
            if (saved["input_first_id"] != source_rows[0]["id"]
                    or saved["input_last_id"] != source_rows[-1]["id"]
                    or not doc_path.exists() or not passage_path.exists()
                    or sha(doc_path) != saved["documents_sha256"]
                    or sha(passage_path) != saved["passages_sha256"]):
                raise ValueError(f"Existing checkpoint mismatch: {name}")
            counts.update(saved["status_counts"])
            counts["passages"] += saved["passages"]
            part_count += 1
            continue
        documents, passages = [], []
        for row in source_rows:
            document, chunks = process(row)
            documents.append(document)
            passages.extend(chunks)
            counts[document["status"]] += 1
        write_atomic(doc_path, documents, DOC_SCHEMA)
        write_atomic(passage_path, passages, PASSAGE_SCHEMA)
        part_status = dict(Counter(row["status"] for row in documents))
        sidecar.write_text(json.dumps({"input_first_id": source_rows[0]["id"],
                                       "input_last_id": source_rows[-1]["id"],
                                       "documents": len(documents), "passages": len(passages),
                                       "status_counts": part_status,
                                       "documents_sha256": sha(doc_path),
                                       "passages_sha256": sha(passage_path)}, indent=2) + "\n")
        counts["passages"] += len(passages)
        part_count += 1
        print(f"{source} {name}: {len(documents)} docs, {part_status}, {len(passages)} passages", flush=True)
    complete = max_parts is None and sum(v for k, v in counts.items() if k != "passages") == selected_total
    report = {"source": source, "input_rows": reader.metadata.num_rows,
              "processed_rows": sum(v for k, v in counts.items() if k != "passages"),
              "parts": part_count, "shard_index": shard_index, "num_shards": num_shards,
              "selected_input_rows": selected_total,
              "status_counts": {k: v for k, v in counts.items() if k != "passages"},
              "passages": counts["passages"], "complete": complete,
              "revision": REVISION, "extractor": "trafilatura_2_3_1_favor_recall",
              "passage_policy": "v4 exact offsets, 256 tokens, overlap 32"}
    progress_name = "progress.json" if num_shards == 1 else f"progress_shard_{shard_index:02d}.json"
    (root / progress_name).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=["needs_extract", "legacy_chunk_candidate"])
    parser.add_argument("--max-parts", type=int)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(run(args.source, args.max_parts, shard_index=args.shard_index,
                         num_shards=args.num_shards), indent=2))


if __name__ == "__main__":
    main()

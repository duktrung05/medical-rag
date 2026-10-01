"""Chunk successful ViBioMIR probe extractions using a cached BGE-M3 tokenizer."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from tokenizers import Tokenizer

from src.chunking import ChunkingConfig, chunk_text, detect_language

TOKENIZER_NAME = "BAAI/bge-m3"
DEFAULT_INPUT = Path("data/vibiomir/probe/extracted_documents.parquet")
DEFAULT_OUTPUT = Path("data/vibiomir/probe/chunks.parquet")
DEFAULT_SUMMARY = Path("data/vibiomir/probe/chunking_summary.json")
DEFAULT_REVIEW = Path("docs/vibiomir_chunking_review_vi.md")
SCHEMA = pa.schema([
    ("chunk_id", pa.string()), ("doc_id", pa.int64()),
    ("chunk_order", pa.int32()), ("chunk_text", pa.string()),
    ("char_start", pa.int64()), ("char_end", pa.int64()),
    ("token_count", pa.int32()), ("title", pa.string()),
    ("url", pa.string()), ("domain", pa.string()),
    ("language", pa.string()), ("tokenizer_name", pa.string()),
    ("source_content_sha256", pa.string()),
])


class CachedBgeTokenizer:
    """Load tokenizer.json directly from the local Hugging Face cache; no hub call."""

    name = TOKENIZER_NAME

    def __init__(self, cache_dir: Path | None = None):
        base = cache_dir or Path.home() / ".cache/huggingface/hub/models--BAAI--bge-m3"
        refs = base / "refs" / "main"
        if not refs.is_file():
            raise FileNotFoundError(f"BGE-M3 tokenizer revision is not cached: {refs}")
        self.revision = refs.read_text(encoding="utf-8").strip()
        tokenizer_json = base / "snapshots" / self.revision / "tokenizer.json"
        if not tokenizer_json.is_file():
            raise FileNotFoundError(f"BGE-M3 tokenizer.json is not cached: {tokenizer_json}")
        self.tokenizer = Tokenizer.from_file(str(tokenizer_json))

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text).ids)


def _atomic_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                                     suffix=".tmp", delete=False) as handle:
        return Path(handle.name)


def _atomic_text(destination: Path, content: str) -> None:
    temp = _atomic_path(destination)
    try:
        temp.write_text(content, encoding="utf-8")
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def _stats(values: list[int]) -> dict[str, float | int | None]:
    if not values:
        return {"min": None, "median": None, "max": None}
    return {"min": min(values), "median": statistics.median(values), "max": max(values)}


def _validate(rows: list[dict], documents: dict[int, dict], tokenizer,
              config: ChunkingConfig) -> list[str]:
    errors = []
    by_doc = defaultdict(list)
    all_ids = set()
    for row in rows:
        doc_id = row["doc_id"]
        by_doc[doc_id].append(row)
        if row["chunk_id"] in all_ids:
            errors.append(f"Duplicate chunk_id {row['chunk_id']}")
        all_ids.add(row["chunk_id"])
        if doc_id not in documents:
            errors.append(f"Unknown doc_id {doc_id}")
            continue
        text = documents[doc_id]["text"]
        if not row["chunk_text"] or row["chunk_text"] != text[row["char_start"]:row["char_end"]]:
            errors.append(f"Invalid source slice {row['chunk_id']}")
        if tokenizer.count(row["chunk_text"]) != row["token_count"]:
            errors.append(f"Token count mismatch {row['chunk_id']}")
        if row["token_count"] > config.max_tokens:
            errors.append(f"Oversized chunk {row['chunk_id']}")
    for doc_id, doc in documents.items():
        chunks = by_doc[doc_id]
        if [r["chunk_order"] for r in chunks] != list(range(len(chunks))):
            errors.append(f"Noncontinuous chunk order for {doc_id}")
        if len({r["chunk_text"] for r in chunks}) != len(chunks):
            errors.append(f"Duplicate text in {doc_id}")
        covered = bytearray(len(doc["text"]))
        for row in chunks:
            covered[row["char_start"]:row["char_end"]] = b"\x01" * (
                row["char_end"] - row["char_start"]
            )
        if any(not covered[i] for i, c in enumerate(doc["text"]) if not c.isspace()):
            errors.append(f"Uncovered meaningful text in {doc_id}")
    return errors


def chunk_probe(input_file: Path = DEFAULT_INPUT, output_file: Path = DEFAULT_OUTPUT,
                summary_file: Path = DEFAULT_SUMMARY, review_file: Path = DEFAULT_REVIEW,
                tokenizer=None, config: ChunkingConfig | None = None) -> list[dict]:
    tokenizer = tokenizer or CachedBgeTokenizer()
    config = config or ChunkingConfig()
    source = pq.read_table(input_file).to_pylist()
    documents = {int(r["id"]): r for r in source
                 if r["extraction_status"] == "success" and r.get("text", "").strip()}
    if len(documents) != sum(r["extraction_status"] == "success" and bool((r.get("text") or "").strip())
                             for r in source):
        raise ValueError("Duplicate document ID in extraction input")
    rows = []
    for doc_id, document in sorted(documents.items()):
        language = detect_language(document["text"])
        for order, span in enumerate(chunk_text(document["text"], tokenizer, config)):
            rows.append({
                "chunk_id": f"{doc_id}:{order:04d}", "doc_id": doc_id,
                "chunk_order": order,
                "chunk_text": document["text"][span.char_start:span.char_end],
                "char_start": span.char_start, "char_end": span.char_end,
                "token_count": span.token_count, "title": document.get("title"),
                "url": document.get("url"), "domain": document.get("domain"),
                "language": language, "tokenizer_name": TOKENIZER_NAME,
                "source_content_sha256": document.get("content_sha256"),
            })
    errors = _validate(rows, documents, tokenizer, config)
    if errors:
        raise ValueError("Chunk validation failed: " + "; ".join(errors[:10]))
    by_doc = defaultdict(list)
    for row in rows:
        by_doc[row["doc_id"]].append(row)
    summary = {
        "tokenizer_name": TOKENIZER_NAME,
        "tokenizer_revision": getattr(tokenizer, "revision", "test-fixture"),
        "config": {
            "target_tokens": config.target_tokens, "max_tokens": config.max_tokens,
            "overlap_tokens": config.overlap_tokens,
            "min_chunk_tokens": config.min_chunk_tokens,
        },
        "document_count": len(documents), "total_chunks": len(rows),
        "chunks_per_document": _stats([len(by_doc[i]) for i in documents]),
        "tokens_per_chunk": _stats([r["token_count"] for r in rows]),
        "chunks_below_min_tokens": sum(r["token_count"] < config.min_chunk_tokens for r in rows),
        "chunks_at_max_tokens": sum(r["token_count"] == config.max_tokens for r in rows),
        "validation_error_count": len(errors),
        "language_distribution": dict(sorted(Counter(
            detect_language(d["text"]) for d in documents.values()
        ).items())),
    }
    review = ["# Rà soát chunk ViBioMIR probe", "",
              (f"Tokenizer: `{TOKENIZER_NAME}` revision `{summary['tokenizer_revision']}`. "
               f"Tổng: {len(rows)} chunk từ {len(documents)} tài liệu."), ""]
    for doc_id, doc in sorted(documents.items()):
        chunks = by_doc[doc_id]
        review += [f"## Document {doc_id}: {doc.get('title') or ''}", "",
                   f"- Language: {detect_language(doc['text'])}",
                   f"- Độ dài: {len(doc['text'])} ký tự; số chunk: {len(chunks)}", ""]
        for row in chunks:
            review += [f"### Chunk {row['chunk_order']}", "",
                       (f"- Tokens: {row['token_count']}; chars: "
                        f"[{row['char_start']}, {row['char_end']})"), "",
                       row["chunk_text"][:250], ""]
    temp = _atomic_path(output_file)
    try:
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), temp, compression="zstd")
        os.replace(temp, output_file)
    finally:
        temp.unlink(missing_ok=True)
    _atomic_text(summary_file, json.dumps(summary, ensure_ascii=False, indent=2,
                                          sort_keys=True) + "\n")
    _atomic_text(review_file, "\n".join(review))
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args(argv)
    rows = chunk_probe(args.input_file, args.output_file, args.summary_file, args.review_file)
    print(f"Chunked {len({r['doc_id'] for r in rows})} documents into {len(rows)} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

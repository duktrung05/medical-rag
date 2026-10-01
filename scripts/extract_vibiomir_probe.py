"""Extract text from successful ViBioMIR probe downloads, without network access."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.extraction.html_extractor import DecodeError, extract_html

DEFAULT_PROBE = Path("data/vibiomir/probe/probe_results.parquet")
DEFAULT_OUTPUT = Path("data/vibiomir/probe/extracted_documents.parquet")
DEFAULT_SUMMARY = Path("data/vibiomir/probe/extraction_summary.json")
DEFAULT_REVIEW = Path("docs/vibiomir_extraction_review_vi.md")

SCHEMA = pa.schema([
    ("id", pa.int64()), ("url", pa.string()), ("final_url", pa.string()),
    ("domain", pa.string()), ("title", pa.string()), ("text", pa.string()),
    ("encoding", pa.string()), ("encoding_source", pa.string()),
    ("extraction_method", pa.string()), ("paragraph_count", pa.int32()),
    ("text_char_count", pa.int64()), ("raw_bytes", pa.int64()),
    ("content_sha256", pa.string()), ("extraction_status", pa.string()),
    ("extraction_error", pa.string()),
])


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


def _safe_raw_path(probe_file: Path, raw_file: str) -> Path:
    base = probe_file.parent.resolve()
    path = (base / raw_file).resolve()
    if not path.is_relative_to(base):
        raise ValueError(f"Raw path outside probe directory: {raw_file}")
    return path


def extract_probe(
    probe_file: Path = DEFAULT_PROBE,
    output_file: Path = DEFAULT_OUTPUT,
    summary_file: Path = DEFAULT_SUMMARY,
    review_file: Path = DEFAULT_REVIEW,
) -> list[dict]:
    probe_rows = pq.read_table(probe_file).to_pylist()
    selected = sorted((r for r in probe_rows if r["status"] == "success"),
                      key=lambda r: int(r["id"]))
    ids = [int(r["id"]) for r in selected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate success IDs in probe results")
    results = []
    for probe in selected:
        record = {
            "id": int(probe["id"]), "url": probe["url"],
            "final_url": probe.get("final_url"), "domain": probe.get("domain"),
            "title": "", "text": "", "encoding": None, "encoding_source": None,
            "extraction_method": None, "paragraph_count": 0, "text_char_count": 0,
            "raw_bytes": None, "content_sha256": None,
            "extraction_status": "missing_raw_file", "extraction_error": None,
        }
        raw_name = probe.get("raw_file")
        try:
            if not raw_name:
                raise FileNotFoundError("Probe success has no raw_file")
            raw_path = _safe_raw_path(probe_file, raw_name)
            raw = raw_path.read_bytes()
            record["raw_bytes"] = len(raw)
            record["content_sha256"] = hashlib.sha256(raw).hexdigest()
            if probe.get("content_sha256") and record["content_sha256"] != probe["content_sha256"]:
                raise ValueError("Raw SHA-256 differs from probe metadata")
            if probe.get("bytes_downloaded") is not None and len(raw) != probe["bytes_downloaded"]:
                raise ValueError("Raw byte count differs from probe metadata")
            extracted = extract_html(raw, probe.get("content_type"))
            record.update(title=extracted.title, text=extracted.text,
                          encoding=extracted.encoding,
                          encoding_source=extracted.encoding_source,
                          extraction_method=extracted.extraction_method,
                          paragraph_count=extracted.paragraph_count,
                          text_char_count=len(extracted.text))
            record["extraction_status"] = "success" if extracted.text else "empty_content"
            if re.search(r"<script\b|<style\b|\b(?:function\s*\(|var\s+\w+\s*=|@font-face)\b",
                         extracted.text, re.IGNORECASE):
                raise ValueError("Obvious script/CSS found in extracted text")
        except FileNotFoundError as exc:
            record["extraction_status"] = "missing_raw_file"
            record["extraction_error"] = str(exc)
        except DecodeError as exc:
            record["extraction_status"] = "decode_error"
            record["extraction_error"] = str(exc)
        except (OSError, ValueError, UnicodeError, TypeError) as exc:
            record["extraction_status"] = "extraction_error"
            record["extraction_error"] = str(exc)
        results.append(record)
    if [r["id"] for r in results] != ids:
        raise AssertionError("Output IDs differ from input success IDs")
    if any(r["extraction_status"] == "success" and not r["text"] for r in results):
        raise AssertionError("Successful extraction has empty text")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    temp = _atomic_path(output_file)
    try:
        pq.write_table(pa.Table.from_pylist(results, schema=SCHEMA), temp, compression="zstd")
        os.replace(temp, output_file)
    finally:
        temp.unlink(missing_ok=True)
    counts = dict(sorted(Counter(r["extraction_status"] for r in results).items()))
    encodings = dict(sorted(Counter(r["encoding"] for r in results if r["encoding"]).items()))
    summary = {
        "probe_file": str(probe_file), "success_probe_rows": len(selected),
        "extracted_rows": len(results), "status_counts": counts,
        "encoding_counts": encodings,
    }
    _atomic_text(summary_file, json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    lines = ["# Rà soát trích xuất HTML ViBioMIR", "",
             f"Số raw HTML được xử lý: {len(results)}. Nội dung dưới đây là trích đoạn đầu, cần kiểm tra thủ công.", ""]
    for row in results:
        lines.extend([
            f"## ID {row['id']} — {row['domain']}", "",
            f"- URL: {row['url']}", f"- Tiêu đề: {row['title']}",
            f"- Trạng thái: {row['extraction_status']}",
            f"- Encoding: {row['encoding']} ({row['encoding_source']})",
            f"- Phương pháp: {row['extraction_method']}",
            f"- Đoạn: {row['paragraph_count']}; ký tự: {row['text_char_count']}",
            f"- Lỗi: {row['extraction_error'] or 'không có'}", "",
            "**800 ký tự đầu:**", "", row["text"][:800] or "_(rỗng)_", "",
        ])
    _atomic_text(review_file, "\n".join(lines))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-file", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary-file", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--review-file", type=Path, default=DEFAULT_REVIEW)
    args = parser.parse_args(argv)
    rows = extract_probe(args.probe_file, args.output_file, args.summary_file, args.review_file)
    print(json.dumps(dict(Counter(r["extraction_status"] for r in rows)), sort_keys=True))
    return 0 if all(r["extraction_status"] == "success" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())

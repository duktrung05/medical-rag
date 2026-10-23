"""Recover Báo Tây Ninh article text from its explicit article container."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import trafilatura
from bs4 import BeautifulSoup

from scripts.workflows.vibiomir.materialize_html_content import (
    BASE, DOC_SCHEMA, PASSAGE_SCHEMA, REVISION, title_from_html,
)
from src.extraction.html_extractor import decode_html
from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.contracts import Source
from src.medrag.evidence.verifier import verify


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run() -> dict:
    rows = [row for row in pq.read_table(BASE / "legacy_chunk_candidate.parquet",
                                         columns=["id", "official_url", "hostname", "raw_path"]).to_pylist()
            if row["hostname"] == "baotayninh.vn"]
    if len(rows) != 1198:
        raise ValueError(f"Expected 1198 official Báo Tây Ninh rows, found {len(rows)}")
    # The fallback is only valid for rows rejected by the general extractor.
    prior = {}
    for path in (BASE / "content/legacy_chunk_candidate/documents").glob("part_*.parquet"):
        table = pq.read_table(path, columns=["id", "hostname", "status"])
        for row in table.to_pylist():
            if row["hostname"] == "baotayninh.vn":
                prior[row["id"]] = row["status"]
    if len(prior) != len(rows) or any(value != "quality_review_empty" for value in prior.values()):
        raise ValueError("General extractor status changed; re-audit fallback eligibility")
    documents, passages = [], []
    for row in rows:
        path = Path(row["raw_path"])
        compressed = path.read_bytes()
        raw = gzip.decompress(compressed)
        decoded, encoding, encoding_source = decode_html(raw)
        body = BeautifulSoup(decoded, "lxml").select_one("#content-detail")
        content = trafilatura.extract(str(body), include_tables=True,
                                      favor_recall=True) if body else None
        content = (content or "").strip()
        status = "usable" if len(content) >= 80 else "quality_review_empty"
        document = {"id": row["id"], "official_url": row["official_url"],
                    "hostname": row["hostname"], "status": status,
                    "title": title_from_html(decoded), "text": content,
                    "encoding": encoding, "encoding_source": encoding_source,
                    "extraction_method": "baotayninh_css_content_detail_v1",
                    "raw_path": str(path), "gzip_sha256": hashlib.sha256(compressed).hexdigest(),
                    "raw_sha256": hashlib.sha256(raw).hexdigest(),
                    "raw_bytes": len(compressed), "body_bytes": len(raw),
                    "text_sha256": hashlib.sha256(content.encode()).hexdigest() if content else None,
                    "error": None}
        documents.append(document)
        if status != "usable":
            continue
        source = Source("official-" + REVISION, row["id"], row["official_url"],
                        document["title"], content, str(path), document["raw_sha256"])
        for index, passage in enumerate(build_passages(source)):
            if not verify(passage, source):
                raise ValueError(f"Passage offset mismatch for ID {row['id']}")
            passages.append({"id": row["id"], "official_url": row["official_url"],
                             "passage_id": passage.passage_id, "passage_index": index,
                             "text": passage.text, "source_start": passage.source_start,
                             "source_end": passage.source_end,
                             "source_sha256": passage.source_sha256,
                             "language_hint": passage.language})
    output = BASE / "content/baotayninh_css_fallback"
    output.mkdir(parents=True, exist_ok=True)
    destinations = [output / "documents.parquet", output / "passages.parquet"]
    if any(path.exists() for path in destinations):
        raise FileExistsError("Fallback output already exists")
    for path, records, schema in zip(destinations, (documents, passages),
                                     (DOC_SCHEMA, PASSAGE_SCHEMA), strict=True):
        temporary = path.with_suffix(".partial")
        pq.write_table(pa.Table.from_pylist(records, schema=schema), temporary,
                       compression="zstd")
        os.replace(temporary, path)
    report = {"status": "complete", "revision": REVISION,
              "source": "legacy_chunk_candidate", "prior_status": "quality_review_empty",
              "domain": "baotayninh.vn", "selector": "#content-detail",
              "documents": len(documents), "passages": len(passages),
              "status_counts": dict(Counter(row["status"] for row in documents)),
              "documents_sha256": sha(destinations[0]),
              "passages_sha256": sha(destinations[1])}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))

"""Script tests on local tiny Parquet data, without a model or network."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workflows.vibiomir.chunk_vibiomir_probe import chunk_probe
from src.chunking import ChunkingConfig


class FakeTokenizer:
    revision = "local-fixture"

    def count(self, text: str) -> int:
        return len(text)


def test_chunk_script_determinism_and_skipped_rows(tmp_path):
    docs = [
        {"id": 583, "text": "Sức khỏe cộng đồng.\n\nĐiều trị phù hợp cho mọi người.",
         "title": "Một bài", "url": "https://example.org/583", "domain": "example.org",
         "content_sha256": "abc", "extraction_status": "success"},
        {"id": 584, "text": "", "extraction_status": "success"},
        {"id": 585, "text": "Nội dung lỗi", "extraction_status": "empty_content"},
    ]
    source = tmp_path / "extracted.parquet"
    pq.write_table(pa.Table.from_pylist(docs), source)
    output, summary, review = (tmp_path / name for name in
                               ("chunks.parquet", "summary.json", "review.md"))
    cfg = ChunkingConfig(target_tokens=30, max_tokens=70, overlap_tokens=10,
                         min_chunk_tokens=10)
    rows = chunk_probe(source, output, summary, review, FakeTokenizer(), cfg)
    assert {r["doc_id"] for r in rows} == {583}
    assert [r["chunk_order"] for r in rows] == list(range(len(rows)))
    assert rows[0]["chunk_id"] == "583:0000"
    assert all(r["source_content_sha256"] == "abc" for r in rows)
    assert all(r["chunk_text"] == docs[0]["text"][r["char_start"]:r["char_end"]]
               for r in rows)
    assert json.loads(summary.read_text())["validation_error_count"] == 0
    assert "Chunk 0" in review.read_text(encoding="utf-8")
    before = [p.read_bytes() for p in (output, summary, review)]
    chunk_probe(source, output, summary, review, FakeTokenizer(), cfg)
    assert before == [p.read_bytes() for p in (output, summary, review)]
    assert pq.read_table(output).num_rows == len(rows)

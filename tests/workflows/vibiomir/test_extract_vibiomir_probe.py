"""Offline script tests with tiny Parquet fixtures."""

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workflows.vibiomir.extract_vibiomir_probe import extract_probe


def _probe_row(official_id, raw_file, raw=None, status="success"):
    return {
        "id": official_id, "url": f"https://example.org/{official_id}",
        "final_url": f"https://example.org/{official_id}", "domain": "example.org",
        "status": status, "raw_file": raw_file, "content_type": "text/html; charset=utf-8",
        "content_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
        "bytes_downloaded": len(raw) if raw is not None else None,
    }


def test_success_missing_raw_and_deterministic_output(tmp_path):
    probe_dir = tmp_path / "probe"
    raw_dir = probe_dir / "raw"
    raw_dir.mkdir(parents=True)
    html = b"<article><p>First medical paragraph with enough detail.</p>"
    html += b"<p>Second medical paragraph with enough detail.</p></article>"
    (raw_dir / "1.bin").write_bytes(html)
    rows = [_probe_row(1, "raw/1.bin", html), _probe_row(2, "raw/2.bin"),
            _probe_row(3, None, status="robots_unavailable")]
    probe = probe_dir / "probe_results.parquet"
    pq.write_table(pa.Table.from_pylist(rows), probe)
    output = probe_dir / "extracted.parquet"
    summary = probe_dir / "summary.json"
    review = tmp_path / "review.md"
    results = extract_probe(probe, output, summary, review)
    assert [r["id"] for r in results] == [1, 2]
    assert [r["extraction_status"] for r in results] == ["success", "missing_raw_file"]
    assert results[0]["raw_bytes"] == len(html)
    assert results[0]["content_sha256"] == hashlib.sha256(html).hexdigest()
    assert "First medical" in review.read_text()
    assert json.loads(summary.read_text())["status_counts"]["missing_raw_file"] == 1
    before = [path.read_bytes() for path in (output, summary, review)]
    extract_probe(probe, output, summary, review)
    assert before == [path.read_bytes() for path in (output, summary, review)]
    assert (raw_dir / "1.bin").read_bytes() == html


def test_empty_content_and_hash_mismatch(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    (raw_dir / "1.bin").write_bytes(b"<nav>Only navigation</nav>")
    (raw_dir / "2.bin").write_bytes(b"<article><p>Medical article text.</p></article>")
    rows = [_probe_row(1, "raw/1.bin", (raw_dir / "1.bin").read_bytes()),
            _probe_row(2, "raw/2.bin", b"different")]
    probe = tmp_path / "probe.parquet"
    pq.write_table(pa.Table.from_pylist(rows), probe)
    result = extract_probe(probe, tmp_path / "out.parquet", tmp_path / "summary.json",
                           tmp_path / "review.md")
    assert [r["extraction_status"] for r in result] == ["empty_content", "extraction_error"]
    assert "SHA-256" in result[1]["extraction_error"]

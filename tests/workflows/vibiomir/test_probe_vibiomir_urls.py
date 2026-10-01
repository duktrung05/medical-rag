"""Probe orchestration tests use tiny Parquet samples and mock HTTP transport."""

import json
import socket
from collections import Counter

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.workflows.vibiomir.prepare_vibiomir import sha256_file
from scripts.workflows.vibiomir.probe_vibiomir_urls import RESULT_SCHEMA, probe, select_probe_urls
from scripts.workflows.vibiomir.sample_vibiomir_urls import SAMPLE_SCHEMA, selection_hash


@pytest.fixture
def sample(tmp_path):
    rows = []
    for index in range(30):
        domain = f"d{index:02d}.test"
        for variant in ("a", "b"):
            official_id = 10000 + index * 10 + (variant == "b")
            url = f"https://{domain}/{variant}"
            rows.append({
                "id": official_id, "url": url, "domain": domain,
                "domain_url_count": 3000 - index * 10, "domain_quota": 2,
                "sample_rank_in_domain": int(variant == "b"),
                "selection_hash": selection_hash(official_id, url),
                "sampling_reason": "domain_base",
            })
    path = tmp_path / "sample_urls.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA), path)
    (tmp_path / "sample_summary.json").write_text(json.dumps({
        "output_sha256": sha256_file(path), "sample_size_actual": len(rows)}))
    return path, tmp_path / "probe", rows


def run(sample, handler, *, limit=20, **kwargs):
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        return probe(sample[0], sample[1], limit=limit, client=client,
                     delay_seconds=0, sleep=lambda _: None,
                     progress=lambda _: None, **kwargs)


def test_selects_twenty_distinct_large_medium_small_deterministically(sample):
    selected = select_probe_urls(sample[2], 20)
    assert len(selected) == len({row["domain"] for row in selected}) == 20
    assert Counter(row["selected_group"] for row in selected) == {
        "large": 8, "medium": 6, "small": 6}
    assert [row["domain"] for row in selected[:8]] == [f"d{i:02d}.test" for i in range(8)]
    assert [row["domain"] for row in selected[8:14]] == [f"d{i:02d}.test" for i in range(13, 19)]
    assert [row["domain"] for row in selected[14:]] == [f"d{i:02d}.test" for i in range(24, 30)]
    assert selected == select_probe_urls(list(reversed(sample[2])), 20)
    assert {row["id"] for row in selected} != {row["id"] for row in sample[2][:20]}
    for row in selected:
        candidates = [candidate for candidate in sample[2] if candidate["domain"] == row["domain"]]
        assert row["selection_hash"] == min(item["selection_hash"] for item in candidates)


def test_publishes_selection_before_network_then_results_and_resume(sample):
    calls = []
    selected_path = sample[1] / "selected_urls.json"
    def handler(request):
        assert selected_path.exists()
        selected = json.loads(selected_path.read_text())
        assert len(selected["selected_urls"]) == 20
        calls.append(str(request.url))
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(
            200, content=b"<html>ok</html>", headers={"content-type": "text/html"})
    summary = run(sample, handler)
    assert summary["requested_urls"] == summary["processed_urls"] == summary["success_count"] == 20
    assert summary["status_counts"] == {"success": 20}
    assert summary["robots_allowed_count"] == 20
    assert summary["content_type_counts"] == {"text/html": 20}
    assert summary["total_bytes"] == 20 * len(b"<html>ok</html>")
    results = pq.read_table(sample[1] / "probe_results.parquet")
    assert results.schema == RESULT_SCHEMA and results.num_rows == 20
    assert len({row["id"] for row in results.to_pylist()}) == 20
    assert len(list((sample[1] / "raw").glob("*.bin"))) == 20
    assert len(calls) == 40
    def no_request(_):
        raise AssertionError("Resume must not fetch terminal IDs")
    repeated = run(sample, no_request)
    assert repeated == summary
    assert len(calls) == 40


def test_retry_failed_only_reruns_failure_and_keeps_unique_ids(sample):
    selected = select_probe_urls(sample[2], 3)
    failed_id = selected[0]["id"]
    failed_host = selected[0]["domain"]
    first_pages = []
    def first_handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        first_pages.append(request.url.host)
        return httpx.Response(500) if request.url.host == failed_host else httpx.Response(200, content=b"ok")
    initial = run(sample, first_handler, limit=3)
    assert initial["processed_urls"] == 3 and initial["success_count"] == 2
    assert first_pages.count(failed_host) == 2
    assert not (sample[1] / "raw" / f"{failed_id}.bin").exists()
    rerun_pages = []
    def rerun_handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        rerun_pages.append(request.url.host)
        return httpx.Response(200, content=b"recovered")
    repaired = run(sample, rerun_handler, limit=3, retry_failed=True)
    assert rerun_pages == [failed_host]
    assert repaired["success_count"] == 3
    rows = pq.read_table(sample[1] / "probe_results.parquet").to_pylist()
    assert len(rows) == len({row["id"] for row in rows}) == 3
    assert (sample[1] / "raw" / f"{failed_id}.bin").read_bytes() == b"recovered"


def test_force_reruns_all_and_removes_stale_raw_after_failure(sample):
    selected = select_probe_urls(sample[2], 3)
    failed_id = selected[0]["id"]
    def success(request):
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200, content=b"old")
    run(sample, success, limit=3)
    assert (sample[1] / "raw" / f"{failed_id}.bin").exists()
    pages = []
    def failure(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        pages.append(request.url.host)
        return httpx.Response(403) if request.url.host == selected[0]["domain"] else httpx.Response(200, content=b"new")
    summary = run(sample, failure, limit=3, force=True)
    assert len(pages) == 3
    assert summary["success_count"] == 2
    assert not (sample[1] / "raw" / f"{failed_id}.bin").exists()
    assert len(pq.read_table(sample[1] / "probe_results.parquet")) == 3


def test_partial_results_survive_interruption_and_resume(sample):
    page_count = 0
    def interrupted(request):
        nonlocal page_count
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        page_count += 1
        if page_count == 2:
            raise RuntimeError("Pilot interrupted")
        return httpx.Response(200, content=b"ok")
    with pytest.raises(RuntimeError, match="interrupted"):
        run(sample, interrupted, limit=3)
    assert pq.read_table(sample[1] / "probe_results.parquet").num_rows == 1
    def complete(request):
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200, content=b"ok")
    final = run(sample, complete, limit=3)
    assert final["processed_urls"] == final["success_count"] == 3
    assert pq.read_table(sample[1] / "probe_results.parquet").num_rows == 3


def test_sample_hash_mismatch_prevents_network(sample):
    summary_path = sample[0].with_name("sample_summary.json")
    summary_path.write_text(json.dumps({"output_sha256": "bad", "sample_size_actual": 60}))
    def blocked(_):
        raise AssertionError("Network call attempted")
    with pytest.raises(ValueError, match="SHA-256"):
        run(sample, blocked)
    assert not sample[1].exists()


def test_limit_above_domain_count_rejected_before_network(sample):
    with pytest.raises(ValueError, match="only 30 domains"):
        run(sample, lambda _: (_ for _ in ()).throw(AssertionError("network")), limit=31)


def test_no_real_socket_call(sample, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Real socket used")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    def handler(request):
        return httpx.Response(404) if request.url.path == "/robots.txt" else httpx.Response(200, content=b"ok")
    summary = run(sample, handler, limit=3)
    assert summary["success_count"] == 3

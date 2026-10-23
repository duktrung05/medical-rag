"""Safety tests for the isolated failed-URL recovery workflow."""

import hashlib
import json
from datetime import UTC, datetime, timedelta

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.workflows.vibiomir.audit_failed_urls import REVISION as AUDIT_REVISION
from scripts.workflows.vibiomir.crawl_vibiomir_batch import crawl_batch
from scripts.workflows.vibiomir.recover_failed_urls import (
    REVISION,
    RecoveryGuard,
    execute,
    plan_rows,
    pilot_gates,
    protected_unchanged,
    with_recent_attempts,
)
from src.collection.fetcher import BatchFetchState, BatchFetcher, PoliteFetcher


def _write_guard_files(tmp_path, *, usable=False, recovered=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    eligible_path = tmp_path / "eligible.parquet"
    status_path = tmp_path / "status.parquet"
    baseline_path = tmp_path / "baseline.json"
    result_path = tmp_path / "results.parquet"
    eligible = [{
        "id": 7, "official_url": "https://a.test/article", "hostname": "a.test",
        "network_allowed": True, "next_retry_at": None,
    }]
    pq.write_table(pa.Table.from_pylist(eligible), eligible_path)
    pq.write_table(pa.Table.from_pylist([{
        "id": 7, "official_url": "https://a.test/article", "usable": usable,
    }]), status_path)
    stat = status_path.stat()
    baseline_path.write_text(json.dumps({"files": [{
        "path": "status.parquet", "bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
    }]}))
    if recovered:
        pq.write_table(pa.Table.from_pylist([{"id": 7, "usable": True}]), result_path)
    return eligible_path, status_path, baseline_path, result_path


def _guard(tmp_path, *, usable=False, recovered=False):
    eligible, status, baseline, results = _write_guard_files(
        tmp_path, usable=usable, recovered=recovered)
    return RecoveryGuard(eligible, baseline, status, results_path=results,
                         status_key="status.parquet")


def _entry(url="https://a.test/article", ident=7):
    return {"id": ident, "url": url, "domain": "a.test"}


def _crawl_sample(tmp_path):
    path = tmp_path / "sample.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "id": 7, "url": "https://a.test/article", "domain": "a.test",
        "domain_url_count": 1,
    }]), path)
    return path


def test_verified_urls_never_refetched(tmp_path):
    with pytest.raises(ValueError, match="Verified-success"):
        _guard(tmp_path, usable=True)(_entry())


def test_verified_files_never_overwritten(tmp_path):
    protected = tmp_path / "verified.raw"
    protected.write_bytes(b"immutable verified bytes")
    before = hashlib.sha256(protected.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="Verified-success"):
        _guard(tmp_path, usable=True)(_entry())
    assert hashlib.sha256(protected.read_bytes()).hexdigest() == before


def test_recovery_allowlist_only(tmp_path):
    with pytest.raises(ValueError, match="outside the recovery allowlist"):
        _guard(tmp_path)(_entry(ident=999))


def test_duplicate_doc_ids_rejected():
    row = {"id": 1, "official_url": "https://a.test/1", "hostname": "a.test",
           "recovery_group": "http_error", "recovery_action": "bounded_network_retry",
           "network_allowed": True, "next_retry_at": None}
    with pytest.raises(ValueError, match="duplicate"):
        plan_rows([row, {**row, "official_url": "https://a.test/2"}], pilot=True,
                  now=datetime.now(UTC))


def test_corpus_revision_pinned():
    assert REVISION == AUDIT_REVISION == "71be0e1239a19dfd7a717e2bc13e3bec21bd0334"


def test_429_domain_cooldown():
    future = (datetime.now(UTC) + timedelta(hours=1)).isoformat()
    row = {"id": 1, "official_url": "https://a.test/1", "hostname": "a.test",
           "recovery_group": "rate_limited", "recovery_action": "bounded_network_retry",
           "network_allowed": True, "next_retry_at": future}
    plan, summary = plan_rows([row], pilot=True, now=datetime.now(UTC))
    assert plan == []
    assert summary["cooldown_by_group"] == {"rate_limited": 1}


def test_new_429_holds_entire_domain(tmp_path):
    attempted = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)
    result_path = tmp_path / "results.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "id": 1, "hostname": "a.test", "recovery_group": "rate_limited",
        "fetch_status": "rate_limited", "http_status": 429,
        "attempted_at": attempted.isoformat(), "usable": False,
    }]), result_path)
    eligible = [{"id": ident, "official_url": f"https://a.test/{ident}",
                 "hostname": "a.test", "recovery_group": "rate_limited",
                 "next_retry_at": None, "network_allowed": True} for ident in (1, 2)]
    updated = with_recent_attempts(eligible, result_path)
    assert all(row["next_retry_at"] == (attempted + timedelta(hours=24)).isoformat()
               for row in updated)
    assert plan_rows(updated, pilot=True, now=attempted + timedelta(hours=23))[0] == []


def test_access_blocked_domain_not_replanned(tmp_path):
    result_path = tmp_path / "results.parquet"
    pq.write_table(pa.Table.from_pylist([{
        "id": 1, "hostname": "a.test", "fetch_status": "access_blocked",
        "http_status": 403, "attempted_at": datetime.now(UTC).isoformat(),
        "usable": False,
    }]), result_path)
    assert with_recent_attempts([{"id": 2, "hostname": "a.test"}], result_path) == []


def test_pilot_rejects_http_403_and_run_requires_limit(tmp_path):
    rows = [{"recovery_group": "rate_limited", "usable": True,
             "fetch_status": "success", "http_status": 200,
             "error_type": None} for _ in range(9)]
    rows.append({**rows[0], "usable": False, "fetch_status": "access_blocked",
                 "http_status": 403})
    assert pilot_gates(rows)["rate_limited"]["passed"] is False
    with pytest.raises(ValueError, match="max-urls"):
        execute(tmp_path, "run", wave="test")


def test_antibot_challenge_detection():
    assert PoliteFetcher._challenge(
        b"<html><title>Just a moment</title><div>cf-chl</div></html>", "text/html")
    assert not PoliteFetcher._challenge(b"ordinary medical article", "text/plain")


def test_robots_policy_enforced(tmp_path):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /article")
        return httpx.Response(200, text="must not be fetched")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = BatchFetcher(tmp_path, state=BatchFetchState(), client=client,
                           delay_seconds=0, sleep=lambda _: None)
    try:
        result = fetcher.fetch({**_entry(), "selected_group": "test"})
    finally:
        fetcher.close()
        client.close()
    assert result["status"] == "robots_denied"
    assert calls == ["/robots.txt"]


def test_bounded_retry(tmp_path):
    page_calls = []

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        page_calls.append(request.url.path)
        return httpx.Response(503)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    fetcher = BatchFetcher(tmp_path, state=BatchFetchState(), client=client,
                           delay_seconds=0, sleep=lambda _: None)
    try:
        result = fetcher.fetch({**_entry(), "selected_group": "test"})
    finally:
        fetcher.close()
        client.close()
    assert result["status"] == "http_error"
    assert page_calls == ["/article", "/article"]


def test_resume_without_duplicate_fetch(tmp_path):
    sample = _crawl_sample(tmp_path)
    page_calls = []

    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        page_calls.append(request.url.path)
        return httpx.Response(200, content=b"<html>medical article</html>",
                              headers={"content-type": "text/html"})

    kwargs = dict(limit=1, concurrency=1, delay_seconds=0,
                  client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)),
                  sleep=lambda _: None, progress=lambda _: None)
    crawl_batch(sample, tmp_path / "crawl", **kwargs)
    second = crawl_batch(sample, tmp_path / "crawl", **kwargs)
    assert page_calls == ["/article"]
    assert second["skipped_due_to_resume"] == 1


def test_recovered_source_mapping(tmp_path):
    with pytest.raises(ValueError, match="source mapping differs"):
        _guard(tmp_path)(_entry(url="https://a.test/wrong"))
    with pytest.raises(ValueError, match="Recovered-success"):
        _guard(tmp_path / "recovered", recovered=True)(_entry())


def test_original_data_unchanged(tmp_path, monkeypatch):
    import scripts.workflows.vibiomir.recover_failed_urls as recovery

    protected = tmp_path / "protected.bin"
    protected.write_bytes(b"original")
    stat = protected.stat()
    root = tmp_path / "recovery"
    (root / "audit").mkdir(parents=True)
    (root / "audit/PROTECTED_DATA_BASELINE.json").write_text(json.dumps({
        "files": [{"path": "protected.bin", "bytes": stat.st_size,
                   "mtime_ns": stat.st_mtime_ns}],
        "canonical_sha256": {"protected.bin": hashlib.sha256(b"original").hexdigest()},
    }))
    monkeypatch.setattr(recovery, "ROOT", tmp_path)
    assert protected_unchanged(root)["unchanged"] is True

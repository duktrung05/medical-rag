"""HTTP probe tests use httpx.MockTransport only; never contact websites."""

import hashlib
import socket

import httpx
import pytest

from src.collection.fetcher import PoliteFetcher


def entry(url="https://example.test/page", id_=42):
    return {"id": id_, "url": url, "domain": "example.test", "selected_group": "large"}


def fetch(tmp_path, handler, *, url="https://example.test/page", max_bytes=1000,
          max_redirects=5):
    with (httpx.Client(transport=httpx.MockTransport(handler)) as client,
          PoliteFetcher(tmp_path, client=client, max_bytes=max_bytes,
                        max_redirects=max_redirects, delay_seconds=0,
                        sleep=lambda _: None) as crawler):
        return crawler.fetch(entry(url))


def test_success_html_sha_raw_and_no_forwarded_cookies(tmp_path):
    calls = []
    body = b"<html>medical content</html>"
    def handler(request):
        calls.append((str(request.url), dict(request.headers)))
        assert "cookie" not in request.headers
        assert "authorization" not in request.headers
        assert request.headers["user-agent"] == "medical-rag-research/0.1"
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /", headers={"set-cookie": "sid=secret"})
        return httpx.Response(200, content=body, headers={"content-type": "text/html"})
    with (httpx.Client(transport=httpx.MockTransport(handler)) as client,
          PoliteFetcher(tmp_path, client=client, delay_seconds=0, sleep=lambda _: None) as crawler):
        first = crawler.fetch(entry())
        second = crawler.fetch(entry("https://example.test/other", 43))
    assert first["status"] == second["status"] == "success"
    assert first["content_sha256"] == hashlib.sha256(body).hexdigest()
    assert first["bytes_downloaded"] == len(body)
    assert first["raw_file"] == "raw/42.bin"
    assert (tmp_path / "raw/42.bin").read_bytes() == body
    assert sum(url.endswith("/robots.txt") for url, _ in calls) == 1
    assert first["attempt_count"] == 1 and first["robots_status"] == 200


def test_cross_origin_redirect_checks_target_robots_before_page(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "example.test":
            return httpx.Response(302, headers={"location": "https://other.test/final"})
        return httpx.Response(200, content=b"done")
    result = fetch(tmp_path, handler)
    assert result["status"] == "success"
    assert result["redirect_count"] == 1
    assert result["attempt_count"] == 2
    assert result["final_url"] == "https://other.test/final"
    assert calls == ["https://example.test/robots.txt", "https://example.test/page",
                     "https://other.test/robots.txt", "https://other.test/final"]


def test_redirect_target_robots_denial_prevents_page_request(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "example.test" and request.url.path == "/robots.txt":
            return httpx.Response(404)
        if request.url.host == "example.test":
            return httpx.Response(302, headers={"location": "https://blocked.test/secret"})
        return httpx.Response(200, text="User-agent: *\nDisallow: /secret")
    result = fetch(tmp_path, handler)
    assert result["status"] == "robots_denied"
    assert result["attempt_count"] == 1
    assert "https://blocked.test/secret" not in calls


def test_robots_denied_no_page_or_raw(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(200, text="User-agent: medical-rag-research\nDisallow: /")
    result = fetch(tmp_path, handler)
    assert result["status"] == "robots_denied"
    assert result["robots_allowed"] is False
    assert result["attempt_count"] == 0
    assert calls == ["https://example.test/robots.txt"]
    assert not (tmp_path / "raw/42.bin").exists()


def test_robots_404_allows_page(tmp_path):
    result = fetch(tmp_path, lambda request: httpx.Response(404) if request.url.path == "/robots.txt"
                   else httpx.Response(200, content=b"ok"))
    assert result["status"] == "success"
    assert result["robots_status"] == 404 and result["robots_allowed"] is True


def test_http_robots_may_upgrade_to_https_on_same_hostname(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt" and request.url.scheme == "http":
            return httpx.Response(301, headers={"location": "https://example.test/robots.txt"})
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(200, content=b"page")
    result = fetch(tmp_path, handler, url="http://example.test/page")
    assert result["status"] == "success"
    assert result["robots_url"] == "https://example.test/robots.txt"
    assert calls[:3] == ["http://example.test/robots.txt",
                         "https://example.test/robots.txt", "http://example.test/page"]


def test_cross_host_robots_redirect_applies_original_authority(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.host == "example.test" and request.url.path == "/robots.txt":
            return httpx.Response(301, headers={"location": "https://other.test/robots.txt"})
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /page")
        raise AssertionError("Disallowed page must not be requested")
    result = fetch(tmp_path, handler, url="http://example.test/page")
    assert result["status"] == "robots_denied"
    assert result["robots_policy_status"] == "denied"
    assert result["robots_final_url"] == "https://other.test/robots.txt"
    assert result["robots_redirect_count"] == 1
    assert calls == ["http://example.test/robots.txt", "https://other.test/robots.txt"]


@pytest.mark.parametrize("robot_status", [429, 503, 521])
def test_robots_http_error_is_conservative(tmp_path, robot_status):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        return httpx.Response(robot_status)
    result = fetch(tmp_path, handler)
    assert result["status"] == "robots_unavailable"
    assert result["http_status"] is None
    assert result["attempt_count"] == 0
    assert len(calls) == 2
    assert result["robots_policy_status"] == "unreachable"


@pytest.mark.parametrize("robot_status", [403, 404])
def test_robots_4xx_allows_page_then_page_error_is_recorded(tmp_path, robot_status):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(robot_status)
        return httpx.Response(403)
    result = fetch(tmp_path, handler)
    assert result["status"] == "http_error"
    assert result["http_status"] == 403
    assert result["robots_policy_status"] == "unavailable_allow"
    assert result["robots_allowed"] is True
    assert len(calls) == 2


def test_robots_five_redirects_across_hosts(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt" and request.url.host != "hop5.test":
            index = 0 if request.url.host == "example.test" else int(request.url.host[3])
            return httpx.Response(301, headers={
                "location": f"https://hop{index + 1}.test/robots.txt"})
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nAllow: /")
        return httpx.Response(200, content=b"page")
    result = fetch(tmp_path, handler)
    assert result["status"] == "success"
    assert result["robots_redirect_count"] == 5
    assert result["robots_final_url"] == "https://hop5.test/robots.txt"
    assert len(calls) == 7


def test_robots_html_without_rules_warns_but_allows(tmp_path):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="<html><body>No rules</body></html>",
                                  headers={"content-type": "text/html"})
        return httpx.Response(200, content=b"page")
    result = fetch(tmp_path, handler)
    assert result["status"] == "success"
    assert result["robots_allowed"] is True
    assert result["robots_error_type"] == "no_parseable_rules"


def test_robots_valid_and_invalid_lines_uses_valid_rule(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="<bad>\nUser-agent: *\nINVALID LINE\nDisallow: /page")
        raise AssertionError("Page must not be requested")
    result = fetch(tmp_path, handler)
    assert result["status"] == "robots_denied"
    assert len(calls) == 1


def test_robots_invalid_redirect_target_and_protocol_error(tmp_path):
    def invalid(request):
        return httpx.Response(301, headers={"location": "ftp://elsewhere.test/robots.txt"})
    first = fetch(tmp_path, invalid)
    assert first["status"] == "robots_unavailable"
    assert first["robots_policy_status"] == "unreachable"
    def protocol(request):
        raise httpx.RemoteProtocolError("bad protocol")
    second = fetch(tmp_path, protocol)
    assert second["status"] == "robots_unavailable"
    assert second["robots_error_type"] == "RemoteProtocolError"


def test_robots_timeout_retries_once_then_stops(tmp_path):
    calls = []
    def handler(request):
        calls.append(str(request.url))
        raise httpx.ReadTimeout("robots timeout")
    result = fetch(tmp_path, handler)
    assert result["status"] == "robots_unavailable"
    assert result["robots_status"] is None
    assert result["attempt_count"] == 0
    assert len(calls) == 2


def test_page_timeout_retries_once_and_leaves_no_raw(tmp_path):
    calls = []
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls.append(str(request.url))
        raise httpx.ReadTimeout("page timeout")
    result = fetch(tmp_path, handler)
    assert result["status"] == "timeout"
    assert result["attempt_count"] == len(calls) == 2
    assert not list((tmp_path / "raw").glob("*")) if (tmp_path / "raw").exists() else True


@pytest.mark.parametrize("code,attempts", [(400, 1), (401, 1), (403, 1), (404, 1), (429, 2), (500, 2)])
def test_http_errors_retry_policy(tmp_path, code, attempts):
    pages = []
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        pages.append(str(request.url))
        return httpx.Response(code)
    result = fetch(tmp_path, handler)
    assert result["status"] == "http_error"
    assert result["http_status"] == code
    assert result["attempt_count"] == len(pages) == attempts
    assert not (tmp_path / "raw/42.bin").exists()


class CountingStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.reads = 0
    def __iter__(self):
        for chunk in self.chunks:
            self.reads += 1
            yield chunk


def test_content_length_too_large_does_not_read_body(tmp_path):
    stream = CountingStream([b"long body"])
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, headers={"content-length": "100"}, stream=stream)
    result = fetch(tmp_path, handler, max_bytes=5)
    assert result["status"] == "too_large"
    assert result["bytes_downloaded"] == 0
    assert stream.reads == 0
    assert not (tmp_path / "raw/42.bin").exists()


def test_stream_limit_discards_partial_raw(tmp_path):
    stream = CountingStream([b"abcd", b"efgh"])
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, stream=stream)
    result = fetch(tmp_path, handler, max_bytes=5)
    assert result["status"] == "too_large"
    assert result["raw_file"] is None
    assert not list((tmp_path / "raw").glob("*"))


def test_connection_reset_retries_but_other_network_error_does_not(tmp_path):
    calls = []
    def reset_handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        calls.append(1)
        if len(calls) == 1:
            raise httpx.ReadError("connection reset by peer")
        return httpx.Response(200, content=b"recovered")
    recovered = fetch(tmp_path, reset_handler)
    assert recovered["status"] == "success" and recovered["attempt_count"] == 2
    def dns_handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        raise httpx.ConnectError("DNS lookup failed")
    failed = fetch(tmp_path, dns_handler)
    assert failed["status"] == "network_error" and failed["attempt_count"] == 1
    assert not (tmp_path / "raw/42.bin").exists()  # Old success body removed after failed force-like fetch.


def test_unsupported_scheme_and_url_credentials_never_request(tmp_path):
    def blocked(_):
        raise AssertionError("No HTTP request expected")
    assert fetch(tmp_path, blocked, url="ftp://example.test/file")["status"] == "unsupported_scheme"
    assert fetch(tmp_path, blocked, url="https://user:password@example.test/file")["status"] == "unsupported_scheme"


def test_redirect_limit_and_challenge_not_saved(tmp_path):
    def redirect_handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(302, headers={"location": "/other"})
    result = fetch(tmp_path, redirect_handler, max_redirects=0)
    assert result["status"] == "http_error" and result["error_type"] == "too_many_redirects"
    def challenge_handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, text="<title>Just a Moment</title>",
                              headers={"content-type": "text/html"})
    result = fetch(tmp_path, challenge_handler)
    assert result["status"] == "http_error" and result["error_type"] == "access_challenge"
    assert not (tmp_path / "raw/42.bin").exists()


def test_cookie_reload_challenge_is_not_saved(tmp_path):
    body = (b'<html><body><script>document.cookie="D1N=abc";'
            b'window.location.reload(true);</script></body></html>')
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(404)
        return httpx.Response(200, content=body, headers={"content-type": "text/html"})
    result = fetch(tmp_path, handler)
    assert result["status"] == "http_error"
    assert result["error_type"] == "access_challenge"
    assert result["raw_file"] is None
    assert not (tmp_path / "raw/42.bin").exists()


def test_mocked_transport_never_uses_socket(tmp_path, monkeypatch):
    def blocked(*args, **kwargs):
        raise AssertionError("Network call attempted")
    monkeypatch.setattr(socket.socket, "connect", blocked)
    result = fetch(tmp_path, lambda request: httpx.Response(404) if request.url.path == "/robots.txt"
                   else httpx.Response(200, content=b"ok"))
    assert result["status"] == "success"

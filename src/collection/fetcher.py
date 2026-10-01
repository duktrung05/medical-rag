"""Sequential HTTP fetcher with robots checks, bounded bodies, and safe raw files."""

import hashlib
import os
import re
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from tempfile import mkstemp
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx


@dataclass
class RobotPolicy:
    url: str
    status: int | None
    parser: RobotFileParser | None = None
    allow_all: bool = False
    error_type: str | None = None
    error_message: str | None = None
    policy_status: str = "fetched"
    redirect_count: int = 0

    def allowed(self, user_agent: str, page_url: str) -> bool | None:
        if self.policy_status == "unreachable":
            return None
        if self.allow_all:
            return True
        return self.parser.can_fetch(user_agent, page_url) if self.parser else None


class PoliteFetcher:
    """Fetch only URLs passed by the caller; manually check each redirect origin."""

    def __init__(self, output_dir: Path, *, timeout_seconds: float = 30,
                 max_bytes: int = 5_242_880, delay_seconds: float = 0.5,
                 user_agent: str = "medical-rag-research/0.1", max_redirects: int = 5,
                 client: httpx.Client | None = None, sleep=time.sleep, clock=time.monotonic):
        if timeout_seconds <= 0 or max_bytes <= 0 or delay_seconds < 0 or max_redirects < 0:
            raise ValueError("timeout/max_bytes must be positive; delay/max_redirects nonnegative")
        if not user_agent.strip() or "\n" in user_agent or "\r" in user_agent:
            raise ValueError("Invalid User-Agent")
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / "raw"
        self.timeout = timeout_seconds
        self.max_bytes = max_bytes
        self.delay = delay_seconds
        self.user_agent = user_agent
        self.max_redirects = max_redirects
        self.sleep = sleep
        self.clock = clock
        self.last_page_at: float | None = None
        self.robots_cache: dict[tuple, RobotPolicy] = {}
        self.owns_client = client is None
        self.client = client or httpx.Client(follow_redirects=False, trust_env=True,
                                             limits=httpx.Limits(max_connections=1,
                                                                 max_keepalive_connections=1))

    def close(self):
        if self.owns_client:
            self.client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _origin(self, url: str) -> tuple[tuple, str] | None:
        try:
            parts = urlsplit(url)
            if parts.scheme.lower() not in ("http", "https") or not parts.hostname:
                return None
            if parts.username is not None or parts.password is not None:
                return None  # Never turn URL credentials into Authorization.
            port = parts.port
            key = (parts.scheme.lower(), parts.hostname.lower().rstrip("."), port)
            return key, f"{parts.scheme.lower()}://{parts.netloc.lower()}/robots.txt"
        except ValueError:
            return None

    @staticmethod
    def _connection_reset(error: BaseException) -> bool:
        cause = error
        while cause is not None:
            if isinstance(cause, ConnectionResetError) or "connection reset" in str(cause).lower():
                return True
            cause = cause.__cause__
        return False

    def _request(self, url: str):
        # httpx stores Set-Cookie in its jar; clear it before every request.
        self.client.cookies.clear()
        return self.client.stream("GET", url, timeout=self.timeout,
                                  follow_redirects=False,
                                  headers={"User-Agent": self.user_agent})

    def _robots(self, url: str) -> RobotPolicy:
        origin = self._origin(url)
        if origin is None:
            return RobotPolicy("", None, error_type="unsupported_scheme",
                               error_message="Only credential-free HTTP(S) URLs are allowed",
                               policy_status="unreachable")
        key, robots_url = origin
        if key in self.robots_cache:
            return self.robots_cache[key]
        current = robots_url
        redirects = 0
        retries = 0
        visited = set()
        while True:
            if current in visited:
                policy = RobotPolicy(current, None, error_type="robots_redirect_loop",
                                     error_message="Robots redirect loop detected",
                                     policy_status="unreachable", redirect_count=redirects)
                break
            visited.add(current)
            try:
                with self._request(current) as response:
                    status = response.status_code
                    if status in (301, 302, 303, 307, 308) and response.headers.get("location"):
                        target = urljoin(current, response.headers["location"])
                        if self._origin(target) is None or redirects >= 5:
                            policy = RobotPolicy(current, status, error_type="robots_redirect",
                                                 error_message="Invalid robots redirect or exceeded five hops",
                                                 policy_status="unreachable",
                                                 redirect_count=redirects)
                            break
                        current = target
                        redirects += 1
                        continue
                    if 400 <= status < 500 and status != 429:
                        policy = RobotPolicy(current, status, allow_all=True,
                                             policy_status="unavailable_allow",
                                             redirect_count=redirects)
                        break
                    if status == 429 or status >= 500:
                        if retries < 1:
                            retries += 1
                            self._backoff(retries, response.headers.get("retry-after"))
                            visited.discard(current)
                            continue
                        policy = RobotPolicy(current, status, error_type="robots_http_error",
                                             error_message=f"robots.txt returned HTTP {status}",
                                             policy_status="unreachable", redirect_count=redirects)
                        break
                    if not 200 <= status < 300:
                        policy = RobotPolicy(current, status, error_type="robots_http_error",
                                             error_message=f"robots.txt returned HTTP {status}",
                                             policy_status="unreachable", redirect_count=redirects)
                        break
                    body = bytearray()
                    for chunk in response.iter_bytes(chunk_size=65_536):
                        if len(body) + len(chunk) > 1_048_576:
                            raise ValueError("robots.txt exceeded 1 MiB")
                        body.extend(chunk)
                    lines = body.decode("utf-8", errors="replace").splitlines()
                    parser = RobotFileParser()
                    parser.parse(lines)
                    has_agent = any(re.match(r"^\s*user-agent\s*:\s*([A-Za-z_-]+|\*)\s*(?:#.*)?$",
                                             line, re.IGNORECASE) for line in lines)
                    has_rule = any(re.match(r"^\s*(?:allow|disallow)\s*:\s*(?:/[^#]*|)\s*(?:#.*)?$",
                                            line, re.IGNORECASE) for line in lines)
                    parseable = has_agent and has_rule
                    policy = RobotPolicy(current, status, parser=parser if parseable else None,
                                         allow_all=not parseable, redirect_count=redirects)
                    if not parseable:
                        policy.error_type = "no_parseable_rules"
                        policy.error_message = "HTTP 2xx robots body has no parseable rules"
                    break
            except (httpx.TimeoutException, httpx.TransportError) as error:
                if retries < 1 and (isinstance(error, httpx.TimeoutException)
                                    or self._connection_reset(error)):
                    retries += 1
                    self._backoff(retries)
                    visited.discard(current)
                    continue
                policy = RobotPolicy(current, None, error_type=type(error).__name__,
                                     error_message=str(error), policy_status="unreachable",
                                     redirect_count=redirects)
                break
            except (httpx.RequestError, httpx.StreamError) as error:
                policy = RobotPolicy(current, None, error_type=type(error).__name__,
                                     error_message=str(error), policy_status="unreachable",
                                     redirect_count=redirects)
                break
            except ValueError as error:
                policy = RobotPolicy(current, 200, error_type="invalid_robots",
                                     error_message=str(error), policy_status="unreachable",
                                     redirect_count=redirects)
                break
        self.robots_cache[key] = policy
        return policy

    def _wait_for_page(self):
        if self.last_page_at is not None:
            wait = self.delay - (self.clock() - self.last_page_at)
            if wait > 0:
                self.sleep(wait)
        self.last_page_at = self.clock()

    def _backoff(self, retry_number: int, retry_after: str | None = None) -> None:
        """Delay before a retry; batch crawling overrides this policy."""
        self.sleep(0.2)

    @staticmethod
    def _challenge(prefix: bytes, content_type: str | None) -> bool:
        if not content_type or "text/html" not in content_type.lower():
            return False
        head = prefix.lower()
        if b"document.cookie" in head and (b"location.reload" in head
                                           or b"window.location" in head):
            return True
        return any(marker in head for marker in (
            b"cf-chl", b"cf-browser-verification", b"verify you are human",
            b"<title>captcha", b"<title>access denied", b"<title>just a moment",
        ))

    def fetch(self, entry: dict) -> dict:
        start = self.clock()
        official_id, initial_url = entry["id"], entry["url"]
        raw_path = self.raw_dir / f"{official_id}.bin"
        result = {
            "id": official_id, "url": initial_url, "domain": entry["domain"],
            "selected_group": entry["selected_group"], "status": None,
            "attempt_count": 0, "robots_url": None, "robots_status": None,
            "robots_policy_status": None, "robots_http_status": None,
            "robots_final_url": None, "robots_redirect_count": 0,
            "robots_error_type": None, "robots_error_message": None,
            "robots_allowed": None, "http_status": None, "final_url": initial_url,
            "redirect_count": 0, "content_type": None, "declared_content_length": None,
            "bytes_downloaded": 0, "elapsed_ms": 0.0, "fetched_at": None,
            "content_sha256": None, "raw_file": None,
            "error_type": None, "error_message": None,
        }

        def finish(status: str, error_type: str | None = None,
                   message: str | None = None) -> dict:
            result["status"] = status
            result["error_type"] = error_type
            result["error_message"] = message
            result["elapsed_ms"] = round((self.clock() - start) * 1000, 3)
            result["fetched_at"] = datetime.now(UTC).isoformat()
            if status != "success":
                raw_path.unlink(missing_ok=True)  # Remove stale raw body after --force failure.
            return result

        current = initial_url
        visited = set()
        retries = 0
        while True:
            if self._origin(current) is None:
                return finish("unsupported_scheme", "invalid_target",
                              "Only credential-free HTTP(S) URLs are allowed")
            if current in visited:
                return finish("http_error", "redirect_loop", "Redirect loop detected")
            visited.add(current)
            result["final_url"] = current
            policy = self._robots(current)
            result.update(robots_url=policy.url or None, robots_status=policy.status,
                          robots_http_status=policy.status,
                          robots_final_url=policy.url or None,
                          robots_redirect_count=policy.redirect_count,
                          robots_policy_status=policy.policy_status,
                          robots_error_type=policy.error_type,
                          robots_error_message=policy.error_message,
                          robots_allowed=policy.allowed(self.user_agent, current))
            if policy.policy_status == "unreachable":
                return finish("robots_unavailable", policy.error_type, policy.error_message)
            if result["robots_allowed"] is False:
                result["robots_policy_status"] = "denied"
                result["robots_error_type"] = "robots_disallow"
                result["robots_error_message"] = "robots.txt disallows this URL"
                return finish("robots_denied", "robots_disallow", "robots.txt disallows this URL")
            self._wait_for_page()
            result["attempt_count"] += 1
            temporary = None
            try:
                with self._request(current) as response:
                    status = response.status_code
                    result["http_status"] = status
                    result["content_type"] = response.headers.get("content-type")
                    declared = response.headers.get("content-length")
                    try:
                        result["declared_content_length"] = int(declared) if declared is not None else None
                    except ValueError:
                        result["declared_content_length"] = None
                    if status in (301, 302, 303, 307, 308) and response.headers.get("location"):
                        if result["redirect_count"] >= self.max_redirects:
                            return finish("http_error", "too_many_redirects", "Redirect limit exceeded")
                        current = urljoin(current, response.headers["location"])
                        result["redirect_count"] += 1
                        retries = 0
                        continue
                    if status == 429 or status >= 500:
                        if retries < 1:
                            retries += 1
                            self._backoff(retries, response.headers.get("retry-after"))
                            visited.discard(current)
                            continue
                        return finish("http_error", f"http_{status}", f"HTTP {status}")
                    if not 200 <= status < 300:
                        return finish("http_error", f"http_{status}", f"HTTP {status}")
                    if (result["declared_content_length"] is not None
                            and result["declared_content_length"] > self.max_bytes):
                        return finish("too_large", "content_length", "Content-Length exceeds max_bytes")
                    self.raw_dir.mkdir(parents=True, exist_ok=True)
                    handle, name = mkstemp(prefix=f".{official_id}.", suffix=".tmp", dir=self.raw_dir)
                    temporary = Path(name)
                    digest = hashlib.sha256()
                    preview = bytearray()
                    body_bytes = 0
                    with os.fdopen(handle, "wb") as raw:
                        for chunk in response.iter_bytes(chunk_size=65_536):
                            result["bytes_downloaded"] += len(chunk)
                            body_bytes += len(chunk)
                            if body_bytes > self.max_bytes:
                                return finish("too_large", "stream_limit", "Stream exceeded max_bytes")
                            raw.write(chunk)
                            digest.update(chunk)
                            if len(preview) < 16_384:
                                preview.extend(chunk[:16_384 - len(preview)])
                    if self._challenge(preview, result["content_type"]):
                        return finish("http_error", "access_challenge", "Access challenge detected")
                    temporary.replace(raw_path)
                    temporary = None
                    result["content_sha256"] = digest.hexdigest()
                    result["raw_file"] = f"raw/{official_id}.bin"
                    return finish("success")
            except httpx.TimeoutException as error:
                if retries < 1:
                    retries += 1
                    self._backoff(retries)
                    visited.discard(current)
                    continue
                return finish("timeout", type(error).__name__, str(error))
            except httpx.TransportError as error:
                if retries < 1 and self._connection_reset(error):
                    retries += 1
                    self._backoff(retries)
                    visited.discard(current)
                    continue
                return finish("network_error", type(error).__name__, str(error))
            except (httpx.RequestError, httpx.StreamError) as error:
                return finish("network_error", type(error).__name__, str(error))
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)


class BatchFetchState:
    """Shared robots decisions and origin throttles across batch workers."""

    def __init__(self):
        self.guard = threading.Lock()
        self.locks: dict[str, threading.RLock] = {}
        self.last_page_at: dict[str, float] = {}
        self.robots_cache: dict[tuple, RobotPolicy] = {}

    def origin_lock(self, key: str) -> threading.RLock:
        with self.guard:
            return self.locks.setdefault(key, threading.RLock())


class BatchFetcher(PoliteFetcher):
    """Reuse probe fetch/robots logic with per-origin locks and batch retry policy."""

    def __init__(self, output_dir: Path, *, state: BatchFetchState,
                 timeout_seconds: float = 30, max_bytes: int = 15_728_640,
                 delay_seconds: float = 1, user_agent: str = "medical-rag-vibiomir-pilot/0.1",
                 client: httpx.Client | None = None, sleep=time.sleep,
                 clock=time.monotonic):
        super().__init__(output_dir, timeout_seconds=timeout_seconds, max_bytes=max_bytes,
                         delay_seconds=delay_seconds, user_agent=user_agent,
                         client=client, sleep=sleep, clock=clock)
        self.state = state
        self.robots_cache = state.robots_cache
        self._local = threading.local()

    def _robots(self, url: str) -> RobotPolicy:
        origin = self._origin(url)
        if origin is None:
            return super()._robots(url)
        with self.state.origin_lock(origin[0][1].removeprefix("www.")):
            self._local.robots_request = True
            try:
                return super()._robots(url)
            finally:
                self._local.robots_request = False

    def _wait_for_page(self):
        # The origin lock in _request must contain both waiting and streaming.
        return None

    @contextmanager
    def _request(self, url: str):
        origin = self._origin(url)
        if origin is None:
            raise ValueError("Invalid HTTP(S) URL")
        key = origin[0][1].removeprefix("www.")
        with self.state.origin_lock(key):
            if not getattr(self._local, "robots_request", False):
                previous = self.state.last_page_at.get(key)
                if previous is not None:
                    wait = self.delay - (self.clock() - previous)
                    if wait > 0:
                        self.sleep(wait)
                self.state.last_page_at[key] = self.clock()
            with super()._request(url) as response:
                yield response

    def _backoff(self, retry_number: int, retry_after: str | None = None) -> None:
        delay = float(2 ** (retry_number - 1))
        if retry_after:
            try:
                delay = max(delay, float(retry_after))
            except ValueError:
                try:
                    when = parsedate_to_datetime(retry_after)
                    if when.tzinfo is not None:
                        delay = max(delay, (when - datetime.now(UTC)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        self.sleep(max(0.0, delay))

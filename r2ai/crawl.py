"""Async crawler for ViBioMIR corpus URLs.

Politeness is per-domain: a bounded number of in-flight requests per host plus a
minimum gap between them, so a domain holding most of the corpus cannot be
hammered. Raw bytes are kept gzipped on disk so extraction can be re-run without
re-fetching, and a manifest makes every run resumable.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import aiohttp
import pyarrow as pa
import pyarrow.parquet as pq

# Run as a script (python r2ai/crawl.py), so the package root is not on
# sys.path by default; the deferred import of r2ai.deadhosts needs it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data/raw/vibiomir"
OUT = ROOT / "data/vibiomir"
CRAWL = OUT / "crawl"
PAGES = CRAWL / "pages"

# Several corpus hosts 403 any non-browser agent string. robots.txt is still
# honoured and per-domain rate limits still apply; only the agent string changes.
UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
BROWSER_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9,en;q=0.8,zh-CN;q=0.7",
    "Accept-Encoding": "gzip, deflate, br",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Cache-Control": "max-age=0",
}
MANIFEST_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("url", pa.string()), ("domain", pa.string()),
    ("status", pa.string()), ("http_status", pa.int32()),
    ("content_type", pa.string()), ("bytes", pa.int64()),
    ("final_url", pa.string()), ("error", pa.string()),
])
MAX_BYTES = 6 * 1024 * 1024


# A host that bounces us to a verification page still answers 200, so without
# these checks the crawl records thousands of interstitials as successful pages
# and they silently pollute the index.
_VERIFY_URL = re.compile(r"/(verify|captcha|checkcode|seccode|challenge)\b", re.I)
# Some hosts answer with a tiny script that sets a cookie and reloads; echoing
# that cookie is exactly what a browser does, so one retry recovers the page.
_COOKIE_JS = re.compile(rb'document\.cookie\s*=\s*"([^"=]+)=([^";]+)')


def verification_redirect(final_url: str, original: str) -> bool:
    return bool(final_url) and final_url != original and bool(_VERIFY_URL.search(final_url))


def cookie_challenge(body: bytes) -> tuple[str, str] | None:
    """Return the (name, value) a cookie-wall page asks us to set, if any."""
    if len(body) > 4096 or b"document.cookie" not in body:
        return None
    match = _COOKIE_JS.search(body)
    if not match:
        return None
    return match.group(1).decode("latin-1"), match.group(2).decode("latin-1")


def shard_path(doc_id: int) -> Path:
    """Spread pages over 1000 directories to keep directory sizes sane."""
    return PAGES / f"{doc_id % 1000:03d}" / f"{doc_id}.gz"


# Hosts observed to return 403/429 under normal concurrency; these are capped
# below the size-derived budget rather than above it.
SENSITIVE_DOMAINS = {
    # 39.net starts serving image.39.net/verify.html under sustained load, but
    # recovers; 20 is the compromise between that risk and its 417k remaining URLs.
    "39.net": 20,
    "nhathuoclongchau.com.vn": 2,
    "tiemchunglongchau.com.vn": 2,
    "www.qdnd.vn": 2,
    "baohaiphong.vn": 2,
    "baonghean.vn": 2,
    "baodanang.vn": 2,
}


# Suffixes under which the label before them is still a registrar-level domain,
# so "ask.39.net" and "baby.39.net" must share one budget rather than get one each.
_MULTI_SUFFIX = {
    "com.cn", "net.cn", "org.cn", "gov.cn", "edu.cn",
    "com.vn", "net.vn", "org.vn", "gov.vn", "edu.vn",
    "co.uk", "com.tw", "com.hk",
}


def base_domain(netloc: str) -> str:
    """Registrable domain, so sibling subdomains share one rate-limit budget.

    39.net spreads its corpus over ~15 subdomains. Treating each as its own host
    gave the shared backend ~90 concurrent requests and got every one of them
    blocked, losing ~460k URLs.
    """
    host = netloc.split(":")[0].lower().rstrip(".")
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_SUFFIX:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def plan_domain_limits(
    targets: list[tuple[int, str]], base: int, cap: int, per_slot: int = 20_000,
) -> dict[str, int]:
    """Give hosts holding more of the corpus a proportionally larger budget.

    With a uniform limit the few hosts that own most of the corpus set the wall
    time: 120ask alone holds ~918k URLs and answers in seconds, so at 8 in flight
    it would need days on its own while small hosts finish in minutes. Hosts
    known to throttle are pinned low regardless of how much corpus they hold.
    """
    sizes: Counter[str] = Counter(base_domain(urlsplit(url).netloc) for _, url in targets)
    limits = {
        domain: max(base, min(cap, -(-count // per_slot)))
        for domain, count in sizes.items()
    }
    for domain, ceiling in SENSITIVE_DOMAINS.items():
        key = base_domain(domain)
        if key in limits:
            # A caller may deliberately request a lower cap for a recovery pass.
            # Sensitive-domain defaults must never raise that explicit limit.
            limits[key] = min(limits[key], ceiling)
    return limits


class DomainLimiter:
    """Bounded concurrency plus a minimum delay per host."""

    def __init__(self, per_domain: int, delay: float,
                 limits: dict[str, int] | None = None) -> None:
        self._per_domain = per_domain
        self._delay = delay
        self._limits = limits or {}
        self._sem: dict[str, asyncio.Semaphore] = {}
        self._next_ok: dict[str, float] = defaultdict(float)

    def semaphore(self, domain: str) -> asyncio.Semaphore:
        if domain not in self._sem:
            self._sem[domain] = asyncio.Semaphore(self._limits.get(domain, self._per_domain))
        return self._sem[domain]

    async def wait(self, domain: str) -> None:
        now = time.monotonic()
        gap = self._next_ok[domain] - now
        if gap > 0:
            await asyncio.sleep(gap)
        self._next_ok[domain] = max(now, self._next_ok[domain]) + self._delay


class CircuitBreaker:
    """Stop hammering a host that has started refusing everything.

    A host behind a Cloudflare challenge fails every request, but each attempt
    still costs a retry plus backoff while holding one of its few slots. Left
    alone, ~11k such URLs can add many hours to a run and starve healthy hosts of
    workers. After `threshold` consecutive failures the host is skipped outright.
    """

    def __init__(self, threshold: int = 40) -> None:
        self._threshold = threshold
        self._consecutive: defaultdict[str, int] = defaultdict(int)
        self._open: set[str] = set()

    def record(self, domain: str, ok: bool) -> None:
        if ok:
            self._consecutive[domain] = 0
            return
        self._consecutive[domain] += 1
        if self._consecutive[domain] >= self._threshold and domain not in self._open:
            self._open.add(domain)
            print(f"  circuit open: skipping {domain} after "
                  f"{self._threshold} consecutive failures", flush=True)

    def is_open(self, domain: str) -> bool:
        return domain in self._open

    @property
    def opened(self) -> set[str]:
        return self._open


class RobotsCache:
    """Fetch and cache robots.txt per host; fail open on fetch errors."""

    def __init__(self, session: aiohttp.ClientSession, enabled: bool) -> None:
        self._session = session
        self._enabled = enabled
        self._cache: dict[str, RobotFileParser | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, url: str) -> bool:
        if not self._enabled:
            return True
        parts = urlsplit(url)
        host = f"{parts.scheme}://{parts.netloc}"
        if host not in self._cache:
            lock = self._locks.setdefault(host, asyncio.Lock())
            async with lock:
                if host not in self._cache:
                    self._cache[host] = await self._load(host)
        parser = self._cache[host]
        return True if parser is None else parser.can_fetch(UA, url)

    async def _load(self, host: str) -> RobotFileParser | None:
        try:
            async with self._session.get(f"{host}/robots.txt", timeout=aiohttp.ClientTimeout(total=20)) as resp:
                if resp.status != 200:
                    return None
                body = await resp.text(errors="ignore")
        except Exception:
            return None
        parser = RobotFileParser()
        parser.parse(body.splitlines())
        return parser


async def fetch_one(
    session: aiohttp.ClientSession, robots: RobotsCache, limiter: DomainLimiter,
    doc_id: int, url: str, retries: int, breaker: CircuitBreaker | None = None,
    cookies: dict[str, dict[str, str]] | None = None,
) -> dict:
    netloc = urlsplit(url).netloc
    cookies = {} if cookies is None else cookies
    # Rate limiting and circuit breaking are per registrable domain; the manifest
    # still records the exact host it fetched.
    domain = base_domain(netloc)
    row = {
        "doc_id": doc_id, "url": url, "domain": netloc, "status": "error",
        "http_status": 0, "content_type": "", "bytes": 0, "final_url": "", "error": "",
    }
    if breaker is not None and breaker.is_open(domain):
        row["status"] = "circuit_open"
        return row
    try:
        if not await robots.allowed(url):
            row["status"] = "robots_denied"
            return row
    except Exception as exc:  # noqa: BLE001
        row["error"] = f"robots:{type(exc).__name__}"

    for attempt in range(retries + 1):
        backoff = 0.0
        # The semaphore is released before any backoff sleep: holding a host's slot
        # (and a global slot) while merely waiting lets a few throttling hosts stall
        # the whole crawl.
        async with limiter.semaphore(domain):
            await limiter.wait(domain)
            try:
                async with session.get(url, allow_redirects=True,
                                       cookies=cookies.get(domain)) as resp:
                    row["http_status"] = resp.status
                    row["final_url"] = str(resp.url)
                    row["content_type"] = resp.headers.get("Content-Type", "")[:120]
                    if resp.status != 200:
                        row["status"] = "http_error"
                        # 403 here is rate-based WAF blocking, so it is worth a slower retry.
                        if resp.status in (403, 429, 500, 502, 503, 504) and attempt < retries:
                            backoff = 3 * (attempt + 1) + random.random() * 2
                        else:
                            return row
                    else:
                        # content.read(n) returns only what is already buffered,
                        # which silently truncated large pages to their first
                        # chunk. Accumulate explicitly, still capped at MAX_BYTES.
                        buf = bytearray()
                        async for part in resp.content.iter_chunked(65536):
                            buf.extend(part)
                            if len(buf) >= MAX_BYTES:
                                break
                        body = bytes(buf)
                        if verification_redirect(row["final_url"], url):
                            row.update(status="blocked", bytes=len(body),
                                       error="verification page")
                            return row
                        challenge = cookie_challenge(body)
                        if challenge and attempt < retries:
                            cookies[domain] = {challenge[0]: challenge[1]}
                            row["error"] = "cookie challenge"
                            continue
                        path = shard_path(doc_id)
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_bytes(gzip.compress(body, compresslevel=5))
                        row.update(status="ok", bytes=len(body), error="")
                        return row
            except Exception as exc:  # noqa: BLE001
                row["error"] = f"{type(exc).__name__}"
                if attempt < retries:
                    backoff = 2 ** attempt + random.random()
                else:
                    row["status"] = ("timeout" if isinstance(exc, asyncio.TimeoutError)
                                     else "network_error")
                    return row
        if backoff:
            await asyncio.sleep(backoff)
    return row


def load_targets(source: str, limit: int | None) -> list[tuple[int, str]]:
    corpus = pq.read_table(RAW / "links_corpus.parquet").to_pydict()
    url_by_id = dict(zip(corpus["id"], corpus["url"]))
    if source == "candidates":
        cand = pq.read_table(OUT / "url_candidates.parquet").to_pydict()
        ids = list(dict.fromkeys(cand["doc_id"]))
    else:
        ids = list(url_by_id)
    targets = [(i, url_by_id[i]) for i in ids if i in url_by_id]
    return targets[:limit] if limit else targets


def interleave_by_domain(
    targets: list[tuple[int, str]], weights: dict[str, int] | None = None,
) -> list[tuple[int, str]]:
    """Interleave hosts, giving each a share of the queue matching its capacity.

    Candidate order groups many URLs of one host together, so the global
    concurrency slots fill with requests that then queue behind that host's own
    limit while other hosts sit idle.

    Plain round-robin fixes that but overcorrects: it hands every host an equal
    share, so a host with 900k URLs left and 45 concurrent slots gets the same
    share as one with a handful of URLs and 8 slots. The big host is then starved
    and overall throughput collapses to a fraction of the available capacity.
    Emitting `weights[domain]` items per cycle keeps each host supplied to its
    own limit.
    """
    buckets: dict[str, list[tuple[int, str]]] = defaultdict(list)
    for doc_id, url in targets:
        buckets[base_domain(urlsplit(url).netloc)].append((doc_id, url))
    live = [(domain, queue) for domain, queue in buckets.items() if queue]
    ordered: list[tuple[int, str]] = []
    while live:
        nxt = []
        for domain, queue in live:
            for _ in range(max(1, (weights or {}).get(domain, 1))):
                if not queue:
                    break
                ordered.append(queue.pop())
            if queue:
                nxt.append((domain, queue))
        live = nxt
    return ordered


def already_done(skip_failed: bool = False) -> set[int]:
    """Doc ids not worth fetching again.

    By default every id in the manifest counts as done. With skip_failed, only
    successful fetches do, so a resume pass retries the hosts that were
    throttling earlier (their per-domain budget may since have been lowered).
    """
    done: set[int] = set()
    for part in sorted(CRAWL.glob("manifest_*.parquet")):
        try:
            table = pq.read_table(part, columns=["doc_id", "status"]).to_pydict()
        except Exception:  # noqa: BLE001
            continue
        if skip_failed:
            done.update(d for d, st in zip(table["doc_id"], table["status"]) if st == "ok")
        else:
            done.update(table["doc_id"])
    # The manifest is only flushed every few thousand rows, so a run that is
    # interrupted leaves pages on disk that it never recorded. Those bytes are
    # already fetched; counting them avoids re-downloading them on resume.
    for page in PAGES.glob("*/*.gz"):
        try:
            done.add(int(page.stem))
        except ValueError:
            continue
    return done


async def run(source: str, limit: int | None, concurrency: int, per_domain: int,
              delay: float, retries: int, respect_robots: bool,
              per_domain_cap: int = 8, retry_failed: bool = False,
              circuit_threshold: int = 40, only_domains: str = "",
              exclude_domains: str = "", dry_run: bool = False) -> None:
    CRAWL.mkdir(parents=True, exist_ok=True)
    targets = load_targets(source, limit)
    done = already_done(skip_failed=retry_failed)
    todo = [(i, u) for i, u in targets if i not in done]
    print(f"targets={len(targets):,} done={len(done):,} todo={len(todo):,}", flush=True)
    if not todo:
        return
    from r2ai.deadhosts import load as load_dead_hosts
    # Applied even with --retry-failed: these hosts are permanently blocked, while
    # the hosts that retry-failed is meant to recover (e.g. ones fixed by the
    # cookie handshake) are no longer on the list.
    if only_domains:
        # A recovery pass: hosts that were working and then got flagged mid-run
        # are worth re-attempting on their own, at a rate low enough not to
        # trigger the same block again.
        wanted = {d.strip() for d in only_domains.split(",") if d.strip()}
        before = len(todo)
        todo = [(i, u) for i, u in todo
                if base_domain(urlsplit(u).netloc) in wanted]
        print(f"restricted to {sorted(wanted)}: {len(todo):,} of {before:,} urls",
              flush=True)

    if exclude_domains:
        excluded = {d.strip() for d in exclude_domains.split(",") if d.strip()}
        before = len(todo)
        todo = [(i, u) for i, u in todo
                if base_domain(urlsplit(u).netloc) not in excluded]
        print(f"excluded {sorted(excluded)}: removed {before - len(todo):,} urls",
              flush=True)

    dead = load_dead_hosts()
    if dead and not only_domains:
        before = len(todo)
        todo = [(i, u) for i, u in todo
                if base_domain(urlsplit(u).netloc) not in dead]
        print(f"skipping {before - len(todo):,} urls on {len(dead)} known-dead hosts",
              flush=True)

    print(f"effective todo after domain filters: {len(todo):,}", flush=True)
    if dry_run:
        print("DRY RUN: no network requests were made", flush=True)
        return

    limits = plan_domain_limits(todo, per_domain, per_domain_cap)
    todo = interleave_by_domain(todo, limits)
    top = sorted(limits.items(), key=lambda kv: -kv[1])[:5]
    print(f"per-domain budgets (top): {top}", flush=True)
    limiter = DomainLimiter(per_domain, delay, limits)
    stamp = int(time.time())
    rows: list[dict] = []
    counts: defaultdict[str, int] = defaultdict(int)
    part = 0
    started = time.time()

    connector = aiohttp.TCPConnector(limit=concurrency, limit_per_host=max(limits.values(), default=per_domain), ttl_dns_cache=600)
    timeout = aiohttp.ClientTimeout(total=45, connect=15, sock_read=25)
    headers = dict(BROWSER_HEADERS)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout, headers=headers) as session:
        robots = RobotsCache(session, respect_robots)
        breaker = CircuitBreaker(circuit_threshold)
        cookie_store: dict[str, dict[str, str]] = {}

        # A fixed pool pulling from a shared cursor, rather than one task per URL:
        # at 4.4M URLs the task-per-url form costs gigabytes of pending coroutines,
        # and a global gate would let slow hosts hold slots they are not using.
        cursor = 0
        completed = 0
        lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal cursor, completed, rows, part
            while True:
                async with lock:
                    if cursor >= len(todo):
                        return
                    doc_id, url = todo[cursor]
                    cursor += 1
                row = await fetch_one(session, robots, limiter, doc_id, url,
                                      retries, breaker, cookie_store)
                breaker.record(base_domain(row["domain"]), row["status"] == "ok")
                async with lock:
                    rows.append(row)
                    counts[row["status"]] += 1
                    completed += 1
                    if completed % 2000 == 0:
                        rate = completed / max(time.time() - started, 1e-6)
                        print(f"  {completed:,}/{len(todo):,}  {rate:.1f}/s  {dict(counts)}",
                              flush=True)
                    if len(rows) >= 5000:
                        _flush(rows, stamp, part)
                        part += 1
                        rows = []

        await asyncio.gather(*(asyncio.create_task(worker()) for _ in range(concurrency)))
    if rows:
        _flush(rows, stamp, part)
    print(f"DONE {dict(counts)} in {time.time() - started:.0f}s", flush=True)
    if breaker.opened:
        print(f"hosts skipped by circuit breaker: {sorted(breaker.opened)}", flush=True)


def _flush(rows: list[dict], stamp: int, part: int) -> None:
    table = pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA)
    pq.write_table(table, CRAWL / f"manifest_{stamp}_{part:05d}.parquet")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["candidates", "all"], default="candidates")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=192)
    ap.add_argument("--per-domain", type=int, default=8)
    ap.add_argument("--delay", type=float, default=0.12)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--per-domain-cap", type=int, default=8)
    ap.add_argument("--only-domains", default="",
                    help="comma-separated registrable domains to restrict this run to")
    ap.add_argument("--exclude-domains", default="",
                    help="comma-separated registrable domains to omit from this run")
    ap.add_argument("--circuit-threshold", type=int, default=40,
                    help="consecutive failures before a host is skipped")
    ap.add_argument("--retry-failed", action="store_true",
                    help="re-fetch urls whose previous attempt failed")
    ap.add_argument("--dry-run", action="store_true",
                    help="report targets/done/todo after filters without making network requests")
    ap.add_argument("--no-robots", action="store_true")
    a = ap.parse_args()
    try:
        asyncio.run(run(a.source, a.limit, a.concurrency, a.per_domain, a.delay,
                        a.retries, not a.no_robots, a.per_domain_cap,
                        a.retry_failed, a.circuit_threshold, a.only_domains,
                        a.exclude_domains, a.dry_run))
    except KeyboardInterrupt:
        sys.exit(130)

"""Feasibility probe: can we actually fetch the big Chinese-language domains?

The candidate crawl only exercised Vietnamese hosts, but ~70% of ViBioMIR lives on
Chinese sites. A few samples per domain tell us whether a full crawl is viable
before committing days of bandwidth to it.
"""
from __future__ import annotations

import argparse
import asyncio
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import aiohttp
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from r2ai.crawl import BROWSER_HEADERS, RAW  # noqa: E402


async def probe(url: str, session: aiohttp.ClientSession) -> tuple[str, int, int]:
    try:
        async with session.get(url, allow_redirects=True) as resp:
            body = await resp.content.read(300_000)
            return ("ok" if resp.status == 200 else "http_error", resp.status, len(body))
    except asyncio.TimeoutError:
        return ("timeout", 0, 0)
    except Exception:  # noqa: BLE001
        return ("network_error", 0, 0)


async def main(per_domain: int, top_domains: int) -> None:
    table = pq.read_table(RAW / "links_corpus.parquet").to_pydict()
    by_domain: dict[str, list[str]] = defaultdict(list)
    for url in table["url"]:
        by_domain[urlsplit(url).netloc].append(url)
    ranked = sorted(by_domain.items(), key=lambda kv: -len(kv[1]))[:top_domains]

    random.seed(7)
    timeout = aiohttp.ClientTimeout(total=30, connect=12)
    connector = aiohttp.TCPConnector(limit=40, limit_per_host=4)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout,
                                     headers=BROWSER_HEADERS) as session:
        for domain, urls in ranked:
            sample = random.sample(urls, min(per_domain, len(urls)))
            started = time.time()
            results = await asyncio.gather(*(probe(u, session) for u in sample))
            stats = Counter(r[0] for r in results)
            codes = Counter(r[1] for r in results if r[0] != "ok")
            sizes = [r[2] for r in results if r[0] == "ok"]
            avg = int(sum(sizes) / len(sizes)) if sizes else 0
            elapsed = time.time() - started
            print(
                f"{domain:34s} n={len(urls):>9,}  ok={stats['ok']}/{len(sample)}  "
                f"avg_bytes={avg:>7,}  {elapsed:5.1f}s  {dict(codes) if codes else ''}",
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=6)
    ap.add_argument("--top-domains", type=int, default=20)
    a = ap.parse_args()
    asyncio.run(main(a.per_domain, a.top_domains))

#!/usr/bin/env python3
"""Create a final URL-state report without modifying crawl data."""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/vibiomir"
CRAWL = ROOT / "data/vibiomir/crawl"
OUT = ROOT / "outputs/crawl_final_audit_20261006.json"


def domain(url: str) -> str:
    host = (urlsplit(url).hostname or "").lower()
    parts = host.split(".")
    return ".".join(parts[-2:]) if len(parts) >= 2 else host


def main() -> None:
    target = pq.read_table(RAW / "links_corpus.parquet", columns=["id", "url"]).to_pydict()
    urls = {int(i): u for i, u in zip(target["id"], target["url"])}
    latest: dict[int, tuple[str, str, str, int]] = {}
    raw_status = Counter()
    for part in sorted(CRAWL.glob("manifest_*.parquet")):
        cols = pq.read_table(part, columns=["doc_id", "domain", "status", "error", "http_status"]).to_pydict()
        for doc_id, dom, status, error, http_status in zip(
            cols["doc_id"], cols["domain"], cols["status"], cols["error"], cols["http_status"]
        ):
            doc_id = int(doc_id)
            if doc_id in urls:
                latest[doc_id] = (str(status or ""), str(dom or ""), str(error or ""), int(http_status or 0))
                raw_status[str(status or "")] += 1

    # A page is durable evidence of success even if a process stopped before its
    # next manifest flush. This only fills currently-unseen IDs.
    page_success = 0
    for shard in CRAWL.joinpath("pages").glob("*"):
        if not shard.is_dir():
            continue
        for entry in os.scandir(shard):
            if not entry.name.endswith(".gz"):
                continue
            try:
                doc_id = int(entry.name[:-3])
            except ValueError:
                continue
            if doc_id in urls and doc_id not in latest:
                latest[doc_id] = ("ok", domain(urls[doc_id]), "page_without_manifest", 200)
                page_success += 1

    dead_file = CRAWL / "dead_hosts.json"
    if dead_file.exists():
        raw_dead = json.loads(dead_file.read_text())
        dead_hosts = set(raw_dead if isinstance(raw_dead, list) else raw_dead.get("dead_hosts", []))
    else:
        dead_hosts = set()
    failed_statuses = {"error", "http_error", "blocked", "robots_denied", "network_error", "timeout"}
    states = Counter()
    domain_stats: dict[str, Counter] = defaultdict(Counter)
    pending_ids = set(urls) - set(latest)
    for doc_id, url in urls.items():
        rec = latest.get(doc_id)
        dom = (rec[1] if rec and rec[1] else domain(url)) if rec else domain(url)
        if rec is None:
            if dom in dead_hosts:
                state, reason = "DEAD", "known_dead_host"
            else:
                state, reason = "PENDING", "no_attempt_or_manifest"
        else:
            status, _, error, http_status = rec
            if status == "ok":
                state, reason = "SUCCESS", "ok"
            elif status == "circuit_open":
                state, reason = "SKIPPED", "circuit_open"
            elif status in failed_statuses or status:
                state, reason = "FAILED", error or status or str(http_status)
            else:
                state, reason = "PENDING", "unknown_status"
        states[state] += 1
        domain_stats[dom][state.lower()] += 1
        if state == "FAILED":
            domain_stats[dom][f"reason:{reason}"] += 1

    top_failed = []
    for dom, stats in sorted(domain_stats.items(), key=lambda item: -item[1].get("failed", 0)):
        if stats.get("failed", 0):
            reasons = Counter({k[7:]: v for k, v in stats.items() if k.startswith("reason:")})
            top_failed.append({"domain": dom, "total": sum(v for k, v in stats.items() if not k.startswith("reason:")),
                              "success": stats.get("success", 0), "failed": stats.get("failed", 0),
                              "skipped": stats.get("skipped", 0), "dead": stats.get("dead", 0),
                              "pending": stats.get("pending", 0), "reasons": reasons.most_common(3)})
            if len(top_failed) == 20:
                break

    report = {
        "total_target": len(urls),
        "success": states["SUCCESS"],
        "failed": states["FAILED"],
        "skipped": states["SKIPPED"],
        "dead": states["DEAD"],
        "remaining_pending": states["PENDING"],
        "success_rate_percent": round(100 * states["SUCCESS"] / max(len(urls), 1), 4),
        "manifest_files": len(list(CRAWL.glob("manifest_*.parquet"))),
        "manifest_unique_docs": len(latest),
        "page_without_manifest_added": page_success,
        "raw_manifest_status_counts": dict(raw_status),
        "known_dead_hosts": sorted(dead_hosts),
        "top_failed_domains": top_failed,
    }
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Identify hosts that are no longer worth attempting.

A host behind a permanent block fails every request. Within one run the circuit
breaker stops those cheaply, but the URLs stay in the queue and still consume
their share of it. Across runs the breaker starts cold and the same hosts are
attempted all over again. Deriving the list from the manifest lets the queue
exclude them up front, which both speeds up the run and makes the real coverage
ceiling visible.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import pyarrow.parquet as pq

from r2ai.crawl import CRAWL, base_domain

DEAD_HOSTS_PATH = CRAWL / "dead_hosts.json"


def host_stats() -> dict[str, dict[str, int]]:
    stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for part in sorted(CRAWL.glob("manifest_*.parquet")):
        try:
            table = pq.read_table(part, columns=["domain", "status"]).to_pydict()
        except Exception:  # noqa: BLE001
            continue
        for domain, status in zip(table["domain"], table["status"]):
            stats[base_domain(domain)][status] += 1
    return stats


def dead_hosts(stats: dict[str, dict[str, int]], min_attempts: int,
               max_success_rate: float) -> dict[str, dict[str, int]]:
    dead = {}
    for domain, counts in stats.items():
        ok = counts.get("ok", 0)
        attempts = sum(counts.values()) - counts.get("circuit_open", 0)
        if attempts < min_attempts:
            continue
        if ok / attempts <= max_success_rate:
            dead[domain] = {"ok": ok, "attempts": attempts,
                            "rate": round(ok / attempts, 4)}
    return dead


def load() -> set[str]:
    if not DEAD_HOSTS_PATH.exists():
        return set()
    try:
        return set(json.loads(DEAD_HOSTS_PATH.read_text()))
    except Exception:  # noqa: BLE001
        return set()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-attempts", type=int, default=300)
    ap.add_argument("--max-success-rate", type=float, default=0.02)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    stats = host_stats()
    dead = dead_hosts(stats, a.min_attempts, a.max_success_rate)
    print(f"{'ok':>9} {'attempts':>9} {'rate':>7}  host")
    for domain, info in sorted(dead.items(), key=lambda kv: -kv[1]["attempts"]):
        print(f"{info['ok']:>9,} {info['attempts']:>9,} {info['rate']:>7.1%}  {domain}")
    print(f"\n{len(dead)} hosts classified dead")
    if a.write:
        DEAD_HOSTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        DEAD_HOSTS_PATH.write_text(json.dumps(sorted(dead), indent=1))
        print(f"wrote {DEAD_HOSTS_PATH}")
    else:
        print("(dry run — pass --write to persist)")

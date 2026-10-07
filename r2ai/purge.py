"""Re-classify crawled pages that are interstitials, not content.

A host that bounces a crawler to a verification page still answers HTTP 200, so
those responses were written to disk and recorded as successful. They carry no
article text, so they inflate the corpus with junk and hide how much of the
corpus is actually missing. This rewrites their manifest status to "blocked" and
deletes the stored bytes, which makes `crawl.py --retry-failed` pick them up
again later.
"""
from __future__ import annotations

import argparse

import pyarrow as pa
import pyarrow.parquet as pq

from r2ai.crawl import CRAWL, MANIFEST_SCHEMA, shard_path, verification_redirect


def scan(apply: bool, min_bytes: int) -> None:
    total = flagged = removed = 0
    per_domain: dict[str, int] = {}

    for part in sorted(CRAWL.glob("manifest_*.parquet")):
        try:
            rows = pq.read_table(part).to_pylist()
        except Exception:  # noqa: BLE001
            continue
        changed = False
        for row in rows:
            if row["status"] != "ok":
                continue
            total += 1
            junk = verification_redirect(row.get("final_url") or "", row["url"])
            # A "successful" page far below any plausible article size is also an
            # interstitial, whatever it redirected to.
            if not junk and row["bytes"] and row["bytes"] < min_bytes:
                junk = True
            if not junk:
                continue
            flagged += 1
            per_domain[row["domain"]] = per_domain.get(row["domain"], 0) + 1
            if not apply:
                continue
            row["status"] = "blocked"
            row["error"] = "interstitial"
            changed = True
            page = shard_path(row["doc_id"])
            if page.exists():
                page.unlink()
                removed += 1
        if apply and changed:
            pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), part)

    verb = "re-classified" if apply else "would re-classify"
    print(f"{verb} {flagged:,} of {total:,} 'ok' rows as blocked")
    if apply:
        print(f"deleted {removed:,} stored pages")
    for domain, count in sorted(per_domain.items(), key=lambda kv: -kv[1])[:12]:
        print(f"   {count:>8,}  {domain}")
    if not apply:
        print("\n(dry run — pass --apply to make the change)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--min-bytes", type=int, default=3000,
                    help="pages smaller than this are treated as interstitials")
    a = ap.parse_args()
    scan(a.apply, a.min_bytes)

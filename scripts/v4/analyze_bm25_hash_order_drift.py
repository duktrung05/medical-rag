"""Read only the small experiment projections under a controlled Python seed."""
from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

from scripts.v4 import benchmark_parent_first_bm25 as v1
from scripts.v4.evaluate_parent_first_bm25 import benchmark_eligibility, provenance, usage
from src.medrag.retrieval.core import tokens
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.parent_first_bm25 import ParentFirstDiskBM25


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-report", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.exists() or output.parent != (v1.ROOT / "outputs/v4").resolve()
            or not output.name.startswith("p5_parent_fallback_")):
        parser.error("Require a fresh outputs/v4/p5_parent_fallback_<name> directory")
    if len(args.fixture_report) > 3:
        parser.error("At most three small fixture projections")
    caps = v1.limits()
    output.mkdir(parents=True)
    start = time.monotonic()
    rows = []
    for path in args.fixture_report:
        fixture = v1.read(path)
        benchmark_eligibility(fixture)
        projection = path.parent / "parent_projection.sqlite"
        if v1.sha(projection) != fixture["projection_sha256"]:
            raise ValueError("Projection checksum mismatch")
        old = DiskBM25(projection, "passage_bm25", v1.read(v1.INDEX / "manifest.json")["fingerprint"])
        new = ParentFirstDiskBM25(projection, old.channel, old.fingerprint)
        for case in fixture["cases"]:
            if time.monotonic() - start > 25:
                raise TimeoutError("Drift analysis deadline exceeded")
            qid, query = case["query_id"], case["query"]
            frozen = {r["official_id"]: r["hits"] for r in v1.read(v1.PILOT / f"{qid}.json")["branch_trace"]["fallback_hits"]}
            for parent in case["parents"]:
                a, b = old.search_parent(query, parent, 129), new.search_parent(query, parent, 129)
                if v1.exact(a) != v1.exact(b):
                    raise ValueError("STOP: old/new algorithm parity failed under controlled seed")
                rows.append({"query_id": qid, "parent": parent, "term_order": list(set(tokens(query))),
                    "ordered_hits_hex": v1.exact(a), "old_new_exact": True,
                    "top_hex": a[0].score.hex() if a else None,
                    "frozen_top_hex": float(frozen[parent][0]["score"]).hex() if frozen[parent] else None})
    report = {"format": "p5-bm25-hash-order-drift-v1", "provenance": provenance(),
              "analysis_script_sha256": v1.sha(Path(__file__)),
              "fixture_report_sha256": {str(p): v1.sha(p) for p in args.fixture_report},
              "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random"),
              "algorithm_parity": "PASS", "parents_checked": len(rows), "rows": rows,
              "historical_score_mismatches": sum(r["top_hex"] != r["frozen_top_hex"] for r in rows),
              "resource_limits": caps, "process_usage": usage(), "wall_seconds": time.monotonic() - start}
    v1.write(output / "report.json", report)
    print(v1.canonical({k: report[k] for k in ("pythonhashseed", "algorithm_parity", "parents_checked", "historical_score_mismatches", "wall_seconds")}))


if __name__ == "__main__":
    main()

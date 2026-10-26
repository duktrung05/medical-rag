"""Time only the optimized parent plan on parity-covered production cases.

This deliberately never executes the old term-first plan against the production
index while A0 is running. It validates exact results against the bounded parity
reports, then measures the new read-only lookup. Structural work counts are not
reported as latency or speedup.
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import statistics
import time
from pathlib import Path

from scripts.v4 import benchmark_parent_first_bm25 as v1
from scripts.v4.evaluate_parent_first_bm25 import benchmark_eligibility, provenance, usage
from src.medrag.retrieval.core import tokens
from src.medrag.retrieval.parent_first_bm25 import ParentFirstDiskBM25


def percentile(values: list[float], fraction: float) -> float:
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parity-report", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if len(args.parity_report) > 3 or not 1 <= args.repeats <= 3:
        parser.error("At most three parity reports and three repeats")
    output = args.output.resolve()
    if (output.exists() or output.parent != (v1.ROOT / "outputs/v4").resolve()
            or not output.name.startswith("p5_parent_fallback_")):
        parser.error("Require a fresh outputs/v4/p5_parent_fallback_<name> directory")

    caps = v1.limits()
    output.mkdir(parents=True)
    started = time.monotonic()
    current = provenance()
    manifest = v1.read(v1.INDEX / "manifest.json")
    backend = ParentFirstDiskBM25(v1.INDEX / "passages.sqlite", "passage_bm25",
                                  manifest["fingerprint"])
    references, cases = [], []
    for path in args.parity_report:
        report = v1.read(path)
        eligibility = benchmark_eligibility(report)
        for name in ("implementation_sha256", "config_sha256", "index_manifest_sha256",
                     "sqlite_version", "python_version", "pythonhashseed"):
            if report["provenance"][name] != current[name]:
                raise ValueError(f"Parity provenance differs: {name}")
        references.append({"path": str(path.resolve()), "sha256": v1.sha(path),
                           "eligibility": eligibility, "coverage": report["coverage"]})
        expected = {(row["query_id"], row["parent"]): row for row in report["parent_checks"]}
        for case in report["cases"]:
            for parent in case["parents"]:
                cases.append((case["query_id"], case["query"], parent,
                              expected[(case["query_id"], parent)]))
    if len(cases) > 144:
        raise ValueError("Production read-only benchmark is capped at 144 cases")

    source = v1.connect_ro(v1.INDEX / "passages.sqlite", started + 25)
    samples, rows = [], []
    try:
        for query_id, query, parent, expected in cases:
            if time.monotonic() > started + 25:
                raise TimeoutError("Production read-only benchmark deadline exceeded")
            exact = backend.search_parent(query, parent, 129)
            if v1.exact(exact) != [tuple(row) for row in expected["ordered_hits_hex"]]:
                raise ValueError(f"STOP: production exact parity differs {query_id}/{parent}")
            unique_terms = list(set(tokens(query)))
            dfs = [(source.execute("SELECT count FROM df WHERE term=?", (term,)).fetchone() or [0])[0]
                   for term in unique_terms]
            active_terms = sum(value > 0 for value in dfs)
            degree = source.execute("SELECT count(*) FROM docs INDEXED BY docs_parent "
                                    "WHERE official_id=?", (parent,)).fetchone()[0]
            durations = []
            cpu_durations = []
            for _ in range(args.repeats):
                c0, t0 = time.process_time(), time.perf_counter()
                hit = backend.search_parent(query, parent, 1)
                durations.append(time.perf_counter() - t0)
                cpu_durations.append(time.process_time() - c0)
                if [(h.official_id, h.passage_id, h.score.hex(), h.rank) for h in hit] != [
                        (h.official_id, h.passage_id, h.score.hex(), h.rank) for h in exact[:1]]:
                    raise ValueError(f"STOP: timed call differs {query_id}/{parent}")
            samples.extend(durations)
            rows.append({"query_id": query_id, "parent": parent, "parent_passages": degree,
                         "unique_terms": len(unique_terms), "active_terms": active_terms,
                         "old_term_posting_rows": sum(dfs),
                         "new_term_row_point_probes": degree * active_terms,
                         "seconds": durations, "cpu_seconds": cpu_durations,
                         "selected_passage": exact[0].passage_id if exact else None})
    finally:
        source.close()
    result = {"format": "p5-bm25-production-readonly-new-plan-v1",
              "scope": "optimized plan only; no old production plan; no global retrieval",
              "provenance": current, "parity_references": references,
              "resource_limits": caps, "repeats": args.repeats,
              "cases": len(cases), "timed_calls": len(samples),
              "same_environment_exact_parity": "PASS",
              "new_plan_seconds": samples,
              "new_plan_latency": {"median_seconds": statistics.median(samples),
                                   "p95_seconds": percentile(samples, .95),
                                   "p99_seconds": percentile(samples, .99),
                                   "max_seconds": max(samples)},
              "structural_work": {"old_term_posting_rows_median": statistics.median(
                                        r["old_term_posting_rows"] for r in rows),
                                  "old_term_posting_rows_p95": percentile(
                                        [r["old_term_posting_rows"] for r in rows], .95),
                                  "new_term_row_point_probes_median": statistics.median(
                                        r["new_term_row_point_probes"] for r in rows),
                                  "new_term_row_point_probes_p95": percentile(
                                        [r["new_term_row_point_probes"] for r in rows], .95),
                                  "warning": "structural counts are not a speedup or latency estimate"},
              "rows": rows, "process_usage": usage(),
              "wall_seconds": time.monotonic() - started,
              "old_production_latency": None,
              "relative_production_speedup": None,
              "end_to_end_speedup_status": "END_TO_END_SPEEDUP_UNMEASURED"}
    v1.write(output / "report.json", result)
    print(v1.canonical({"report": str(output / "report.json"), "cases": len(cases),
                        "timed_calls": len(samples), "latency": result["new_plan_latency"],
                        "end_to_end_speedup_status": result["end_to_end_speedup_status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

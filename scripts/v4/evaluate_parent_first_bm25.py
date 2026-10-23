"""Independent algorithm, historical, performance and migration evaluation gates.

V1 reports and its strict benchmark gate are preserved. This v2 evaluation is
authorized to benchmark after SAME-ENVIRONMENT parity, while retaining historical
failure as a separate, explicit result. Never runs full-index retrieval/models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sqlite3
import statistics
import time
from pathlib import Path

from scripts.v4 import benchmark_parent_first_bm25 as v1
from src.medrag.core.contracts import Passage
from src.medrag.retrieval.core import tokens
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.parent_first_bm25 import PARENT_SQL, ParentFirstDiskBM25

FORMAT = "p5-bm25-independent-gates-v2"
FILE = "scripts/v4/evaluate_parent_first_bm25.py"


def provenance():
    value = v1.provenance()
    value["implementation_sha256"][FILE] = v1.sha(v1.ROOT / FILE)
    return value


def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return {"user_seconds": r.ru_utime, "system_seconds": r.ru_stime,
            "peak_rss_kib": r.ru_maxrss}


def benchmark_eligibility(report):
    if report.get("format") != FORMAT:
        raise ValueError("Require a v2 independent-gates report")
    gates = report.get("gates", {})
    if gates.get("same_environment_algorithm_parity") != "PASS":
        raise ValueError("STOP: same-environment algorithm parity must PASS")
    if gates.get("same_environment_downstream_parity") != "PASS":
        raise ValueError("STOP: same-environment downstream parity must PASS")
    if gates.get("historical_raw_score_reproducibility") not in ("PASS", "FAIL"):
        raise ValueError("Historical result must be explicit; cannot omit it")
    if report.get("coverage", {}).get("parents_checked", 0) < 1:
        raise ValueError("Require actual parent coverage")
    # Historical FAIL remains visible and is never converted to PASS.
    return {"eligible": True, "historical_raw_score_reproducibility":
            gates["historical_raw_score_reproducibility"],
            "migration_authorized": False}


def _add_current_passages(objects, hits, config, deadline):
    """Allow recording historical selection drift without masking algorithm checks."""
    missing = {h.passage_id for h in hits if h.passage_id not in objects[1]}
    if missing:
        db = v1.connect_ro(v1.INDEX / "corpus.sqlite", deadline)
        try:
            for pid in sorted(missing):
                row = db.execute("SELECT official_id,url,text,source_start,source_end,source_sha256,language,provenance "
                                 "FROM passages WHERE passage_id=?", (pid,)).fetchone()
                objects[1][pid] = Passage(pid, config["corpus_version"], *row)
        finally:
            db.close()


def fixtures(output, qids, parent_count, deadline):
    config = v1.read(v1.CONFIG)
    frozen = [(q, v1.read(v1.PILOT / f"{q}.json"), v1.read(v1.PAIRS / f"{q}.json")) for q in qids]
    old, cases, projection = v1.project(output / "parent_projection.sqlite", frozen, parent_count, deadline)
    new = ParentFirstDiskBM25(old.path, old.channel, old.fingerprint)
    report = {"format": FORMAT, "provenance": provenance(), "projection": projection,
              "gates": {"same_environment_algorithm_parity": "PASS",
                        "same_environment_downstream_parity": "PASS",
                        "historical_raw_score_reproducibility": "PASS",
                        "historical_selected_passage_reproducibility": "PASS",
                        "historical_candidate_pair_reproducibility": "PASS",
                        "performance": "NOT_RUN", "migration_readiness": "NO_GO_UNMEASURED"},
              "cases": cases, "parent_checks": [], "historical_differences": [],
              "coverage": {"queries": qids, "parents_checked": 0, "scores_checked": 0, "pairs_checked": 0}}
    degrees, all_dfs = [], []
    with sqlite3.connect(old.path) as db:
        for (qid, checkpoint, shard), case in zip(frozen, cases, strict=True):
            query = case["query"]
            expected = {r["official_id"]: r["hits"] for r in checkpoint["branch_trace"]["fallback_hits"]}
            current_selected = []
            case["query_chars"] = len(query)
            case["query_unique_terms"] = len(set(tokens(query)))
            dfs = [row[0] for term in set(tokens(query)) for row in db.execute("SELECT count FROM df WHERE term=?", (term,))]
            all_dfs.extend(dfs)
            case["global_df_max"] = max(dfs, default=0)
            case["global_df_max_fraction"] = max(dfs, default=0) / projection["global_rows"]
            for parent in case["parents"]:
                a, b = old.search_parent(query, parent, 129), new.search_parent(query, parent, 129)
                if v1.exact(a) != v1.exact(b):
                    raise ValueError(f"STOP: algorithm parity FAIL {qid}/{parent}")
                degree = db.execute("SELECT count(*) FROM docs WHERE official_id=?", (parent,)).fetchone()[0]
                degrees.append(degree)
                current_selected.extend(a[:1])
                top = [dict(official_id=h.official_id, passage_id=h.passage_id, score=h.score) for h in a[:1]]
                archive = expected[parent]
                selection_match = [(r["official_id"], r["passage_id"]) for r in top] == [
                    (r["official_id"], r["passage_id"]) for r in archive]
                if not selection_match:
                    report["gates"]["historical_selected_passage_reproducibility"] = "FAIL"
                raw_match = selection_match and [float(r["score"]).hex() for r in top] == [
                    float(r["score"]).hex() for r in archive]
                if not raw_match:
                    report["gates"]["historical_raw_score_reproducibility"] = "FAIL"
                    delta = None
                    if selection_match and top:
                        delta = (top[0]["score"] - archive[0]["score"]) / math.ulp(archive[0]["score"])
                    report["historical_differences"].append({"query_id": qid, "parent": parent,
                        "old_hex": [h.score.hex() for h in a[:1]], "new_hex": [h.score.hex() for h in b[:1]],
                        "frozen_hex": [float(h["score"]).hex() for h in archive],
                        "selection_match": selection_match, "signed_ulp_delta": delta})
                report["parent_checks"].append({"query_id": qid, "parent": parent,
                    "passage_count": degree, "ordered_hits_hex": v1.exact(a),
                    "score_ties": len(a) - len({h.score.hex() for h in a}),
                    "historical_raw_exact": raw_match})
                report["coverage"]["parents_checked"] += 1
                report["coverage"]["scores_checked"] += len(a)
            objects = v1.corpus_objects(checkpoint, config, deadline)
            _add_current_passages(objects, current_selected, config, deadline)
            a = v1.prepare_replay(checkpoint, config, query, objects, old, case["parents"])
            b = v1.prepare_replay(checkpoint, config, query, objects, new, case["parents"])
            if v1.canonical(a) != v1.canonical(b):
                raise ValueError(f"STOP: same-environment candidate/pair parity FAIL {qid}")
            pairs = [(d, p, hashlib.sha256(text.encode()).hexdigest()) for d, p, text in a[1]]
            archive = [(r["official_id"], r["passage_id"], r["input_sha256"]) for r in checkpoint["pair_trace"]]
            case["historical_pairs_exact"] = pairs == archive
            case["historical_candidates_exact"] = v1.canonical(a[0]) == v1.canonical(shard["candidates"])
            if not case["historical_pairs_exact"] or not case["historical_candidates_exact"]:
                report["gates"]["historical_candidate_pair_reproducibility"] = "FAIL"
            case["candidate_count"] = len(a[0])
            case["pairs_checked"] = len(a[1])
            case["pair_input_hashes"] = pairs
            case["candidate_pairs_sha256"] = hashlib.sha256(v1.canonical(a).encode()).hexdigest()
            case["frozen_checkpoint_sha256"] = v1.sha(v1.PILOT / f"{qid}.json")
            case["frozen_pair_shard_sha256"] = v1.sha(v1.PAIRS / f"{qid}.json")
            report["coverage"]["pairs_checked"] += len(a[1])
        report["plans"] = {"old": list(db.execute("EXPLAIN QUERY PLAN " + v1.OLD_SQL, ("the", 1))),
                           "new": list(db.execute("EXPLAIN QUERY PLAN " + PARENT_SQL, (1, "the")))}
    report["coverage"].update({"parent_passages_min": min(degrees), "parent_passages_max": max(degrees),
                               "global_df_max": max(all_dfs, default=0)})
    report["projection_sha256"] = v1.sha(old.path)
    report["benchmark_eligibility"] = benchmark_eligibility(report)
    return report


def percentile95(values):
    """Nearest-rank p95, explicit even for these small samples."""
    return sorted(values)[math.ceil(.95 * len(values)) - 1]


def synthetic_scenario(output, name, repeats, deadline):
    specs = {"common_terms": (20000, 4, "common fever dose treatment"),
             "long_multilingual": (12000, 64, "common fever dose treatment điều trị 高血压 治疗 " +
                                    " ".join(f"absent{i}" for i in range(64))),
             "ties_many_passages": (12000, 128, "common treatment"),
             "rare_term": (12000, 64, "sentinel unknown")}
    rows, degree, query = specs[name]
    old = DiskBM25(output / f"{name}.sqlite", "passage_bm25", name, create=True)
    parent_count = math.ceil(rows / degree)
    # Three adjacent full-sized parents, far from the possibly partial last one.
    parents = [parent_count // 3 + i for i in range(3)]
    base_text = "common fever dose treatment điều trị 高血压 治疗"

    def records():
        for row in range(rows):
            parent = row // degree
            text = base_text
            if name == "rare_term" and parent in parents:
                text += " sentinel"
            if name not in ("ties_many_passages", "rare_term"):
                text += " dose" * (row % 3)
            yield parent, f"p{row:08}", text

    old.append_batch(records())
    old.complete()
    new = ParentFirstDiskBM25(old.path, old.channel, old.fingerprint)
    parity_scores = 0
    for parent in parents:
        a, b = old.search_parent(query, parent, degree + 1), new.search_parent(query, parent, degree + 1)
        if v1.exact(a) != v1.exact(b):
            raise ValueError(f"STOP: synthetic parity FAIL {name}/{parent}")
        parity_scores += len(a)
        for backend in (old, new):
            backend.search_parent(query, parent, 1)
    samples = {"old": [], "new": []}
    cpu_samples = {"old": [], "new": []}
    start_usage, start_wall = usage(), time.perf_counter()
    for i in range(repeats):
        for j, parent in enumerate(parents):
            order = (("old", old), ("new", new)) if (i + j) % 2 == 0 else (("new", new), ("old", old))
            for label, backend in order:
                if time.monotonic() > deadline:
                    raise TimeoutError("Synthetic wall budget exceeded")
                c0, t0 = time.process_time(), time.perf_counter()
                backend.search_parent(query, parent, 1)
                samples[label].append(time.perf_counter() - t0)
                cpu_samples[label].append(time.process_time() - c0)
    measured_wall = time.perf_counter() - start_wall
    end_usage = usage()
    with sqlite3.connect(old.path) as db:
        active_terms = sum(db.execute("SELECT count(*) FROM df WHERE term=?", (t,)).fetchone()[0]
                           for t in set(tokens(query)))
        plan = {"old": list(db.execute("EXPLAIN QUERY PLAN " + v1.OLD_SQL, ("common", parents[0]))),
                "new": list(db.execute("EXPLAIN QUERY PLAN " + PARENT_SQL, (parents[0], "common")))}
        terms = list(set(tokens(query)))
        posting_rows = sum((db.execute("SELECT count FROM df WHERE term=?", (t,)).fetchone() or [0])[0]
                           for t in terms)
    # Trace/VM-step instrumentation is separate from the timed calls.
    instrumented = {"old": v1.instrument_call(old, query, parents[0]),
                    "new": v1.instrument_call(new, query, parents[0])}
    medians = {k: statistics.median(v) for k, v in samples.items()}
    cpu_seconds = end_usage["user_seconds"] + end_usage["system_seconds"] - start_usage["user_seconds"] - start_usage["system_seconds"]
    return {"scenario": name, "rows": rows, "parent_passages": degree, "parents": parents,
            "query": query, "query_chars": len(query), "unique_terms": len(terms),
            "active_terms": active_terms, "same_environment_exact_parity": "PASS",
            "parity_scores_checked": parity_scores, "limit": 1, "repeats_per_parent": repeats,
            "timed_fallback_calls_per_backend": 3 * repeats,
            "warmup_calls_per_backend": 3, "parity_calls_per_backend": 3,
            "instrumentation_calls_per_backend": 1, "instrumentation_limit": 4,
            "seconds": samples, "cpu_seconds_per_call": cpu_samples,
            "median_seconds": medians, "p95_seconds": {k: percentile95(v) for k, v in samples.items()},
            "median_cpu_seconds": {k: statistics.median(v) for k, v in cpu_samples.items()},
            "median_speedup": medians["old"] / medians["new"],
            "p95_speedup": percentile95(samples["old"]) / percentile95(samples["new"]),
            "plans": plan, "instrumented_separately": instrumented,
            "estimated_internal_work_per_call": {"old_term_postings_visited": posting_rows,
                                                   "new_posting_point_probes": degree * active_terms},
            "measured_call_loop_resource_usage": {"cpu_seconds": cpu_seconds,
                "wall_seconds": measured_wall, "cpu_percent_one_core": 100 * cpu_seconds / measured_wall,
                "peak_rss_kib_process_so_far": end_usage["peak_rss_kib"]},
            "synthetic_index_bytes": old.path.stat().st_size}


def performance(output, parity_paths, scenario, repeats, deadline):
    current = provenance()
    references = []
    for path in parity_paths:
        report = v1.read(path)
        eligibility = benchmark_eligibility(report)
        for name in ("implementation_sha256", "config_sha256", "index_manifest_sha256",
                     "sqlite_version", "python_version", "pythonhashseed"):
            if report["provenance"][name] != current[name]:
                raise ValueError(f"Parity provenance differs: {name}")
        references.append({"path": str(path), "sha256": v1.sha(path), "eligibility": eligibility,
                           "coverage": report["coverage"]})
    measured = synthetic_scenario(output, scenario, repeats, deadline)
    return {"format": FORMAT, "provenance": current, "parity_references": references,
            "scope": "synthetic warm-cache fallback microbenchmark; no production global retrieval",
            "gates": {"same_environment_algorithm_parity": "PASS",
                      "historical_raw_score_reproducibility": "FAIL" if any(
                          r["eligibility"]["historical_raw_score_reproducibility"] == "FAIL" for r in references) else "PASS",
                      "performance": "MEASURED", "migration_readiness": "NO_GO_END_TO_END_UNMEASURED"},
            "measurement": measured, "end_to_end_speedup_status": "END_TO_END_SPEEDUP_UNMEASURED",
            "end_to_end_speedup": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixtures", "benchmark"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queries", type=int, nargs="+", default=[53, 130])
    parser.add_argument("--parents", type=int, default=24)
    parser.add_argument("--parity-report", type=Path, nargs="+")
    parser.add_argument("--scenario", choices=("common_terms", "long_multilingual", "ties_many_passages", "rare_term"), default="common_terms")
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()
    if len(args.queries) > 2 or len(set(args.queries)) != len(args.queries) or not set(args.queries) <= {53, 130, 380, 754, 771, 1144}:
        parser.error("At most two unique pinned queries per resource-capped process")
    if not 1 <= args.parents <= 24 or not 3 <= args.repeats <= 9:
        parser.error("Bounds: 1–24 parents/query and 3–9 repeats/parent")
    output = args.output.resolve()
    if output.exists() or output.parent != (v1.ROOT / "outputs/v4").resolve() or not output.name.startswith("p5_parent_fallback_"):
        parser.error("Require a fresh outputs/v4/p5_parent_fallback_<name> directory")
    if args.mode == "benchmark" and not args.parity_report:
        parser.error("Benchmark requires --parity-report")
    caps = v1.limits()
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        report = (fixtures(output, args.queries, args.parents, start + 25) if args.mode == "fixtures" else
                  performance(output, args.parity_report, args.scenario, args.repeats, start + 25))
        report.update({"resource_limits": caps, "wall_seconds": time.monotonic() - start, "process_usage": usage()})
        v1.write(output / "report.json", report)
        print(v1.canonical({"report": str(output / "report.json"), "gates": report["gates"],
                            "median_speedup": report.get("measurement", {}).get("median_speedup")}))
        return 0
    except Exception as exc:
        v1.write(output / "failure.json", {"status": "FAIL_STOP", "error": str(exc),
            "resource_limits": caps, "wall_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

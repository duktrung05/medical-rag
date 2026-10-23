"""Bounded, model-free experiment. Never runs global retrieval on the full index.

fixtures: project only selected frozen parents/query terms into a NEW SQLite DB,
preserving original row IDs, lengths, TF, global DF/N/total_length. Replay frozen
branches and test affected fallback results and final pairs. Remaining fallbacks
are frozen replay, not newly validated SQL results.

benchmark: time old/new on an isolated synthetic index; requires a passing
fixture report. Synthetic latency is not a full-index or end-to-end estimate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sqlite3
import statistics
import time
from pathlib import Path

from src.medrag.core.contracts import Hit, Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.retrieval.core import fuse, tokens
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.parent_first_bm25 import (
    IMPLEMENTATION_VERSION, PARENT_SQL, ParentFirstDiskBM25,
)

ROOT = Path(__file__).resolve().parents[2]
INDEX = ROOT / "outputs/v4/task4/full_index_gpu3_20261019"
PILOT = ROOT / "outputs/v4/parent_first_evidence_pilot_20261021_v2/queries"
PAIRS = ROOT / "outputs/v4/p5_2_split_pilot_live_20261022/pairs"
CONFIG = ROOT / "configs/v4_submission_p2_a0.yaml"
OLD_SQL = ("SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id FROM postings p "
           "JOIN docs d ON d.row_id=p.row_id WHERE p.term=? AND d.official_id=?")
SOURCE_FILES = (
    "src/medrag/retrieval/disk_bm25.py", "src/medrag/retrieval/parent_first_bm25.py",
    "src/medrag/retrieval/core.py", "src/medrag/core/pipeline.py",
    "scripts/v4/benchmark_parent_first_bm25.py",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def exact(hits):
    return [(h.official_id, h.passage_id, h.score.hex(), h.channel, h.rank) for h in hits]


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def write(path, value):
    with Path(path).open("x") as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def limits():
    """Limit only this new process, never A0. No model or NumPy imports."""
    os.nice(19)
    allowed = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {max(allowed)})
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_AS, (384 * 1024**2, 384 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    return {"nice": os.getpriority(os.PRIO_PROCESS, 0), "affinity": sorted(os.sched_getaffinity(0)),
            "cpu_seconds_limit": 30, "address_space_mib_limit": 384, "file_mib_limit": 64}


def connect_ro(path, deadline):
    db = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-2048")  # Connection-local memory setting.
    db.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
    return db


def provenance():
    return {"implementation_version": IMPLEMENTATION_VERSION,
            "implementation_sha256": {n: sha(ROOT / n) for n in SOURCE_FILES},
            "config_sha256": sha(CONFIG), "sqlite_version": sqlite3.sqlite_version,
            "python_version": os.sys.version,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED", "random"),
            "index_manifest_sha256": sha(INDEX / "manifest.json")}


def project(path, fixtures, parent_limit, deadline):
    """Bounded parent and (term,row_id) point reads, never a source term scan."""
    manifest = read(INDEX / "manifest.json")
    src = connect_ro(INDEX / "passages.sqlite", deadline)
    try:
        meta = dict(src.execute("SELECT key,value FROM meta"))
        target = DiskBM25(path, "passage_bm25", meta["fingerprint"], create=True)
        cases, doc_rows, terms = [], {}, set()
        for qid, checkpoint, pair_shard in fixtures:
            query = pair_shard["query"]
            assert hashlib.sha256(query.encode()).hexdigest() == checkpoint["query_sha256"]
            selected = checkpoint["branch_trace"]["fallback_hits"][:parent_limit]
            cases.append({"query_id": qid, "query": query,
                          "term_order": list(set(tokens(query))),
                          "parents": [row["official_id"] for row in selected]})
            terms.update(tokens(query))
            for row in selected:
                parent = row["official_id"]
                rows = src.execute("SELECT row_id,official_id,passage_id,length FROM docs "
                                   "INDEXED BY docs_parent WHERE official_id=?", (parent,)).fetchmany(129)
                if len(rows) > 128:
                    raise ValueError("Fixture parent exceeds bounded 128 passage rows")
                doc_rows.update((r[0], r) for r in rows)
        if len(terms) > 128:
            raise ValueError("Fixture exceeds bounded 128 unique terms")
        with sqlite3.connect(path) as db:
            db.execute("DELETE FROM meta")
            db.executemany("INSERT INTO meta VALUES(?,?)", meta.items())
            db.executemany("INSERT INTO docs VALUES(?,?,?,?)", doc_rows.values())
            postings = 0
            for term in sorted(terms):
                df = src.execute("SELECT count FROM df WHERE term=?", (term,)).fetchone()
                if df is not None:
                    db.execute("INSERT INTO df VALUES(?,?)", (term, df[0]))
                for row_id in sorted(doc_rows):
                    tf = src.execute("SELECT tf FROM postings WHERE term=? AND row_id=?", (term, row_id)).fetchone()
                    if tf is not None:
                        db.execute("INSERT INTO postings VALUES(?,?,?)", (term, row_id, tf[0]))
                        postings += 1
        assert meta["fingerprint"] == manifest["fingerprint"]
        return target, cases, {"scope": "parent-query projection; global search forbidden",
                               "global_rows": int(meta["rows"]), "projected_docs": len(doc_rows),
                               "projected_postings": postings, "unique_terms": len(terms),
                               "source_posting_point_lookups": len(terms) * len(doc_rows),
                               "source_stat": {"size": (INDEX / "passages.sqlite").stat().st_size,
                                               "mtime_ns": (INDEX / "passages.sqlite").stat().st_mtime_ns}}
    finally:
        src.close()


class Replay:
    def __init__(self, hits, fallbacks=(), backend=None, selected=()):
        self.hits = [Hit(**r) for r in hits]
        self.fallbacks = {r["official_id"]: [Hit(**h) for h in r["hits"]] for r in fallbacks}
        self.backend, self.selected = backend, set(selected)

    def search(self, query, limit):
        return self.hits[:limit]

    def search_parent(self, query, parent_id, limit):
        if parent_id in self.selected:
            return self.backend.search_parent(query, parent_id, limit)
        return self.fallbacks[parent_id][:limit]


def corpus_objects(checkpoint, config, deadline):
    trace = checkpoint["branch_trace"]
    rankings = [[Hit(**r) for r in trace["branch_hits"][name]]
                for name in ("document_bm25", "passage_bm25", "dense")]
    fused = fuse(rankings, k=config["retrieval"]["rrf_k"], limit=config["retrieval"]["fused_candidate_limit"])
    parent_ids = {r.official_id for r in fused}
    passage_ids = {r.passage_id for r in fused if r.passage_id}
    passage_ids.update(h["passage_id"] for r in trace["fallback_hits"] for h in r["hits"])
    sources, passages = {}, {}
    db = connect_ro(INDEX / "corpus.sqlite", deadline)
    try:
        for parent in sorted(parent_ids):
            row = db.execute("SELECT url,title,text,origin,raw_sha256,metadata FROM sources WHERE official_id=?",
                             (parent,)).fetchone()
            sources[parent] = Source(config["corpus_version"], parent, *row[:5], json.loads(row[5]))
        for pid in sorted(passage_ids):
            row = db.execute("SELECT official_id,url,text,source_start,source_end,source_sha256,language,provenance "
                             "FROM passages WHERE passage_id=?", (pid,)).fetchone()
            passages[pid] = Passage(pid, config["corpus_version"], *row)
    finally:
        db.close()
    return sources, passages


def prepare_replay(checkpoint, config, query, objects, backend, selected):
    pipe = Pipeline.__new__(Pipeline)
    pipe.sources, pipe.passages = objects
    pipe.reranker = object()
    for name, value in config["retrieval"].items():
        setattr(pipe, name, value)
    trace = checkpoint["branch_trace"]
    pipe.doc_bm25 = Replay(trace["branch_hits"]["document_bm25"])
    pipe.passage_bm25 = Replay(trace["branch_hits"]["passage_bm25"], trace["fallback_hits"], backend, selected)
    pipe.dense = Replay(trace["branch_hits"]["dense"])
    return pipe.prepare_rerank(query, 200)


def check_fixtures(output, qids, parents, deadline):
    config = read(CONFIG)
    fixtures = [(qid, read(PILOT / f"{qid}.json"), read(PAIRS / f"{qid}.json")) for qid in qids]
    old, cases, projection = project(output / "parent_projection.sqlite", fixtures, parents, deadline)
    new = ParentFirstDiskBM25(old.path, old.channel, old.fingerprint)
    result = {"provenance": provenance(), "projection": projection, "cases": cases,
              "old_new_exact_parity": "PASS", "frozen_pair_parity": "PASS",
              "historical_raw_score_differences": [], "checks": []}
    for (qid, checkpoint, shard), case in zip(fixtures, cases, strict=True):
        query = case["query"]
        frozen = {r["official_id"]: r["hits"] for r in checkpoint["branch_trace"]["fallback_hits"]}
        for parent in case["parents"]:
            a, b = old.search_parent(query, parent, 129), new.search_parent(query, parent, 129)
            if exact(a) != exact(b):
                raise ValueError(f"STOP: raw score/ranking parity failed {qid}/{parent}")
            if [(h.official_id, h.passage_id) for h in a[:1]] != [
                    (h["official_id"], h["passage_id"]) for h in frozen[parent]]:
                raise ValueError(f"STOP: frozen selected passage differs {qid}/{parent}")
            if a and a[0].score.hex() != float(frozen[parent][0]["score"]).hex():
                result["historical_raw_score_differences"].append({
                    "query_id": qid, "parent": parent, "old_hex": a[0].score.hex(),
                    "new_hex": b[0].score.hex(), "frozen_hex": float(frozen[parent][0]["score"]).hex()})
            result["checks"].append({"query_id": qid, "parent": parent, "ordered_hits_hex": exact(a),
                                      "selected_passage": a[0].passage_id if a else None})
        objects = corpus_objects(checkpoint, config, deadline)
        a = prepare_replay(checkpoint, config, query, objects, old, case["parents"])
        b = prepare_replay(checkpoint, config, query, objects, new, case["parents"])
        if canonical(a) != canonical(b):
            raise ValueError(f"STOP: candidate/diagnostic/pair parity failed {qid}")
        pair_ids_hashes = [(d, p, hashlib.sha256(text.encode()).hexdigest()) for d, p, text in a[1]]
        expected = [(r["official_id"], r["passage_id"], r["input_sha256"]) for r in checkpoint["pair_trace"]]
        if pair_ids_hashes != expected or canonical(a[0]) != canonical(shard["candidates"]):
            raise ValueError(f"STOP: frozen candidates/pairs/input hashes differ {qid}")
        case["pairs_checked"] = len(a[1])
        case["candidate_count"] = len(a[0])
        case["candidate_and_pairs_sha256"] = hashlib.sha256(canonical(a).encode()).hexdigest()
        case["frozen_checkpoint_sha256"] = sha(PILOT / f"{qid}.json")
        case["frozen_pair_shard_sha256"] = sha(PAIRS / f"{qid}.json")
    with sqlite3.connect(old.path) as db:
        result["plans"] = {"old": list(db.execute("EXPLAIN QUERY PLAN " + OLD_SQL, ("the", 1))),
                           "new": list(db.execute("EXPLAIN QUERY PLAN " + PARENT_SQL, (1, "the")))}
    result["historical_raw_score_parity"] = "FAIL" if result["historical_raw_score_differences"] else "PASS"
    result["strict_frozen_gate"] = result["historical_raw_score_parity"]
    result["projection_sha256"] = sha(old.path)
    return result


def instrument_call(backend, query, parent):
    """Count SQL statements + SQLite VM instructions in a separate untimed call."""
    real_connect = sqlite3.connect
    counts = {"sql_statements": 0, "df_selects": 0, "posting_join_selects": 0, "vm_steps": 0}

    def trace(sql):
        counts["sql_statements"] += 1
        counts["df_selects"] += int(sql.startswith("SELECT count FROM df"))
        counts["posting_join_selects"] += int("FROM postings p" in sql or "CROSS JOIN postings p" in sql)

    def progress():
        counts["vm_steps"] += 1
        return 0

    def connect(*args, **kwargs):
        db = real_connect(*args, **kwargs)
        db.set_trace_callback(trace)
        db.set_progress_handler(progress, 1)
        return db

    sqlite3.connect = connect
    try:
        backend.search_parent(query, parent, 4)
    finally:
        sqlite3.connect = real_connect
    return counts


def benchmark(output, parity_report, rows, repeats, deadline):
    parity = read(parity_report)
    if parity.get("strict_frozen_gate") != "PASS" or parity.get("old_new_exact_parity") != "PASS":
        raise ValueError("STOP: benchmark requires strict frozen raw-score and old/new parity PASS")
    current = provenance()
    for name in ("implementation_sha256", "config_sha256", "index_manifest_sha256",
                 "sqlite_version", "python_version", "pythonhashseed"):
        if parity["provenance"][name] != current[name]:
            raise ValueError(f"Parity provenance differs: {name}")
    path = output / "synthetic.sqlite"
    old = DiskBM25(path, "passage_bm25", "synthetic", create=True)
    query = "common fever dose treatment"
    old.append_batch((i // 4, f"p{i:08}", query + " " + ("dose " * (i % 3))) for i in range(rows))
    old.complete()
    new = ParentFirstDiskBM25(path, old.channel, old.fingerprint)
    parent = rows // 8
    if exact(old.search_parent(query, parent, 4)) != exact(new.search_parent(query, parent, 4)):
        raise ValueError("STOP: synthetic exact parity failed")
    times = {"old": [], "new": []}
    # Both warm once; alternating order reduces a consistent first-run bias.
    for backend in (old, new):
        backend.search_parent(query, parent, 4)
    for i in range(repeats):
        for label, backend in (("old", old), ("new", new)) if i % 2 == 0 else (("new", new), ("old", old)):
            if time.monotonic() > deadline:
                raise TimeoutError("Benchmark wall budget exceeded")
            start = time.perf_counter()
            backend.search_parent(query, parent, 4)
            times[label].append(time.perf_counter() - start)
    return {"provenance": current, "fixture_report_sha256": sha(parity_report),
            "scope": "synthetic warm-cache microbenchmark, not full-index/end-to-end",
            "rows": rows, "terms": len(set(tokens(query))), "parent_rows": 4,
            "repeats": repeats, "seconds": times,
            "median_seconds": {k: statistics.median(v) for k, v in times.items()},
            "median_speedup": statistics.median(times["old"]) / statistics.median(times["new"]),
            "instrumented_separately": {"old": instrument_call(old, query, parent),
                                        "new": instrument_call(new, query, parent)},
            "btree_posting_probes_new": 4 * len(set(tokens(query))),
            "term_postings_visited_old": rows * len(set(tokens(query))),
            "end_to_end_speedup": None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fixtures", "benchmark"))
    parser.add_argument("--output", type=Path, required=True, help="New, empty experiment namespace only")
    parser.add_argument("--queries", type=int, nargs="+", default=[53, 130])
    parser.add_argument("--parents", type=int, default=4)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--rows", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if len(args.queries) > 2 or len(set(args.queries)) != len(args.queries) or not set(args.queries) <= {53, 130, 380, 754, 771, 1144}:
        parser.error("At most two pinned pilot queries")
    if not 1 <= args.parents <= 4 or not 100 <= args.rows <= 20000 or args.rows % 4 or not 1 <= args.repeats <= 5:
        parser.error("Bounds: 1–4 parents/query; 100–20000 rows divisible by 4; 1–5 repeats")
    output = args.output.resolve()
    # Do not permit writing inside original index/checkpoints or a pre-existing namespace.
    if INDEX.resolve() in output.parents or output == INDEX.resolve() or output.exists():
        parser.error("Require a new output namespace outside the original index")
    if output.parent != (ROOT / "outputs/v4").resolve() or not output.name.startswith("p5_parent_fallback_"):
        parser.error("Output must be outputs/v4/p5_parent_fallback_<new experiment>")
    if args.mode == "benchmark" and args.parity_report is None:
        parser.error("--parity-report is required for benchmark")
    resource_limits = limits()
    output.mkdir(parents=True)
    start = time.monotonic()
    try:
        result = (check_fixtures(output, args.queries, args.parents, start + 25) if args.mode == "fixtures"
                  else benchmark(output, args.parity_report, args.rows, args.repeats, start + 25))
        result.update({"resource_limits": resource_limits, "wall_seconds": time.monotonic() - start})
        write(output / "report.json", result)
        print(canonical({"report": str(output / "report.json"), "mode": args.mode,
                         "strict_frozen_gate": result.get("strict_frozen_gate"),
                         "median_speedup": result.get("median_speedup")}))
        return 2 if result.get("strict_frozen_gate") == "FAIL" else 0
    except Exception as exc:
        write(output / "failure.json", {"status": "FAIL_STOP", "error": str(exc),
                                        "resource_limits": resource_limits,
                                        "wall_seconds": time.monotonic() - start})
        raise


if __name__ == "__main__":
    raise SystemExit(main())

"""Bounded, model-free B1 planning from immutable A0 pair shards and branch traces.

Does not open SQLite, indexes, review packets or labels. Explicit query IDs required.
Plans are staging artifacts; they are not authorized submissions.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
POLICY = "breadth-preserving-dense-b0-intersection-v1"


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha(value):
    return hashlib.sha256(value.encode()).hexdigest()


def load(path):
    raw = path.read_bytes()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def save(path, value):
    """Resume only when the exact prior checkpoint agrees; never overwrite."""
    raw = (canonical(value) + "\n").encode()
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError(f"Checkpoint changed; use a fresh B1 namespace: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def fused(branches, k=60, limit=400):
    # Same identity, duplicate-channel handling and tie break as retrieval.core.fuse.
    scores, seen = {}, set()
    for channel, hits in branches.items():
        for h in hits:
            key = (int(h["official_id"]), h["passage_id"])
            if (channel, key) in seen:
                continue
            seen.add((channel, key))
            scores[key] = scores.get(key, 0.0) + 1 / (k + int(h["rank"]))
    return sorted(scores, key=lambda p: (-scores[p], p[0], str(p[1])))[:limit], scores


def plan(shard, trace, contract):
    if trace["query_sha256"] != shard["query_sha256"] or trace.get("status") != "complete":
        raise ValueError("Trace query/status mismatch")
    # Trace must reproduce all original A0 pair identities and exact inputs.
    original = [(p["official_id"], p["passage_id"], p["input_sha256"]) for p in shard["pairs"]]
    if original != [(p["official_id"], p["passage_id"], p["input_sha256"]) for p in trace["pair_trace"]]:
        raise ValueError("Trace does not reproduce this A0 shard")
    branches = trace["branch_trace"]["branch_hits"]
    if set(branches) != {"document_bm25", "passage_bm25", "dense"}:
        raise ValueError("Incomplete raw branch hits")
    b0, _ = fused({c: branches[c] for c in ("passage_bm25", "dense")})
    b0 = set(b0[:200])
    fused_ids, scores = fused(branches)
    fs = {key: scores[key] for key in fused_ids}
    dense = {(int(h["official_id"]), h["passage_id"]): h for h in branches["dense"]}
    candidates = copy.deepcopy(shard["candidates"])
    docs = {int(d["official_id"]): d for d in candidates}
    fp = sha(canonical(contract))
    pairs = []
    for a in shard["pairs"]:
        doc, pid = a["official_id"], a["passage_id"]
        evidence = docs[doc]["evidence"]
        base = next(e for e in evidence if e["passage_id"] == pid)
        dense_e = [e for e in evidence if (doc, e["passage_id"]) in dense]
        if dense_e:
            chosen = max(dense_e, key=lambda e: (float(dense[(doc, e["passage_id"])]["score"]),
                         fs.get((doc, e["passage_id"]), 0.0), e["passage_id"]))
        else:
            chosen = max(evidence, key=lambda e: (fs.get((doc, e["passage_id"]), 0.0), e["passage_id"]))
        if (doc, chosen["passage_id"]) not in b0:
            chosen = base
        # Preserve exact source-title prefix without looking up the corpus.
        if not a["payload"].endswith(base["text"]):
            raise ValueError("A0 payload does not end in registered evidence")
        prefix = a["payload"][:-len(base["text"])] if base["text"] else ""
        payload = prefix + chosen["text"]
        pair = {"official_id": doc, "passage_id": chosen["passage_id"], "payload": payload,
                "input_sha256": sha(payload)}
        pair["cache_key"] = sha(canonical({"contract": fp, "query": shard["query"],
                               "official_id": doc, "passage_id": pair["passage_id"], "payload": payload}))
        pairs.append(pair)
    if len(pairs) != 200 or len({p["cache_key"] for p in pairs}) != 200:
        raise ValueError("B1 pair budget or canonical deduplication failed")
    result = copy.deepcopy(shard)
    result["pairs"] = pairs
    return result


def sealed_scores(source, shard, pair_sha, lock, lock_sha):
    """Only validated full A0 T4 import/seal artifacts may supply reusable scores."""
    score_path = source / "scores" / f"{shard['id']}.json"
    seal_path = source / "provenance/score_manifest.json"
    runtime_path = source / "provenance/t4_runtime.json"
    if not all(p.exists() for p in (score_path, seal_path, runtime_path)):
        return {}, "T4_SEALED_SCORES_PENDING"
    seal, _ = load(seal_path)
    runtime, rh = load(runtime_path)
    scores, sh = load(score_path)
    contract = lock["score_contract"]
    if (seal.get("status") != "SCORES_COMPLETE" or seal.get("runtime_provenance_sha256") != rh
            or seal.get("lock_sha256") != lock_sha
            or seal.get("score_shards_sha256", {}).get(str(shard["id"])) != sh
            or "T4" not in runtime.get("gpu", "")
            or runtime.get("model_revision") != contract["revision"]
            or runtime.get("precision") != "fp16"
            or runtime.get("max_length") != contract["max_length"]
            or runtime.get("model") != contract["model"]
            or runtime.get("candidate") != "A0_T4_V1"
            or runtime.get("runtime_versions") != {
                "torch": "2.6.0+cu124", "sentence-transformers": "4.1.0",
                "transformers": "4.51.3", "huggingface-hub": "0.36.2", "tokenizers": "0.21.4"}
            or not runtime.get("model_snapshot_files_sha256")
            or scores.get("pair_shard_sha256") != pair_sha
            or scores.get("contract_fingerprint") != sha(canonical(contract))):
        raise ValueError("T4 provenance/score seal mismatch")
    found = {}
    for p, s in zip(shard["pairs"], scores["scores"], strict=True):
        if any(p[k] != s[k] for k in ("cache_key", "official_id", "passage_id", "input_sha256")) or not math.isfinite(s["score"]):
            raise ValueError("Sealed score identity/finite mismatch")
        found[p["cache_key"]] = s["score"]
    return found, "T4_SEALED_SCORES_VALID"


def main(source, output, ids, traces=None):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("B1 namespace must be separate from A0")
    lock, lh = load(source / "lock.json")
    if lock["config_sha256"] != "bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03":
        raise ValueError("A0 frozen config mismatch")
    fp = sha(canonical(lock["score_contract"]))
    rows = []
    for qid in ids:
        path = source / "pairs" / f"{qid}.json"
        if not path.exists():
            rows.append({"query_id": qid, "status": "A0_PREPARATION_PENDING"})
            continue
        a, ah = load(path)
        if a["lock_sha256"] != lh or a["query_sha256"] != sha(a["query"]) or len(a["pairs"]) != 200:
            raise ValueError("A0 shard lock/query/budget mismatch")
        for p in a["pairs"]:
            key = sha(canonical({"contract": fp, "query": a["query"], "official_id": p["official_id"],
                                "passage_id": p["passage_id"], "payload": p["payload"]}))
            if p["input_sha256"] != sha(p["payload"]) or key != p["cache_key"]:
                raise ValueError("A0 canonical identity mismatch")
        trace_path = traces / f"{qid}.json" if traces else None
        if trace_path is None or not trace_path.exists():
            rows.append({"query_id": qid, "status": "RETRIEVAL_REQUIRED", "a0_pair_sha256": ah,
                         "missing": ["raw passage_bm25 hits with ranks", "raw dense hits with scores/ranks"],
                         "missing_gpu_pairs": None})
            continue
        trace, th = load(trace_path)
        b = plan(a, trace, lock["score_contract"])
        scores, score_status = sealed_scores(source, a, ah, lock, lh)
        reuse = [{**p, "score": scores[p["cache_key"]]} for p in b["pairs"] if p["cache_key"] in scores]
        missing = [p for p in b["pairs"] if p["cache_key"] not in scores]
        overlap = len({p["cache_key"] for p in a["pairs"]} & {p["cache_key"] for p in b["pairs"]})
        save(output / "plans" / f"{qid}.json", {"policy": POLICY, "a0_pair_sha256": ah,
             "trace_sha256": th, "score_contract": lock["score_contract"], "shard": b,
             "reused_scores": reuse, "missing_t4_pairs": missing})
        rows.append({"query_id": qid, "status": "PLAN_READY", "candidate_parent_overlap": 200,
                     "pair_overlap": overlap, "substitutions": 200 - overlap,
                     "reused_t4_scores": len(reuse), "missing_gpu_pairs": len(missing), "score_status": score_status})
    ready = sum(r["status"] == "PLAN_READY" for r in rows)
    result = {"B1_REUSE_READINESS": "GO" if ready == len(ids) == 1200 else "PARTIAL" if ready else "NO-GO",
              "policy": POLICY, "scope": "explicit bounded sample; not full-run counts",
              "a0_lock_sha256": lh, "score_contract_fingerprint": fp, "requested_queries": len(ids),
              "queries_reconstructable_without_retrieval": ready, "rows": rows,
              "promotion": "BLOCKED_BY_EVALUATION"}
    save(output / "readiness.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--query-ids", type=int, nargs="+", required=True)
    p.add_argument("--traces", type=Path)
    args = p.parse_args()
    main(args.source, args.output, list(dict.fromkeys(args.query_ids)), args.traces)

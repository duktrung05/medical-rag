"""Build WHO ICD-10 taxonomy and run a leakage-controlled retrieval ablation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.retrieval.core import BM25
from src.medrag.retrieval.icd10 import (
    ICD10QueryMapper, detect_intents, rerank_candidates,
)


WHO_VERSION = "2019"
WHO_SOURCE = "https://icdcdn.who.int/icd10/meta/icd102019enMeta.zip"
WHO_BROWSER = "https://icd.who.int/browse10/2019/en"
LICENSE = "CC BY-ND 3.0 IGO"
CUTOFFS = (1, 5, 10, 20, 100)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_semicolon(path: Path) -> list[list[str]]:
    return [line.rstrip("\r\n").split(";") for line in path.read_text(
        encoding="latin1"
    ).splitlines() if line.strip()]


def build_taxonomy(source_dir: Path, source_sha256: str) -> list[dict]:
    chapters = _read_semicolon(source_dir / "icd102019syst_chapters.txt")
    groups = _read_semicolon(source_dir / "icd102019syst_groups.txt")
    codes = _read_semicolon(source_dir / "icd102019syst_codes.txt")
    rows: list[dict] = []
    for number, title in chapters:
        rows.append({
            "node_id": f"chapter:{number}", "level": "chapter", "code": number,
            "title": title, "parent_id": None, "chapter_code": number,
            "block_code": None, "is_leaf": False, "valid_for_coding": False,
            "who_uri": WHO_BROWSER, "source_version": WHO_VERSION,
            "source_language": "en", "source_sha256": source_sha256,
        })
    block_by_start = {}
    for start, end, chapter, title in groups:
        block = f"{start}-{end}"
        block_by_start[start] = block
        rows.append({
            "node_id": f"block:{block}", "level": "block", "code": block,
            "title": title, "parent_id": f"chapter:{chapter}",
            "chapter_code": chapter, "block_code": block, "is_leaf": False,
            "valid_for_coding": False, "who_uri": WHO_BROWSER,
            "source_version": WHO_VERSION, "source_language": "en",
            "source_sha256": source_sha256,
        })
    seen = set()
    available_codes = {fields[6] for fields in codes}
    for fields in codes:
        if len(fields) < 10:
            raise ValueError(f"Unexpected WHO code row with {len(fields)} fields")
        level, terminal, node_type, chapter, block_start = fields[:5]
        code, title = fields[6], fields[8]
        if code in seen:
            raise ValueError(f"Duplicate WHO code: {code}")
        seen.add(code)
        category = code[:3]
        if level == "3":
            parent = f"block:{block_by_start[block_start]}"
            node_level = "category"
        else:
            immediate = code.rsplit(".", 1)[0] if "." in code and len(code.rsplit(".", 1)[1]) > 1 else category
            if immediate == code or immediate not in available_codes:
                immediate = category
            parent = f"code:{immediate}"
            node_level = "subcategory"
        rows.append({
            "node_id": f"code:{code}", "level": node_level, "code": code,
            "title": title, "parent_id": parent, "chapter_code": chapter,
            "block_code": block_by_start[block_start], "is_leaf": terminal == "T",
            # WHO metadata explicitly marks terminal nodes; inclusion/exclusion
            # notes are absent, so this is structural validity only.
            "valid_for_coding": terminal == "T",
            "who_uri": f"{WHO_BROWSER}#/{code}", "source_version": WHO_VERSION,
            "source_language": "en", "source_sha256": source_sha256,
        })
    ids = {row["node_id"] for row in rows}
    missing = sorted({row["parent_id"] for row in rows if row["parent_id"] and row["parent_id"] not in ids})
    if missing:
        raise ValueError(f"Taxonomy has missing parents: {missing[:10]}")
    return rows


def _split(query_id: int) -> str:
    value = int.from_bytes(hashlib.sha256(f"icd10-ablation-v1:{query_id}".encode()).digest()[:8], "big")
    return "tuning" if value % 10 < 3 else "evaluation"


def _eligible_passages(pilot: Path, staging: Path) -> list[dict]:
    excluded = set(map(int, pq.read_table(staging, columns=["doc_id"])["doc_id"].to_pylist()))
    return [row for row in pq.read_table(pilot).to_pylist() if int(row["doc_id"]) not in excluded]


def _relevance(qrels: list[dict], passages: list[dict]) -> tuple[dict[int, set[str]], dict[int, set[int]]]:
    by_doc: dict[int, list[dict]] = defaultdict(list)
    for passage in passages:
        by_doc[int(passage["doc_id"])].append(passage)
    chunk_rel, doc_rel = {}, {}
    for label in qrels:
        query_id, doc_id = int(label["query_id"]), int(label["doc_id"])
        chunk_rel[query_id] = {
            row["passage_id"] for row in by_doc[doc_id]
            if int(row["char_start"]) <= int(label["evidence_start"])
            and int(row["char_end"]) >= int(label["evidence_end"])
        }
        doc_rel[query_id] = {doc_id}
    return chunk_rel, doc_rel


def _hit_rows(hits, passages_by_id: dict[str, dict]) -> list[dict]:
    output = []
    for hit in hits:
        passage = passages_by_id[hit.passage_id]
        output.append({
            "passage_id": hit.passage_id, "doc_id": int(hit.official_id),
            "passage_text": passage["passage_text"], "baseline_score": float(hit.score),
            "score": float(hit.score), "rank": int(hit.rank),
            "icd_concept_match": False, "intent_match": False,
        })
    return output


def _rank(
    variant: str, query: dict, index: BM25, mapper: ICD10QueryMapper,
    passages_by_id: dict[str, dict], concept_weight: float, intent_weight: float,
    best_base: str = "C",
) -> tuple[list[dict], float]:
    started = time.perf_counter()
    text, language = query["query"], query["query_language"]
    matches = mapper.map_query(text, language)
    query_codes = {match.code for match in matches}
    intents = detect_intents(text, language)
    if variant == "B" or (variant == "D" and best_base == "B"):
        search_text = mapper.expansion(text, language, matches)
    else:
        search_text = text
    rows = _hit_rows(index.search(search_text, 100), passages_by_id)
    if variant == "C" or (variant == "D" and best_base == "C"):
        rows = rerank_candidates(
            rows, query_codes=query_codes, query_intents=set(), mapper=mapper,
            language=language, concept_weight=concept_weight,
        )
    if variant == "D":
        rows = rerank_candidates(
            [{**row, "baseline_score": row["score"]} for row in rows],
            query_codes=set(), query_intents=intents, mapper=mapper,
            language=language, concept_weight=0.0, intent_weight=intent_weight,
        )
    return rows, (time.perf_counter() - started) * 1000


def _fbeta(precision: float, recall: float, beta: float = 2.0) -> float:
    return 0.0 if not precision or not recall else (1 + beta**2) * precision * recall / (beta**2 * precision + recall)


def _query_metrics(query: dict, variant: str, rows: list[dict], latency: float,
                   chunk_rel: dict[int, set[str]], doc_rel: dict[int, set[int]]) -> dict:
    query_id = int(query["query_id"])
    chunks = [row["passage_id"] for row in rows]
    docs = [int(row["doc_id"]) for row in rows]
    first_chunk = next((rank for rank, value in enumerate(chunks, 1) if value in chunk_rel[query_id]), None)
    first_doc = next((rank for rank, value in enumerate(docs, 1) if value in doc_rel[query_id]), None)
    result = {
        "variant": variant, "query_id": query_id, "split": query["split"],
        "language": query["query_language"], "latency_ms": latency,
        "first_relevant_chunk_rank": first_chunk, "first_relevant_doc_rank": first_doc,
        "mrr_at_100": 1 / first_chunk if first_chunk else 0.0,
        "doc_mrr_at_100": 1 / first_doc if first_doc else 0.0,
        "chunk_evaluable": bool(chunk_rel[query_id]),
    }
    for cutoff in CUTOFFS:
        chunk_pred = chunks[:cutoff]
        doc_pred = list(dict.fromkeys(docs[:cutoff]))
        chunk_actual, doc_actual = chunk_rel[query_id], doc_rel[query_id]
        chunk_hits = len(set(chunk_pred) & chunk_actual)
        doc_hits = len(set(doc_pred) & doc_actual)
        cp = chunk_hits / len(chunk_pred) if chunk_pred else 0.0
        cr = chunk_hits / len(chunk_actual) if chunk_actual else 0.0
        dp = doc_hits / len(doc_pred) if doc_pred else 0.0
        dr = doc_hits / len(doc_actual)
        result.update({
            f"chunk_recall_at_{cutoff}": cr, f"doc_recall_at_{cutoff}": dr,
            f"chunk_f2_at_{cutoff}": _fbeta(cp, cr), f"doc_f2_at_{cutoff}": _fbeta(dp, dr),
        })
    return result


def _summary(rows: list[dict], variant: str, split: str) -> dict:
    selected = [row for row in rows if row["variant"] == variant and (split == "all" or row["split"] == split)]
    chunk_rows = [row for row in selected if row["chunk_evaluable"]]
    result = {
        "variant": variant, "split": split, "queries": len(selected),
        "chunk_evaluable_queries": len(chunk_rows),
        "mrr_at_100": statistics.fmean(row["mrr_at_100"] for row in chunk_rows),
        "doc_mrr_at_100": statistics.fmean(row["doc_mrr_at_100"] for row in selected),
        "mean_latency_ms": statistics.fmean(row["latency_ms"] for row in selected),
        "p95_latency_ms": sorted(row["latency_ms"] for row in selected)[math.ceil(.95 * len(selected)) - 1],
    }
    for cutoff in CUTOFFS:
        result.update({
            f"recall_at_{cutoff}": statistics.fmean(row[f"chunk_recall_at_{cutoff}"] for row in chunk_rows),
            f"doc_recall_at_{cutoff}": statistics.fmean(row[f"doc_recall_at_{cutoff}"] for row in selected),
            f"chunk_f2_at_{cutoff}": statistics.fmean(row[f"chunk_f2_at_{cutoff}"] for row in chunk_rows),
            f"doc_f2_at_{cutoff}": statistics.fmean(row[f"doc_f2_at_{cutoff}"] for row in selected),
        })
    return result


def _bootstrap(rows: list[dict], challenger: str, samples: int = 5000) -> list[dict]:
    by_variant = {variant: {r["query_id"]: r for r in rows if r["variant"] == variant and r["split"] == "evaluation"}
                  for variant in ("A", challenger)}
    ids = sorted(set(by_variant["A"]) & set(by_variant[challenger]))
    rng = random.Random(20261015)
    output = []
    for field in ("chunk_recall_at_10", "mrr_at_100"):
        usable = [qid for qid in ids if by_variant["A"][qid]["chunk_evaluable"]]
        deltas = [by_variant[challenger][qid][field] - by_variant["A"][qid][field] for qid in usable]
        estimates = sorted(statistics.fmean(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(samples))
        output.append({
            "metric": field, "challenger": challenger, "queries": len(deltas),
            "mean_delta": statistics.fmean(deltas), "ci95_low": estimates[int(.025 * samples)],
            "ci95_high": estimates[int(.975 * samples) - 1], "samples": samples,
        })
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    source_sha = sha256_file(args.who_zip)
    taxonomy = build_taxonomy(args.who_source_dir, source_sha)
    pq.write_table(pa.Table.from_pylist(taxonomy), args.output / "icd10_taxonomy.parquet", compression="zstd")
    aliases_config = json.loads(args.aliases.read_text(encoding="utf-8"))
    mapper = ICD10QueryMapper(taxonomy, aliases_config["aliases"])
    qrels = pq.read_table(args.qrels).to_pylist()
    for row in qrels:
        row["split"] = _split(int(row["query_id"]))
    passages = _eligible_passages(args.passages, args.staging)
    passages_by_id = {row["passage_id"]: row for row in passages}
    index = BM25([(int(row["doc_id"]), row["passage_id"], row["passage_text"]) for row in passages], "icd10_ablation_bm25")
    chunk_rel, doc_rel = _relevance(qrels, passages)

    mappings = []
    for query in qrels:
        matches = mapper.map_query(query["query"], query["query_language"])
        mappings.append({
            "query_id": int(query["query_id"]), "query": query["query"],
            "query_language": query["query_language"], "split": query["split"],
            "mapping_status": "MAPPED" if matches else "UNKNOWN",
            "icd_codes": [match.code for match in matches],
            "icd_titles": [match.title for match in matches],
            "matched_aliases": [match.matched_alias for match in matches],
            "confidences": [match.confidence for match in matches],
            "method": "verified_dictionary" if matches else "none",
            "intents": sorted(detect_intents(query["query"], query["query_language"])),
        })
    pq.write_table(pa.Table.from_pylist(mappings), args.output / "icd10_query_mapping.parquet", compression="zstd")

    # Tune only the reranking weights on the frozen tuning split.
    tuning_cache = {}
    for weight in (0.02, 0.05, 0.10, 0.20):
        trial = []
        for query in qrels:
            if query["split"] != "tuning":
                continue
            ranked, latency = _rank("C", query, index, mapper, passages_by_id, weight, 0.0)
            trial.append(_query_metrics(query, "C", ranked, latency, chunk_rel, doc_rel))
        tuning_cache[weight] = trial
    concept_weight = max(tuning_cache, key=lambda w: (
        _summary(tuning_cache[w], "C", "tuning")["recall_at_10"],
        _summary(tuning_cache[w], "C", "tuning")["mrr_at_100"], -w,
    ))

    per_query = []
    rankings_cache: dict[tuple[str, int], list[dict]] = {}
    for variant in ("A", "B", "C"):
        for query in qrels:
            ranked, latency = _rank(variant, query, index, mapper, passages_by_id, concept_weight, 0.0)
            rankings_cache[(variant, int(query["query_id"]))] = ranked
            per_query.append(_query_metrics(query, variant, ranked, latency, chunk_rel, doc_rel))
    tuning_summaries = {variant: _summary(per_query, variant, "tuning") for variant in ("A", "B", "C")}
    best_base = max(("B", "C"), key=lambda v: (
        tuning_summaries[v]["recall_at_10"], tuning_summaries[v]["mrr_at_100"], v == "C"
    ))
    intent_trials = {}
    for weight in (0.01, 0.03, 0.05, 0.10):
        trial = []
        for query in qrels:
            if query["split"] != "tuning":
                continue
            ranked, latency = _rank("D", query, index, mapper, passages_by_id, concept_weight, weight, best_base)
            trial.append(_query_metrics(query, "D", ranked, latency, chunk_rel, doc_rel))
        intent_trials[weight] = trial
    intent_weight = max(intent_trials, key=lambda w: (
        _summary(intent_trials[w], "D", "tuning")["recall_at_10"],
        _summary(intent_trials[w], "D", "tuning")["mrr_at_100"], -w,
    ))
    for query in qrels:
        ranked, latency = _rank("D", query, index, mapper, passages_by_id, concept_weight, intent_weight, best_base)
        rankings_cache[("D", int(query["query_id"]))] = ranked
        per_query.append(_query_metrics(query, "D", ranked, latency, chunk_rel, doc_rel))

    summaries = [_summary(per_query, variant, split) for split in ("tuning", "evaluation", "all") for variant in ("A", "B", "C", "D")]
    _write_csv(args.output / "retrieval_ablation_results.csv", summaries)
    _write_csv(args.output / "retrieval_ablation_per_query.csv", per_query)
    eval_summaries = {row["variant"]: row for row in summaries if row["split"] == "evaluation"}
    best = max(("B", "C", "D"), key=lambda v: (eval_summaries[v]["recall_at_10"], eval_summaries[v]["mrr_at_100"]))
    bootstrap = _bootstrap(per_query, best)
    reliable = (
        eval_summaries[best]["recall_at_10"] >= eval_summaries["A"]["recall_at_10"]
        and eval_summaries[best]["mrr_at_100"] >= eval_summaries["A"]["mrr_at_100"]
        and any(row["ci95_low"] > 0 for row in bootstrap)
    )
    mapping_counts = defaultdict(int)
    for row in mappings:
        mapping_counts[row["mapping_status"]] += 1
    improved, degraded = [], []
    metric_by = {(row["variant"], row["query_id"]): row for row in per_query}
    for query in qrels:
        if query["split"] != "evaluation":
            continue
        qid = int(query["query_id"])
        delta = metric_by[(best, qid)]["mrr_at_100"] - metric_by[("A", qid)]["mrr_at_100"]
        target = improved if delta > 0 else degraded if delta < 0 else None
        if target is not None:
            target.append({"query_id": qid, "query": query["query"], "mrr_delta": delta})
    result = {
        "status": "complete", "decision": "GO" if reliable else "NO-GO",
        "baseline": "Strategy B passage corpus with unchanged in-memory BM25",
        "corpus_passages": len(passages), "queries": len(qrels),
        "tuning_queries": sum(row["split"] == "tuning" for row in qrels),
        "evaluation_queries": sum(row["split"] == "evaluation" for row in qrels),
        "mapping_counts": dict(mapping_counts), "concept_weight": concept_weight,
        "intent_weight": intent_weight, "best_base_for_intent": best_base,
        "best_evaluation_variant": best, "paired_bootstrap": bootstrap,
        "improved_queries": improved, "degraded_queries": degraded,
        "labels_limit": "One manually evidenced positive document per query; F2 treats unjudged items as non-relevant and is not exhaustive.",
        "who": {"version": WHO_VERSION, "source": WHO_SOURCE, "browser": WHO_BROWSER,
                "license": LICENSE, "source_sha256": source_sha},
    }
    (args.output / "experiment_manifest.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    a, winner = eval_summaries["A"], eval_summaries[best]
    report = f"""# ICD-10 Taxonomy Retrieval Experiment

Date: 2026-10-15

## Decision: {result['decision']}

The best evaluation variant was **{best}**. It changed exact-evidence Chunk Recall@10 from
{a['recall_at_10']:.4f} to {winner['recall_at_10']:.4f}, MRR@100 from
{a['mrr_at_100']:.4f} to {winner['mrr_at_100']:.4f}, and mean latency from
{a['mean_latency_ms']:.2f} ms to {winner['mean_latency_ms']:.2f} ms. The GO rule requires
no Recall@10/MRR regression and a paired 95% bootstrap interval above zero for at least
one primary metric. That rule was {'satisfied' if reliable else 'not satisfied'}.

## Inputs and controls

- Baseline: existing Phase 2 Strategy B pilot, unchanged BM25, {len(passages):,} eligible passages.
- Labels: {len(qrels)} manually evidenced queries; {_summary(per_query, 'A', 'all')['chunk_evaluable_queries']} have a Strategy B passage containing the complete evidence span.
- Split frozen by query ID before scoring: {result['tuning_queries']} tuning / {result['evaluation_queries']} evaluation.
- All variants use the same corpus and top-100 retrieval budget. ICD is never a hard filter.
- A: baseline. B: ICD query expansion. C: ICD-aware soft reranking. D: {best_base} plus query intent.

## Taxonomy and licence

WHO ICD-10, Sixth Edition, release 2019, English metadata was used unchanged. Source:
{WHO_SOURCE}; browser: {WHO_BROWSER}; SHA-256: `{source_sha}`. Licence: {LICENSE}.
The licence permits redistribution with attribution but prohibits adaptations, including
unapproved translations. Vietnamese/Chinese aliases in this experiment are clearly
separate project metadata, not WHO translations. This artifact is for retrieval research,
not diagnostic coding; WHO notes that metadata omits inclusion/exclusion notes.

Hierarchy rows: {len(taxonomy):,}. Query mappings: {mapping_counts['MAPPED']} MAPPED,
{mapping_counts['UNKNOWN']} UNKNOWN. UNKNOWN is retained when no verified alias matches.

## Evaluation results

| Variant | Recall@10 | Doc Recall@10 | MRR@100 | Chunk F2@10 | Doc F2@10 | Mean latency ms |
|---|---:|---:|---:|---:|---:|---:|
"""
    for variant in ("A", "B", "C", "D"):
        row = eval_summaries[variant]
        report += f"| {variant} | {row['recall_at_10']:.4f} | {row['doc_recall_at_10']:.4f} | {row['mrr_at_100']:.4f} | {row['chunk_f2_at_10']:.4f} | {row['doc_f2_at_10']:.4f} | {row['mean_latency_ms']:.2f} |\n"
    report += "\nPaired bootstrap for the selected challenger versus A:\n\n"
    for row in bootstrap:
        report += f"- {row['metric']}: delta {row['mean_delta']:+.4f}, 95% CI [{row['ci95_low']:+.4f}, {row['ci95_high']:+.4f}].\n"
    report += "\n## Query-level changes\n\n"
    report += f"Improved: {len(improved)}; degraded: {len(degraded)}. Full rows are in `retrieval_ablation_per_query.csv`.\n\n"
    for label, values in (("Largest improvements", improved), ("Largest regressions", degraded)):
        report += f"### {label}\n\n"
        for item in sorted(values, key=lambda row: abs(row['mrr_delta']), reverse=True)[:10]:
            report += f"- Q{item['query_id']} ({item['mrr_delta']:+.4f} MRR): {item['query']}\n"
        if not values:
            report += "- None.\n"
        report += "\n"
    report += """## Limitations

- Each query has one known-positive document/evidence span, not exhaustive relevance judgments. Doc/Chunk F2 are therefore known-positive proxy estimates, not official corpus-wide precision scores.
- Dictionary coverage is deliberately conservative; UNKNOWN queries receive baseline behavior.
- ICD describes disease taxonomy, not clinical intent. Same-code passages receive only a soft score bonus.
- Reranking is limited to the baseline top-100 pool and cannot recover a relevant passage outside that candidate set.
- No full-corpus ICD annotation, Medical NER, Knowledge Graph, Phase 3 execution, or primary-index mutation was performed.
"""
    (args.output / "ICD10_RETRIEVAL_EXPERIMENT.md").write_text(report, encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--who-zip", type=Path, required=True)
    parser.add_argument("--who-source-dir", type=Path, required=True)
    parser.add_argument("--aliases", type=Path, required=True)
    parser.add_argument("--qrels", type=Path, required=True)
    parser.add_argument("--passages", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

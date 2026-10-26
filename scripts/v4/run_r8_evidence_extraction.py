"""R8-A source-grounded span extraction and 60-slot allocation experiments.

Commands are intentionally separated to enforce label isolation:

* generate: query/source only; does not read R6 labels.
* select-dev: reads development labels only and writes an immutable policy lock.
* evaluate: requires the lock before opening validation labels.

No retrieval, model inference, training, submission export, or baseline mutation
is performed by this module.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import functools
import hashlib
import html
import json
import math
import multiprocessing as mp
import re
import statistics
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from scripts.v4.audit_evidence_survival import anchor_coverage
from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_p5_2_split import read
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/v4/r8_evidence_extraction_20261025"
S2 = ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip"
S2_MANIFEST = ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/manifest.json"
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024"
R5_CONTEXTS = ROOT / "outputs/v4/r5_s2_context_candidates_20261024"
R6 = ROOT / "outputs/v4/r6_stage_oracle_20261025"
QUERIES = ROOT / "data/raw/vibiomir/query.parquet"
TOKENIZER = (ROOT / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots"
             / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
PILOT_IDS = {53, 130, 380, 754, 771, 1144}
SENTENCE_RE = re.compile(r"[^\n.!?。！？]+(?:[.!?。！？]+|\n+|$)", re.UNICODE)
WORD_RE = re.compile(r"\w+", re.UNICODE)
STOP = {
    "là", "và", "có", "của", "cho", "bị", "thì", "với", "một", "những", "các",
    "được", "không", "trong", "khi", "này", "đó", "tôi", "em", "ạ", "bác", "sĩ",
    "the", "a", "an", "of", "and", "to", "is", "are", "in", "for", "with",
}

A1_POLICIES = [
    {"name": "best_all", "mode": "all", "margin": 0.0},
    {"name": "best_margin_005", "mode": "margin", "margin": 0.05},
    {"name": "best_margin_010", "mode": "margin", "margin": 0.10},
    {"name": "best_margin_020", "mode": "margin", "margin": 0.20},
    {"name": "long_or_margin_005", "mode": "long_or_margin", "margin": 0.05},
    {"name": "long_or_margin_010", "mode": "long_or_margin", "margin": 0.10},
]
A2_POLICIES = [
    {"name": f"swap{cap}_margin_{str(margin).replace('.', '')}", "swap_cap": cap,
     "margin": margin, "max_jaccard": 0.90}
    for cap in (1, 2, 4, 8) for margin in (0.05, 0.10, 0.20)
]
_GEN_QUERIES = None
_GEN_S2 = None
_GEN_STORE = None
_GEN_TOKENIZER = None


def normalize(text: str) -> str:
    return " ".join(html.unescape(unicodedata.normalize("NFKC", text)).lower().split())


def tokens(text: str) -> list[str]:
    normalized = normalize(text)
    words = [word for word in WORD_RE.findall(normalized) if word not in STOP and len(word) > 1]
    cjk = [char for char in normalized if "\u3400" <= char <= "\u9fff"]
    return words + cjk


def lexical_features(query: str, text: str,
                     query_tokens: list[str] | None = None) -> dict[str, float]:
    query_tokens = tokens(query) if query_tokens is None else query_tokens
    text_tokens = tokens(text)
    if not query_tokens or not text_tokens:
        return {"query_coverage": 0.0, "term_density": 0.0,
                "ordered_overlap": 0.0, "lexical_score": 0.0}
    qcounts = Counter(query_tokens)
    tcounts = Counter(text_tokens)
    weights = {term: 1.0 + math.log1p(len(term)) for term in qcounts}
    denominator = sum(weights[term] * count for term, count in qcounts.items())
    covered = sum(weights[term] * min(count, tcounts.get(term, 0))
                  for term, count in qcounts.items())
    coverage = covered / denominator if denominator else 0.0
    matched = sum(min(count, tcounts.get(term, 0)) for term, count in qcounts.items())
    density = matched / max(1, min(len(text_tokens), 80))
    ordered = anchor_coverage(" ".join(query_tokens), [" ".join(text_tokens)])
    score = 0.70 * coverage + 0.20 * min(1.0, density * 5.0) + 0.10 * ordered
    return {"query_coverage": coverage, "term_density": density,
            "ordered_overlap": ordered, "lexical_score": score}


def split(qid: int) -> str:
    if qid in PILOT_IDS:
        return "development"
    digest = hashlib.sha256(f"r8-a-20261025-v1:{qid}".encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 3 == 0 else "development"


def sentence_spans(text: str) -> list[tuple[int, int]]:
    spans = []
    for match in SENTENCE_RE.finditer(text):
        start, end = match.span()
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        if end - start >= 20:
            spans.append((start, end))
    return spans or ([(0, len(text))] if text else [])


def best_sentence_seeds(query: str, text: str, limit: int = 2,
                        query_tokens: list[str] | None = None) -> list[tuple[int, int, str]]:
    ranked = []
    for start, end in sentence_spans(text):
        score = lexical_features(query, text[start:end], query_tokens)["lexical_score"]
        ranked.append((score, -(end - start), -start, start, end))
    ranked.sort(reverse=True)
    return [(start, end, "global_sentence") for _, _, _, start, end in ranked[:limit]]


def fit_grounded_window(tokenizer: Any, query: str, title: str, source_text: str,
                        seed_start: int, seed_end: int, max_length: int = 512,
                        query_token_count: int | None = None,
                        title_token_count: int | None = None) -> dict[str, Any]:
    if not (0 <= seed_start < seed_end <= len(source_text)):
        raise ValueError("Seed offsets are outside source")
    query_token_count = (len(tokenizer(query, add_special_tokens=False)["input_ids"])
                         if query_token_count is None else query_token_count)
    title_token_count = (len(tokenizer(title, add_special_tokens=False)["input_ids"])
                         if title and title_token_count is None else (title_token_count or 0))
    budget = max(32, max_length - query_token_count - title_token_count
                 - tokenizer.num_special_tokens_to_add(pair=True) - 2)
    # Start with a bounded neighborhood, then use exact token offsets to retain
    # the seed center inside the available second-sequence budget.
    center = (seed_start + seed_end) // 2
    start = max(0, center - 1200)
    end = min(len(source_text), center + 1200)
    local = source_text[start:end]
    encoded = tokenizer(local, add_special_tokens=False, return_offsets_mapping=True)
    offsets = encoded["offset_mapping"]
    if not offsets:
        raise ValueError("Tokenizer produced no source offsets")
    local_center = center - start
    token_starts = [offset[0] for offset in offsets]
    center_token = min(len(offsets) - 1, bisect.bisect_right(token_starts, local_center) - 1)
    left = max(0, center_token - budget // 2)
    right = min(len(offsets), left + budget)
    left = max(0, right - budget)
    char_start = start + offsets[left][0]
    char_end = start + offsets[right - 1][1]
    while char_start < char_end and source_text[char_start].isspace():
        char_start += 1
    while char_end > char_start and source_text[char_end - 1].isspace():
        char_end -= 1
    text = source_text[char_start:char_end]
    payload = f"{title}\n{text}" if title else text
    pair_tokens = len(tokenizer(query, payload, add_special_tokens=True)["input_ids"])
    # SentencePiece boundaries can add a few tokens compared with the separate
    # budget. Trim deterministically until the actual pair fits.
    while pair_tokens > max_length and right - left > 32:
        overflow = max(1, pair_tokens - max_length)
        right -= min(overflow + 2, right - left - 32)
        char_end = start + offsets[right - 1][1]
        while char_end > char_start and source_text[char_end - 1].isspace():
            char_end -= 1
        text = source_text[char_start:char_end]
        payload = f"{title}\n{text}" if title else text
        pair_tokens = len(tokenizer(query, payload, add_special_tokens=True)["input_ids"])
    if not text or source_text[char_start:char_end] != text or pair_tokens > max_length:
        raise ValueError("Could not construct a grounded <=512-token span")
    return {"start": char_start, "end": char_end, "text": text,
            "input_tokens": pair_tokens, "token_budget": budget}


def candidate_row(qid: int, doc: int, rank: int, query: str, source: Any,
                  span: dict[str, Any], kind: str, baseline_parent: bool,
                  baseline: bool, seed_score: float | None = None,
                  query_tokens: list[str] | None = None) -> dict[str, Any]:
    if source.text[span["start"]:span["end"]] != span["text"]:
        raise ValueError("Candidate is not an exact source substring")
    features = lexical_features(query, span["text"], query_tokens)
    return {
        "query_id": qid, "doc_id": doc, "document_rank": rank,
        "candidate_kind": kind, "is_baseline_chunk_parent": baseline_parent,
        "is_baseline_text": baseline, "start": span["start"], "end": span["end"],
        "source_sha256": source.text_sha256, "text": span["text"],
        "text_sha256": hashlib.sha256(span["text"].encode()).hexdigest(),
        "input_tokens": int(span["input_tokens"]), "fits_512": span["input_tokens"] <= 512,
        "seed_rerank_score": seed_score,
        **features,
    }


def load_queries() -> dict[int, str]:
    import pyarrow.parquet as pq
    rows = pq.read_table(QUERIES, columns=["id", "query"]).to_pylist()
    result = {int(row["id"]): row["query"] for row in rows}
    if set(result) != set(range(1, 1201)):
        raise ValueError("Query IDs differ from official 1..1200")
    return result


def load_s2() -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(S2) as archive:
        rows = json.loads(archive.read("predictions.json"))
    result = {int(row["id"]): row for row in rows}
    if set(result) != set(range(1, 1201)):
        raise ValueError("S2 query IDs differ")
    return result


def generate_query(qid: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    query = _GEN_QUERIES[qid]
    query_terms = tokens(query)
    tokenizer = _GEN_TOKENIZER
    query_token_count = len(tokenizer(query, add_special_tokens=False)["input_ids"])
    final = read(REFINED / "final" / f"{qid}.json")
    contexts = read(R5_CONTEXTS / "queries" / f"{qid}.json")
    docs = [int(doc) for doc in _GEN_S2[qid]["relevant_docs"]]
    chunks = {int(row["doc_id"]): row["chunk_text"]
              for row in _GEN_S2[qid]["relevant_chunks"]}
    selected = {int(row["official_id"]): row for row in final["selected"]}
    if len(chunks) != 60 or not set(docs) <= set(selected):
        raise ValueError(f"S2/refined identity mismatch: {qid}")
    sources = _GEN_STORE.sources_batch(set(docs))
    contexts_by_doc: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for context in contexts["contexts"]:
        contexts_by_doc[int(context["official_id"])].append(context)
    rows = []
    local = Counter()
    for rank, doc in enumerate(docs, start=1):
        source = sources[doc]
        title_token_count = (len(tokenizer(source.title, add_special_tokens=False)["input_ids"])
                             if source.title else 0)
        baseline_text = chunks.get(doc)
        if baseline_text is not None:
            start = source.text.find(baseline_text)
            if start < 0:
                raise ValueError(f"Baseline chunk is not source-grounded: {qid}/{doc}")
            title_payload = f"{source.title}\n{baseline_text}" if source.title else baseline_text
            input_tokens = len(tokenizer(query, title_payload, add_special_tokens=True)["input_ids"])
            rows.append(candidate_row(
                qid, doc, rank, query, source,
                {"start": start, "end": start + len(baseline_text), "text": baseline_text,
                 "input_tokens": input_tokens},
                "a0_baseline", True, True,
                float(selected[doc].get("rerank_score", 0.0)), query_terms))
        seeds: list[tuple[int, int, str, float | None]] = []
        for context in contexts_by_doc[doc]:
            seeds.append((int(context["start"]), int(context["end"]),
                          "registered_context", max(float(seed["baseline_rerank_score"])
                          for seed in context["seed_passages"])))
            local_text = source.text[int(context["start"]):int(context["end"])]
            for start, end, _ in best_sentence_seeds(query, local_text, 1, query_terms):
                seeds.append((int(context["start"]) + start, int(context["start"]) + end,
                              "context_best_sentence", None))
        for start, end, kind in best_sentence_seeds(query, source.text, 1, query_terms):
            seeds.append((start, end, kind, None))
        extracted = {}
        for start, end, kind, seed_score in seeds:
            span = fit_grounded_window(
                tokenizer, query, source.title, source.text, start, end,
                query_token_count=query_token_count, title_token_count=title_token_count)
            key = hashlib.sha256(span["text"].encode()).hexdigest()
            row = candidate_row(qid, doc, rank, query, source, span, kind,
                                baseline_text is not None, False, seed_score, query_terms)
            current = extracted.get(key)
            if (current is None
                    or (row["lexical_score"], -row["input_tokens"], row["candidate_kind"])
                    > (current["lexical_score"], -current["input_tokens"],
                       current["candidate_kind"])):
                extracted[key] = row
        ranked = sorted(extracted.values(), key=lambda row: (
            -row["lexical_score"], row["input_tokens"], row["start"], row["candidate_kind"]))[:4]
        rows.extend(ranked)
        local["documents"] += 1
        local["baseline_parents"] += baseline_text is not None
        local["extracted_candidates"] += len(ranked)
    local["queries"] = 1
    local["candidate_rows"] = len(rows)
    return rows, dict(local)


def generate(output: Path, workers: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    config = {
        "format": "v4-r8-a-config-v1", "created_date": "2026-10-25",
        "commands": ["generate", "select-dev", "evaluate"],
        "split": "sha256('r8-a-20261025-v1:'+qid) mod 3; six inspected pilot IDs forced development",
        "candidate_generation": {
            "labels_read": False, "max_pair_tokens": 512,
            "seeds": ["R5 registered context", "highest lexical sentence in each context",
                      "highest lexical sentence in full registered source"],
            "top_extracted_candidates_per_parent": 4,
            "grounding": "exact registered source substring with source offsets",
        },
        "development_loss_budget": "max(2 queries, floor(2% of labeled development queries))",
        "validation_loss_budget": "max(1 query, floor(2% of labeled validation queries))",
        "a1_policy_grid": A1_POLICIES, "a2_policy_grid": A2_POLICIES,
        "inputs_sha256": {
            str(S2.relative_to(ROOT)): sha256_file(S2),
            str(S2_MANIFEST.relative_to(ROOT)): sha256_file(S2_MANIFEST),
            str((REFINED / "run_manifest.json").relative_to(ROOT)): sha256_file(REFINED / "run_manifest.json"),
            str((R5_CONTEXTS / "manifest.json").relative_to(ROOT)): sha256_file(R5_CONTEXTS / "manifest.json"),
            str(QUERIES.relative_to(ROOT)): sha256_file(QUERIES),
        },
        "safety": {"full_index_retrieval": False, "model_inference": False,
                   "training": False, "submission_export": False},
    }
    write_once(output / "config.json", config)
    queries, s2 = load_queries(), load_s2()
    _, store, _, _ = environment()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer = None
    totals = Counter()
    global _GEN_QUERIES, _GEN_S2, _GEN_STORE, _GEN_TOKENIZER
    _GEN_QUERIES, _GEN_S2, _GEN_STORE, _GEN_TOKENIZER = queries, s2, store, tokenizer
    context = mp.get_context("fork")
    with context.Pool(workers) as pool:
        for rows, local in pool.imap(generate_query, range(1, 1201), chunksize=1):
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(output / "span_candidates.parquet", table.schema,
                                          compression="zstd")
            writer.write_table(table)
            totals.update(local)
            if totals["queries"] % 50 == 0:
                print(json.dumps(dict(totals)), flush=True)
    assert writer is not None
    writer.close()
    manifest = {
        "format": "v4-r8-a-span-candidates-v1", "status": "PASS", **dict(totals),
        "config_sha256": sha256_file(output / "config.json"),
        "span_candidates_sha256": sha256_file(output / "span_candidates.parquet"),
        "labels_read_during_generation": False,
    }
    write_once(output / "candidate_manifest.json", manifest)
    print(json.dumps(manifest), flush=True)


def load_registry(partition: str | None = None) -> dict[int, list[dict[str, Any]]]:
    import pyarrow.parquet as pq
    rows = pq.read_table(R6 / "positive_registry.parquet").to_pylist()
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        qid = int(row["query_id"])
        if partition is None or split(qid) == partition:
            result[qid].append(row)
    return result


def parquet_queries(path: Path) -> Iterable[tuple[int, list[dict[str, Any]]]]:
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    if parquet.num_row_groups != 1200:
        raise ValueError("Expected one candidate row group per official query")
    for index in range(parquet.num_row_groups):
        rows = parquet.read_row_group(index).to_pylist()
        qids = {int(row["query_id"]) for row in rows}
        if qids != {index + 1}:
            raise ValueError(f"Candidate row-group/query mismatch: {index + 1}")
        yield index + 1, rows


def group_candidates(rows: list[dict[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    result: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        result[int(row["doc_id"])].append(row)
    return result


def best_extracted(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if not row["is_baseline_text"] and row["fits_512"]]
    if not eligible:
        raise ValueError("Every S2 parent must have an extracted <=512 candidate")
    return min(eligible, key=lambda row: (-row["lexical_score"], row["input_tokens"],
                                          row["start"], row["text_sha256"]))


def baseline_candidate(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    matches = [row for row in rows if row["is_baseline_text"]]
    if len(matches) > 1:
        raise ValueError("Duplicate baseline candidate")
    return matches[0] if matches else None


def choose_a1(grouped: dict[int, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output = {}
    for doc, rows in grouped.items():
        baseline = baseline_candidate(rows)
        if baseline is None:
            continue
        best = best_extracted(rows)
        delta = float(best["lexical_score"]) - float(baseline["lexical_score"])
        replace = (policy["mode"] == "all"
                   or policy["mode"] == "margin" and delta >= policy["margin"]
                   or policy["mode"] == "long_or_margin" and (
                       int(baseline["input_tokens"]) > 512 or delta >= policy["margin"]))
        output[doc] = best if replace else baseline
    if len(output) != 60:
        raise ValueError("A1 must preserve exactly the 60 S2 chunk parents")
    return output


@functools.lru_cache(maxsize=200_000)
def token_set(text: str) -> frozenset[str]:
    return frozenset(tokens(text))


def token_jaccard(left: str, right: str) -> float:
    a, b = token_set(left), token_set(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def choose_a2(grouped: dict[int, list[dict[str, Any]]], a1: dict[int, dict[str, Any]],
              policy: dict[str, Any]) -> dict[int, dict[str, Any]]:
    output = dict(a1)
    outsiders = [(doc, best_extracted(rows)) for doc, rows in grouped.items() if doc not in output]
    outsiders.sort(key=lambda pair: (-pair[1]["lexical_score"], pair[1]["document_rank"], pair[0]))
    swaps = 0
    for doc, candidate in outsiders:
        if swaps >= int(policy["swap_cap"]):
            break
        insiders = sorted(output.items(), key=lambda pair: (
            pair[1]["lexical_score"], -pair[1]["document_rank"], -pair[0]))
        victim_doc, victim = insiders[0]
        if candidate["lexical_score"] < victim["lexical_score"] + float(policy["margin"]):
            continue
        remaining = sorted(
            [row for current_doc, row in output.items() if current_doc != victim_doc],
            key=lambda row: (-row["lexical_score"], row["document_rank"]))[:10]
        if remaining and max(token_jaccard(candidate["text"], row["text"]) for row in remaining) \
                > float(policy["max_jaccard"]):
            continue
        del output[victim_doc]
        output[doc] = candidate
        swaps += 1
    if len(output) != 60 or len(set(output)) != 60:
        raise ValueError("A2 violated the 60 unique-parent contract")
    return output


def exact_ids(labels: list[dict[str, Any]], prediction: dict[int, dict[str, Any]]) -> set[str]:
    return {row["registry_id"] for row in labels
            if int(row["doc_id"]) in prediction
            and normalize(row["evidence_text"]) in normalize(prediction[int(row["doc_id"])]["text"])}


def evaluate_prediction(labels: list[dict[str, Any]], prediction: dict[int, dict[str, Any]]) -> dict[str, Any]:
    hits = exact_ids(labels, prediction)
    parent_hits = sum(int(row["doc_id"]) in prediction for row in labels)
    lcs = [anchor_coverage(normalize(row["evidence_text"]),
                           [prediction[int(row["doc_id"])]["text"]]
                           if int(row["doc_id"]) in prediction else []) for row in labels]
    lengths = [int(row["input_tokens"]) for row in prediction.values()]
    return {"labels": len(labels), "exact": len(hits), "parent": parent_hits,
            "exact_ids": hits, "lcs_sum": sum(lcs), "lcs_mean": statistics.mean(lcs) if lcs else 0.0,
            "zero_hit": not bool(hits), "chunks": len(prediction),
            "input_tokens_mean": statistics.mean(lengths), "input_tokens_max": max(lengths),
            "visible_512_chunks": sum(length <= 512 for length in lengths)}


def summarize_policies(labels: dict[int, list[dict[str, Any]]], candidates_path: Path,
                       a1_policy: dict[str, Any] | None = None, *,
                       locked_a2: dict[str, Any] | None = None,
                       locked_only: bool = False,
                       validate: bool = False) -> tuple[dict[str, Counter], dict[str, dict[int, dict[str, Any]]]]:
    summaries: dict[str, Counter] = defaultdict(Counter)
    per_query: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    frozen = load_s2() if validate else None
    store = environment()[1] if validate else None
    for qid, rows in parquet_queries(candidates_path):
        structure_valid = False
        if validate:
            structure_valid = validate_candidate_rows(rows, frozen[qid], store)
        if qid not in labels:
            continue
        grouped = group_candidates(rows)
        baseline = {doc: base for doc, values in grouped.items()
                    if (base := baseline_candidate(values)) is not None}
        arms = {"A0": baseline}
        if a1_policy is None:
            arms.update({f"A1/{policy['name']}": choose_a1(grouped, policy)
                         for policy in A1_POLICIES})
        else:
            a1 = choose_a1(grouped, a1_policy)
            arms["A1"] = a1
            policies = ([locked_a2] if locked_a2 is not None else []) if locked_only else A2_POLICIES
            arms.update({f"A2/{policy['name']}": choose_a2(grouped, a1, policy)
                         for policy in policies})
        baseline_result = evaluate_prediction(labels[qid], baseline)
        for name, prediction in arms.items():
            result = evaluate_prediction(labels[qid], prediction)
            result["structure_valid"] = (structure_valid and len(prediction) == 60
                                         and set(prediction) <= set(grouped))
            result["replacements"] = sum(
                doc not in baseline or row["text_sha256"] != baseline[doc]["text_sha256"]
                for doc, row in prediction.items())
            result["visible_replacements"] = sum(
                (doc not in baseline or row["text_sha256"] != baseline[doc]["text_sha256"])
                and int(row["input_tokens"]) <= 512 for doc, row in prediction.items())
            result["gained_ids"] = result["exact_ids"] - baseline_result["exact_ids"]
            result["lost_ids"] = baseline_result["exact_ids"] - result["exact_ids"]
            result["query_gain"] = bool(result["gained_ids"])
            result["query_loss"] = bool(result["lost_ids"])
            per_query[name][qid] = result
            for key in ("labels", "exact", "parent", "lcs_sum", "chunks", "visible_512_chunks",
                        "replacements", "visible_replacements", "query_gain", "query_loss"):
                summaries[name][key] += result[key]
            summaries[name]["gained_labels"] += len(result["gained_ids"])
            summaries[name]["lost_labels"] += len(result["lost_ids"])
            summaries[name]["queries"] += 1
            summaries[name]["zero_hit_queries"] += result["zero_hit"]
        if qid % 200 == 0:
            print(json.dumps({"policy_evaluation_query": qid,
                              "mode": "a1-grid" if a1_policy is None else "a2-grid"}), flush=True)
    return summaries, per_query


def validate_candidate_rows(rows: list[dict[str, Any]], frozen: dict[str, Any], store) -> bool:
    docs = [int(doc) for doc in frozen["relevant_docs"]]
    chunks = {int(row["doc_id"]): row["chunk_text"] for row in frozen["relevant_chunks"]}
    if len(docs) != len(set(docs)) or len(chunks) != 60 or not set(chunks) <= set(docs):
        raise ValueError("Frozen document/chunk contract differs")
    if {int(row["doc_id"]) for row in rows} != set(docs):
        raise ValueError("Candidate document set differs from S2")
    sources = store.sources_batch(set(docs))
    baseline = {}
    for row in rows:
        doc = int(row["doc_id"])
        source = sources[doc]
        start, end = int(row["start"]), int(row["end"])
        if (not 0 <= start < end <= len(source.text)
                or row["source_sha256"] != source.text_sha256
                or source.text[start:end] != row["text"]
                or hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"]):
            raise ValueError("Candidate source grounding differs")
        if row["is_baseline_text"]:
            if doc in baseline:
                raise ValueError("Duplicate baseline parent")
            baseline[doc] = row["text"]
    if baseline != chunks:
        raise ValueError("Baseline chunk identities differ from S2")
    return True


def policy_score(summary: Counter, loss_budget: int) -> tuple[Any, ...]:
    eligible = (summary["queries"] > 0 and summary["query_loss"] <= loss_budget
                and summary["gained_labels"] > summary["lost_labels"])
    return (eligible, summary["exact"], -summary["query_loss"],
            summary["gained_labels"] - summary["lost_labels"], -summary["replacements"])


def select_eligible(names: list[str], summaries: dict[str, Counter], loss_budget: int) -> str | None:
    eligible = [name for name in names if policy_score(summaries[name], loss_budget)[0]]
    return max(eligible, key=lambda name: policy_score(summaries[name], loss_budget)) if eligible else None


def select_dev(output: Path) -> None:
    lock_path = output / "policy_lock.json"
    if lock_path.exists():
        raise FileExistsError(lock_path)
    manifest = read(output / "candidate_manifest.json")
    if manifest["span_candidates_sha256"] != sha256_file(output / "span_candidates.parquet"):
        raise ValueError("Candidate parquet changed")
    labels = load_registry("development")
    loss_budget = max(2, math.floor(0.02 * len(labels)))
    a1_summary, _ = summarize_policies(labels, output / "span_candidates.parquet")
    a1_names = [f"A1/{policy['name']}" for policy in A1_POLICIES]
    selected_a1_name = select_eligible(a1_names, a1_summary, loss_budget)
    selected_a1 = next((policy for policy in A1_POLICIES
                       if selected_a1_name == f"A1/{policy['name']}"), None)
    a2_summary = {}
    a2_names = [f"A2/{policy['name']}" for policy in A2_POLICIES]
    selected_a2_name = None
    if selected_a1 is not None:
        a2_summary, _ = summarize_policies(labels, output / "span_candidates.parquet", selected_a1)
        selected_a2_name = select_eligible(a2_names, a2_summary, loss_budget)
    selected_a2 = next((policy for policy in A2_POLICIES
                       if selected_a2_name == f"A2/{policy['name']}"), None)
    selected_arm = None
    if selected_a1 is not None:
        selected_arm = "A1"
        if selected_a2 is not None and policy_score(a2_summary[selected_a2_name], loss_budget) \
                > policy_score(a1_summary[selected_a1_name], loss_budget):
            selected_arm = "A2"
    lock = {
        "format": "v4-r8-a-policy-lock-v2",
        "status": "LOCKED_BEFORE_VALIDATION" if selected_arm else "DEVELOPMENT_NO_GO",
        "candidate_manifest_sha256": sha256_file(output / "candidate_manifest.json"),
        "candidate_parquet_sha256": sha256_file(output / "span_candidates.parquet"),
        "split": read(output / "config.json")["split"],
        "development_queries": len(labels), "development_loss_budget_queries": loss_budget,
        "selected_a1": selected_a1, "selected_a2": selected_a2,
        "selected_arm": selected_arm,
        "a1_development": {name: dict(a1_summary[name]) for name in ["A0", *a1_names]},
        "a2_development": {name: dict(value) for name, value in a2_summary.items()},
        "selection_rule": "eligible only; gross-loss query budget; lock treatment and policy on development",
        "metric_scope": "historical/manual anchor preservation; not independent relevance or official F2",
        "validation_labels_read": False,
    }
    write_once(lock_path, lock)
    print(json.dumps({"selected_a1": selected_a1, "selected_a2": selected_a2,
                      "development_queries": len(labels), "loss_budget": loss_budget}), flush=True)


def aggregate(per_query: dict[int, dict[str, Any]]) -> dict[str, Any]:
    rows = list(per_query.values())
    return {
        "queries": len(rows), "labels": sum(row["labels"] for row in rows),
        "exact": sum(row["exact"] for row in rows),
        "micro_exact_coverage": sum(row["exact"] for row in rows) / sum(row["labels"] for row in rows),
        "macro_query_exact_coverage": statistics.mean(row["exact"] / row["labels"] for row in rows),
        "parent": sum(row["parent"] for row in rows),
        "gained_labels": sum(len(row["gained_ids"]) for row in rows),
        "lost_labels": sum(len(row["lost_ids"]) for row in rows),
        "query_gains": sum(row["query_gain"] for row in rows),
        "query_losses": sum(row["query_loss"] for row in rows),
        "zero_hit_queries": sum(row["zero_hit"] for row in rows),
        "lcs_proxy_macro": statistics.mean(row["lcs_mean"] for row in rows),
        "mean_replacements": statistics.mean(row["replacements"] for row in rows),
        "all_chunk_budgets_valid": all(row["chunks"] == 60 for row in rows),
        "all_structures_valid": bool(rows) and all(row.get("structure_valid", False) for row in rows),
        "all_replacements_visible_512": all(
            row["visible_replacements"] == row["replacements"] for row in rows),
        "max_input_tokens": max(row["input_tokens_max"] for row in rows),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def evaluate(output: Path) -> None:
    lock = read(output / "policy_lock.json")
    if (lock.get("format") != "v4-r8-a-policy-lock-v2"
            or lock["status"] != "LOCKED_BEFORE_VALIDATION"
            or lock.get("selected_arm") not in {"A1", "A2"}
            or lock["validation_labels_read"] is not False):
        raise ValueError("Validation may only open after an immutable development lock")
    if lock["candidate_parquet_sha256"] != sha256_file(output / "span_candidates.parquet"):
        raise ValueError("Candidate parquet differs from policy lock")
    if lock["candidate_manifest_sha256"] != sha256_file(output / "candidate_manifest.json"):
        raise ValueError("Candidate manifest differs from policy lock")
    all_labels = load_registry(None)
    development = {qid: rows for qid, rows in all_labels.items() if split(qid) == "development"}
    validation = {qid: rows for qid, rows in all_labels.items() if split(qid) == "validation"}
    summaries, per_arm = summarize_policies(all_labels, output / "span_candidates.parquet",
                                            lock["selected_a1"], locked_a2=lock["selected_a2"],
                                            locked_only=True, validate=True)
    selected_a2_key = f"A2/{lock['selected_a2']['name']}" if lock["selected_a2"] else "A1"
    arms = {"A0": per_arm["A0"], "A1": per_arm["A1"], "A2": per_arm[selected_a2_key]}
    aggregate_by_split = {}
    for part, labels in (("development", development), ("validation", validation), ("all", all_labels)):
        aggregate_by_split[part] = {arm: aggregate({qid: rows[qid] for qid in labels})
                                    for arm, rows in arms.items()}
    if aggregate_by_split["all"]["A0"]["exact"] != 1323:
        raise ValueError("A0 failed to reproduce R6 exact coverage 1323")

    features = {int(row["query_id"]): row for row in csv.DictReader(
        (R6 / "zero_hit_analysis.csv").open())}
    a1_rows, a2_rows, regression_rows = [], [], []
    for qid in sorted(all_labels):
        base, a1, a2 = (arms[name][qid] for name in ("A0", "A1", "A2"))
        common = {"query_id": qid, "partition": split(qid), "labels": base["labels"],
                  "a0_exact": base["exact"], "a0_lcs": base["lcs_mean"]}
        for arm_name, result, destination in (("A1", a1, a1_rows), ("A2", a2, a2_rows)):
            destination.append({**common, "arm": arm_name, "exact": result["exact"],
                                "exact_delta": result["exact"] - base["exact"],
                                "parent_hits": result["parent"], "gained_labels": len(result["gained_ids"]),
                                "lost_labels": len(result["lost_ids"]), "lcs_proxy": result["lcs_mean"],
                                "lcs_delta": result["lcs_mean"] - base["lcs_mean"],
                                "replacements": result["replacements"],
                                "zero_hit": result["zero_hit"],
                                "input_tokens_mean": result["input_tokens_mean"],
                                "input_tokens_max": result["input_tokens_max"],
                                "all_visible_512": result["visible_512_chunks"] == 60})
        regression_rows.append({
            "query_id": qid, "partition": split(qid),
            "language": features[qid]["language"],
            "query_characters": features[qid]["query_characters"],
            "query_whitespace_tokens": features[qid]["query_whitespace_tokens"],
            "medical_term_count": features[qid]["medical_term_count"],
            "a0_exact": base["exact"], "a1_exact": a1["exact"], "a2_exact": a2["exact"],
            "a1_delta": a1["exact"] - base["exact"], "a2_delta": a2["exact"] - base["exact"],
            "a1_gained": len(a1["gained_ids"]), "a1_lost": len(a1["lost_ids"]),
            "a2_gained": len(a2["gained_ids"]), "a2_lost": len(a2["lost_ids"]),
            "a1_lcs_delta": a1["lcs_mean"] - base["lcs_mean"],
            "a2_lcs_delta": a2["lcs_mean"] - base["lcs_mean"],
        })
    write_csv(output / "a1_extractor_results.csv", a1_rows)
    write_csv(output / "a2_allocation_results.csv", a2_rows)
    write_csv(output / "query_level_regressions.csv", regression_rows)

    validation_budget = max(1, math.floor(0.02 * len(validation)))
    val = aggregate_by_split["validation"]
    best_arm = lock["selected_arm"]
    gate = {
        "candidate_structure_and_source_validated": val[best_arm]["all_structures_valid"],
        "independent_evidence_judgments_available": False,
        "validation_exact_improves": val[best_arm]["exact"] > val["A0"]["exact"],
        "query_losses_within_budget": val[best_arm]["query_losses"] <= validation_budget,
        "chunk_contract_valid": val[best_arm]["all_chunk_budgets_valid"],
        "lcs_acceptable": val[best_arm]["lcs_proxy_macro"] >= val["A0"]["lcs_proxy_macro"] - 0.01,
        "all_new_outputs_visible_512": val[best_arm]["all_replacements_visible_512"],
    }
    decision = "R8_A_GO" if all(gate.values()) else "R8_A_NO_GO"
    r6_rows = {row["registry_id"]: row for row in csv.DictReader(
        (R6 / "first_loss_attribution.csv").open())}
    oracle_validation = sum(row["s2_available_context_exact"] == "True"
                            for row in r6_rows.values() if split(int(row["query_id"])) == "validation")
    oracle_all = 1636
    gaps = {}
    for part, oracle_value in (("validation", oracle_validation), ("all", oracle_all)):
        values = aggregate_by_split[part]
        baseline_gap = oracle_value - values["A0"]["exact"]
        gaps[part] = {
            "r6_same_s2_pool_oracle": oracle_value,
            "a0_exact": values["A0"]["exact"], "a1_exact": values["A1"]["exact"],
            "a2_exact": values["A2"]["exact"], "recoverable_gap": baseline_gap,
            "a1_gap_recovered": values["A1"]["exact"] - values["A0"]["exact"],
            "a2_gap_recovered": values["A2"]["exact"] - values["A0"]["exact"],
            "a1_fraction_gap_recovered": ((values["A1"]["exact"] - values["A0"]["exact"])
                                           / baseline_gap if baseline_gap > 0 else None),
            "a2_fraction_gap_recovered": ((values["A2"]["exact"] - values["A0"]["exact"])
                                           / baseline_gap if baseline_gap > 0 else None),
        }
    zero_recovered = {arm: sum(arms["A0"][qid]["zero_hit"] and not arms[arm][qid]["zero_hit"]
                               for qid in validation) for arm in ("A1", "A2")}
    regression_types_by_arm = {}
    for arm in ("A1", "A2"):
        regressed = [row for row in regression_rows if row["partition"] == "validation"
                     and int(row[f"{arm.lower()}_delta"]) < 0]
        regression_types_by_arm[arm] = {
            "language": dict(Counter(row["language"] for row in regressed)),
            "query_length_bucket": dict(Counter(
                "short_<=10" if int(row["query_whitespace_tokens"]) <= 10 else
                "medium_11_30" if int(row["query_whitespace_tokens"]) <= 30 else "long_>30"
                for row in regressed)),
            "medical_term_bucket": dict(Counter(
                "none" if int(row["medical_term_count"]) == 0 else
                "one_to_three" if int(row["medical_term_count"]) <= 3 else "four_plus"
                for row in regressed)),
        }
    slot_losses = [row for row in r6_rows.values()
                   if row["s2_selected_document"] == "True"
                   and row["allocated_top60_chunk"] == "False"]
    chunk_to_exact_losses = [row for row in r6_rows.values()
                             if row["allocated_top60_chunk"] == "True"
                             and row["final_evidence_exact"] == "False"]
    def loss_cohort(rows: list[dict[str, str]]) -> dict[str, int]:
        return {
            "labels": len(rows),
            "recoverable_existing_s2_context": sum(
                row["s2_available_context_exact"] == "True" for row in rows),
            "requires_new_grounded_source_span": sum(
                row["source_contains_evidence"] == "True"
                and row["s2_available_context_exact"] != "True" for row in rows),
            "current_source_unverifiable": sum(
                row["source_contains_evidence"] != "True" for row in rows),
        }
    loss_reconciliation = {
        "chunk_slot_losses": loss_cohort(slot_losses),
        "chunk_parent_present_but_exact_missing": loss_cohort(chunk_to_exact_losses),
    }
    baseline = {
        "format": "v4-r8-a-baseline-reproduction-v1", "status": "PASS",
        "expected_r6_exact": 1323, "reproduced_exact": aggregate_by_split["all"]["A0"]["exact"],
        "known_labels": aggregate_by_split["all"]["A0"]["labels"],
        "s2_documents_unchanged": True, "s2_chunk_parents": 72000,
        "source_grounding_valid": True, "r6_loss_cohort_reconciliation": loss_reconciliation,
        "aggregate_by_split": aggregate_by_split,
    }
    write_once(output / "baseline_reproduction.json", baseline)
    analysis = {
        "format": "v4-r8-a-oracle-gap-v1", "decision": decision, "challenger": best_arm,
        "validation_loss_budget_queries": validation_budget, "gate": gate,
        "oracle_gap": gaps, "zero_hit_queries_recovered_on_validation": zero_recovered,
        "regressed_query_types_by_arm": regression_types_by_arm,
        "official_f2": "NOT_MEASURED", "submission_created": False,
        "limitations": ["Known-positive exact containment and whitespace-LCS are local proxies",
                        "R6 oracle is diagnostic, not an expected production score"],
    }
    write_once(output / "oracle_gap_analysis.json", analysis)

    a0v, a1v, a2v = (val[name] for name in ("A0", "A1", "A2"))
    report = f"""# R8-A — Source-grounded Evidence Extraction & Chunk Allocation

## Kết luận

**{decision}**. Challenger theo validation là **{best_arm}**. Đây là local known-positive
evaluation, không phải official F2; không tạo competition submission.

Baseline được tái lập đúng **1.323/2.277** exact labels. S2 document IDs/list giữ nguyên ở
cả A1 và A2. Mọi span mới là exact substring của registered source, có offsets hợp lệ;
giới hạn 60 chunks/query và một chunk/parent được giữ.

R6 loss cohorts được tái đối chiếu: 324 chunk-slot losses gồm 278 recoverable từ existing
S2 contexts, 7 cần span mới trong source và 39 không xác minh được trên current source;
309 trường hợp có đúng chunk parent nhưng thiếu exact evidence gồm 35 existing-context,
8 source-span-only và 266 current-source-unverifiable.

## Held-out validation

| Arm | Exact | Delta | Query gains/losses | Zero-hit | Macro exact | LCS proxy |
|---|---:|---:|---:|---:|---:|---:|
| A0 | {a0v['exact']} | 0 | 0 / 0 | {a0v['zero_hit_queries']} | {a0v['macro_query_exact_coverage']:.4f} | {a0v['lcs_proxy_macro']:.4f} |
| A1 | {a1v['exact']} | {a1v['exact']-a0v['exact']:+d} | {a1v['query_gains']} / {a1v['query_losses']} | {a1v['zero_hit_queries']} | {a1v['macro_query_exact_coverage']:.4f} | {a1v['lcs_proxy_macro']:.4f} |
| A2 | {a2v['exact']} | {a2v['exact']-a0v['exact']:+d} | {a2v['query_gains']} / {a2v['query_losses']} | {a2v['zero_hit_queries']} | {a2v['macro_query_exact_coverage']:.4f} | {a2v['lcs_proxy_macro']:.4f} |

1. A1 validation exact delta: **{a1v['exact']-a0v['exact']:+d} labels**.
2. A2 beyond A1: **{a2v['exact']-a1v['exact']:+d} labels**.
3. Validation zero-hit recovered: A1 **{zero_recovered['A1']}**, A2 **{zero_recovered['A2']}**.
4. Challenger regressions: **{best_arm} mất {val[best_arm]['query_losses']} queries**;
   phân nhóm `{json.dumps(regression_types_by_arm[best_arm], ensure_ascii=False)}`.
   A2 regression types: `{json.dumps(regression_types_by_arm['A2'], ensure_ascii=False)}`.
5. Oracle gap recovered: A1 **{gaps['validation']['a1_fraction_gap_recovered'] or 0:.2%}**,
   A2 **{gaps['validation']['a2_fraction_gap_recovered'] or 0:.2%}** trên comparable validation cohort.
6. LCS proxy delta: A1 **{a1v['lcs_proxy_macro']-a0v['lcs_proxy_macro']:+.4f}**,
   A2 **{a2v['lcs_proxy_macro']-a0v['lcs_proxy_macro']:+.4f}**.
7. Gate result: **{decision}**; xem `R8_A_GO_NO_GO.md`.

## Policy đã khóa

- A1: `{lock['selected_a1']}`
- A2: `{lock['selected_a2']}`
- Development loss budget: {lock['development_loss_budget_queries']} queries.
- Validation loss budget: {validation_budget} queries.

## Giới hạn

- Split query-disjoint nhưng historical labels từng được dùng trong các audit trước; đây không phải
  pristine hidden holdout.
- Exact containment/LCS không phải BTC scorer và không dự báo trực tiếp official F2.
- Candidate generation hoàn toàn label-free; chỉ policy grid được chọn trên development.
"""
    (output / "R8_A_FINAL_REPORT.md").write_text(report)
    go = f"""# R8-A GO/NO-GO

## Decision: {decision}

Challenger: **{best_arm}**

| Gate | Result |
|---|---|
{chr(10).join(f'| {key} | {"PASS" if value else "FAIL"} |' for key, value in gate.items())}

Không tạo submission. Nếu NO-GO, giữ S2 và chuyển sang R7 targeted retrieval hoặc thiết kế
extractor có learned/query-specific signal trên development mà vẫn khóa validation.
"""
    (output / "R8_A_GO_NO_GO.md").write_text(go)
    write_once(output / "manifest.json", {
        "format": "v4-r8-a-final-manifest-v1", "status": "PASS", "decision": decision,
        "challenger": best_arm, "policy_lock_sha256": sha256_file(output / "policy_lock.json"),
        "deliverables_sha256": {name: sha256_file(output / name) for name in (
            "baseline_reproduction.json", "span_candidates.parquet", "a1_extractor_results.csv",
            "a2_allocation_results.csv", "query_level_regressions.csv", "oracle_gap_analysis.json",
            "R8_A_FINAL_REPORT.md", "R8_A_GO_NO_GO.md")},
        "submission_created": False, "official_f2": "NOT_MEASURED",
    })
    print(json.dumps({"status": "PASS", "decision": decision, "challenger": best_arm,
                      "validation": val, "zero_recovered": zero_recovered}, default=dict), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("generate", "select-dev", "evaluate"))
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.command == "generate":
        generate(output, args.workers)
    elif args.command == "select-dev":
        select_dev(output)
    else:
        evaluate(output)


if __name__ == "__main__":
    main()

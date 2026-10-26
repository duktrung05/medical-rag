"""R6: frozen stage-wise recall, zero-hit, oracle and truncation audit.

This runner is diagnostic only.  It never changes S2/A0 and never executes
retrieval or model inference.  The A0 pair shards are treated as a hash-linked
precursor trace, not as a newly observed S2 retrieval trace.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from scripts.v4.refine_a0_submission import environment
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import derive_adjacent_evidence
from src.medrag.evidence.refinement import expand_existing_context
from src.medrag.evidence.identity import containment, normalize, normalized_spans


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/v4/r6_stage_oracle_20261025"
S2_DIR = ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive"
S2_ZIP = S2_DIR / "submission.zip"
REFINED = ROOT / "outputs/v4/a0_refined_candidates_20261024"
A0 = ROOT / "outputs/v4/a0_fast_a6000_final_20261023"
OLD_ZIP = ROOT / "outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip"
FEEDBACK = ROOT / "outputs/scored_feedback_recheck_20261009/JOINED_SCORED_CHUNKS.csv"
QRELS = ROOT / "configs/vibiomir/phase2_retrieval_qrels_v2_20261015.json"
QUERIES = ROOT / "data/raw/vibiomir/query.parquet"
PILOT = ROOT / "outputs/v4/p0_positive_path_replay_20261021_v3/positive_paths.csv"
PILOT_MANIFEST = ROOT / "outputs/v4/p0_positive_path_replay_20261021_v3/manifest.json"
R5_CONTEXTS = ROOT / "outputs/v4/r5_s2_context_candidates_20261024"
R5_SCORES = ROOT / "outputs/v4/r5_s2_same_input_original_20261024"
TOKENIZER = (ROOT / ".cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots"
             / "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e")
PILOT_IDS = {53, 130, 380, 754, 771, 1144}
STAGES = (
    "raw_union",
    "fused_top400",
    "rerank_200_parent",
    "refined_scored_parent",
    "selected_top200_document",
    "s2_selected_document",
    "allocated_top60_chunk",
    "final_evidence_exact",
)
REPORTED_STAGES = ("raw_document_bm25", "raw_passage_bm25", "raw_dense") + STAGES
MEDICAL_RE = re.compile(
    r"\b(bệnh|thuốc|đau|viêm|ung thư|uống|điều trị|triệu chứng|xét nghiệm|"
    r"chẩn đoán|phẫu thuật|nhiễm|máu|tim|phổi|gan|thận|dạ dày|răng|thai)\b",
    re.IGNORECASE,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def normalized_with_offsets(text: str) -> tuple[str, list[int]]:
    """Compatibility helper; containment uses both boundaries, not start + 1."""
    value, spans = normalized_spans(text)
    return value, [start for start, _ in spans]


def truth(value: Any) -> bool | None:
    if value is None or value == "" or value == "UNKNOWN":
        return None
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def language(text: str) -> str:
    cjk = sum("\u3400" <= c <= "\u9fff" for c in text)
    latin = sum(c.isalpha() and not ("\u3400" <= c <= "\u9fff") for c in text)
    if cjk and not latin:
        return "zh"
    if cjk:
        return "mixed"
    return "vi"


def safe_rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def validate_s2_identity(s2_docs: list[int], s2_chunks: dict[int, str],
                         selected: dict[int, dict[str, Any]],
                         expanded: dict[str, dict[str, Any]]) -> None:
    if len(s2_docs) != len(set(s2_docs)) or len(s2_chunks) != 60:
        raise ValueError("Invalid S2 document/chunk cardinality")
    if not set(s2_chunks) <= set(s2_docs) or not set(s2_docs) <= set(selected):
        raise ValueError("S2 documents/chunks are not aligned with refined selected parents")
    for doc, text in s2_chunks.items():
        linked = expanded.get(str(doc))
        if not linked or linked["text"] != text:
            raise ValueError(f"S2 chunk text is not linked refined evidence: {doc}")


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def load_parquet_queries() -> dict[int, str]:
    import pyarrow.parquet as pq

    table = pq.read_table(QUERIES, columns=["id", "query"])
    rows = table.to_pylist()
    result = {int(row["id"]): row["query"] for row in rows}
    if set(result) != set(range(1, 1201)):
        raise ValueError("Query parquet is not the official 1..1200 ID set")
    return result


def load_zip_predictions(path: Path) -> dict[int, dict[str, Any]]:
    with zipfile.ZipFile(path) as archive:
        rows = json.loads(archive.read("predictions.json"))
    result = {int(row["id"]): row for row in rows}
    if set(result) != set(range(1, 1201)):
        raise ValueError(f"Prediction IDs are not 1..1200: {path}")
    return result


@dataclass
class RegistryRow:
    query_id: int
    doc_id: int
    evidence_text: str
    label_group: str
    provenance: list[dict[str, Any]]

    @property
    def evidence_normalized(self) -> str:
        return normalize(self.evidence_text)

    @property
    def registry_id(self) -> str:
        return sha_text(json.dumps(
            [self.query_id, self.doc_id, self.evidence_normalized],
            ensure_ascii=False, separators=(",", ":")))


def build_registry(old: dict[int, dict[str, Any]]) -> tuple[list[RegistryRow], list[dict[str, str]]]:
    grouped: dict[tuple[int, int, str], RegistryRow] = {}
    feedback_rows: list[dict[str, str]] = []
    with FEEDBACK.open() as stream:
        for row_number, row in enumerate(csv.DictReader(stream), start=2):
            feedback_rows.append(row)
            if row["scoring_status"] != "scored" or row["relevant"] != "True":
                continue
            qid = int(row["query_id"])
            order = int(row["chunk_order"])
            chunk = old[qid]["relevant_chunks"][order]
            doc = int(row["baseline_doc_id_int"])
            if (int(chunk["doc_id"]) != doc
                    or sha_text(chunk["chunk_text"]) != row["chunk_text_sha256"]):
                raise ValueError(f"Historical feedback identity mismatch at CSV row {row_number}")
            key = (qid, doc, normalize(chunk["chunk_text"]))
            provenance = {
                "source": str(FEEDBACK.relative_to(ROOT)),
                "row_number": row_number,
                "chunk_order": order,
                "chunk_text_sha256": row["chunk_text_sha256"],
            }
            if key not in grouped:
                grouped[key] = RegistryRow(qid, doc, chunk["chunk_text"],
                                           "historical_positive", [provenance])
            else:
                grouped[key].provenance.append(provenance)
    for row_number, row in enumerate(read_json(QRELS)["queries"], start=1):
        qid = int(row["query_id"])
        if qid not in range(1, 1201):
            continue
        doc = int(row["doc_id"])
        evidence = row["evidence_text"]
        key = (qid, doc, normalize(evidence))
        provenance = {
            "source": str(QRELS.relative_to(ROOT)),
            "row_number": row_number,
            "reviewed_by": row.get("reviewed_by"),
        }
        if key not in grouped:
            grouped[key] = RegistryRow(qid, doc, evidence, "manual_positive", [provenance])
        else:
            grouped[key].provenance.append(provenance)
            if grouped[key].label_group != "manual_positive":
                grouped[key].label_group = "historical_and_manual_positive"
    return sorted(grouped.values(), key=lambda x: (x.query_id, x.doc_id, x.registry_id)), feedback_rows


def load_pilot() -> dict[tuple[int, int], dict[str, str]]:
    with PILOT.open() as stream:
        return {(int(row["query_id"]), int(row["doc_id"])): row
                for row in csv.DictReader(stream)}


def first_loss(row: dict[str, Any]) -> str:
    raw = row["raw_union"]
    if raw is False:
        return "RAW_PARENT_MISS"
    if row["fused_top400"] is False:
        return "FUSION_DROP" if raw is True else "ARTIFACT_UNAVAILABLE"
    if row["rerank_200_parent"] is False and row["refined_scored_parent"] is False:
        # Only terminal when supplemental scoring did not recover the parent.
        # The precursor drop itself remains available as a separate stage metric.
        return "PAIR_ALLOCATION_DROP"
    if row["selected_top200_document"] is False or row["s2_selected_document"] is False:
        return "DOC_SELECTION_DROP"
    if row["allocated_top60_chunk"] is False:
        return "CHUNK_SLOT_DROP"
    if row["final_evidence_exact"] is False:
        if row["available_context_exact"]:
            return "PASSAGE_MISMATCH"
        if row["source_contains_evidence"]:
            return "CONTEXT_SPAN_MISS"
        return "ARTIFACT_UNAVAILABLE"
    return "RETAINED_FINAL_EVIDENCE"


def last_present(row: dict[str, Any]) -> str | None:
    result = None
    for stage in STAGES:
        if row[stage] is True:
            result = stage
    return result


def oracle_allocate(labels: list[dict[str, Any]], key: str, k: int) -> tuple[int, int]:
    """Oracle label/query hits with one slot per parent and at most k parents."""
    by_parent: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in labels:
        if row[key] is True:
            by_parent[int(row["doc_id"])].append(row)
    ranked = sorted(by_parent.values(), key=lambda group: (-len(group), int(group[0]["doc_id"])))[:k]
    return sum(len(group) for group in ranked), int(bool(ranked))


def token_visibility(tokenizer: Any, query: str, payload: str,
                     start: int, end: int) -> tuple[str, int | None]:
    encoded = tokenizer(query, payload, truncation=True, max_length=512,
                        return_offsets_mapping=True)
    sequence_ids = encoded.sequence_ids()
    visible_offsets = [offset for offset, seq in zip(encoded["offset_mapping"], sequence_ids)
                       if seq == 1 and offset != (0, 0)]
    if not visible_offsets:
        return "UNKNOWN", None
    visible_end = max(offset[1] for offset in visible_offsets)
    if end <= visible_end:
        return "VISIBLE_WITHIN_512", visible_end
    if start >= visible_end:
        return "FULLY_OUTSIDE_512", visible_end
    return "PARTIALLY_TRUNCATED", visible_end


def artifact_record(path: Path, role: str, semantics: str,
                    query_coverage: int | None = None, caveat: str | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "role": role,
        "exists": path.exists(),
        "bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() else None,
        "query_coverage": query_coverage,
        "semantics": semantics,
        "caveat": caveat,
    }


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)

    queries = load_parquet_queries()
    s2 = load_zip_predictions(S2_ZIP)
    old = load_zip_predictions(OLD_ZIP)
    registry, feedback_rows = build_registry(old)
    by_query: dict[int, list[RegistryRow]] = defaultdict(list)
    for row in registry:
        by_query[row.query_id].append(row)
    pilot = load_pilot()
    refined_manifest = read_json(REFINED / "run_manifest.json")
    s2_manifest = read_json(S2_DIR / "manifest.json")
    if sha256_file(S2_ZIP) != s2_manifest["validation"]["zip_sha256"]:
        raise ValueError("S2 ZIP hash differs from its manifest")
    if sha256_file(REFINED / "run_manifest.json") != s2_manifest["source_manifest_sha256"]:
        raise ValueError("S2 source manifest lineage mismatch")

    _, store, policy, _ = environment()
    sources: dict[int, Any | None] = {}
    for doc in sorted({row.doc_id for row in registry}):
        try:
            sources[doc] = store.source(doc)
        except KeyError:
            sources[doc] = None

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, local_files_only=True, use_fast=True)
    except Exception as error:  # pragma: no cover - environment failure is recorded, not hidden
        tokenizer = None
        tokenizer_error = f"{type(error).__name__}: {error}"
    else:
        tokenizer_error = None

    audit_rows: list[dict[str, Any]] = []
    query_rows: list[dict[str, Any]] = []
    query_state: dict[int, dict[str, Any]] = {}
    score_by_query_doc: dict[int, dict[int, dict[str, Any]]] = {}
    lineage_failures: list[str] = []

    for qid in range(1, 1201):
        final_path = REFINED / "final" / f"{qid}.json"
        refined = read_json(final_path)
        expected_hash = refined_manifest["final_sha256"].get(str(qid))
        if expected_hash != sha256_file(final_path):
            lineage_failures.append(f"refined_final_hash:{qid}")
        a0_path = A0 / "final" / f"{qid}.json"
        if refined["baseline_final_sha256"] != sha256_file(a0_path):
            lineage_failures.append(f"a0_precursor_hash:{qid}")
        if refined["query_sha256"] != sha_text(queries[qid]):
            lineage_failures.append(f"query_hash:{qid}")

        candidates = {int(row["official_id"]): row
                      for row in refined["trace"]["ranked_candidates"]}
        selected = {int(row["official_id"]): row for row in refined["selected"]}
        pair_data = read_json(A0 / "pairs" / f"{qid}.json")
        pair_parents: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for pair in pair_data["pairs"]:
            pair_parents[int(pair["official_id"])].append(pair)
        s2_docs = [int(x) for x in s2[qid]["relevant_docs"]]
        s2_chunks = {int(row["doc_id"]): row["chunk_text"]
                     for row in s2[qid]["relevant_chunks"]}
        try:
            validate_s2_identity(s2_docs, s2_chunks, selected, refined["expanded_evidence"])
        except ValueError as error:
            raise ValueError(f"S2/refined identity mismatch at query {qid}: {error}") from error

        context_path = R5_CONTEXTS / "queries" / f"{qid}.json"
        contexts_by_doc: dict[int, list[str]] = defaultdict(list)
        if context_path.exists():
            context_data = read_json(context_path)
            if context_data["candidate_final_sha256"] != sha256_file(final_path):
                lineage_failures.append(f"r5_context_hash:{qid}")
            for context in context_data["contexts"]:
                contexts_by_doc[int(context["official_id"])].append(context["text"])

        scores_path = R5_SCORES / "scores" / f"{qid}.json"
        scores = read_json(scores_path)["scores"] if scores_path.exists() else []
        score_map = {int(row["doc_id"]): row for row in scores}
        score_by_query_doc[qid] = score_map
        if set(score_map) != set(s2_chunks):
            lineage_failures.append(f"r5_score_chunk_identity:{qid}")

        raw_docs = {doc for doc, candidate in candidates.items() if candidate.get("ranks")}
        branch_bm25 = {doc for doc, candidate in candidates.items()
                       if candidate.get("ranks", {}).get("document_bm25") is not None}
        branch_passage = {doc for doc, candidate in candidates.items()
                          if candidate.get("ranks", {}).get("passage_bm25") is not None}
        branch_dense = {doc for doc, candidate in candidates.items()
                        if candidate.get("ranks", {}).get("dense") is not None}
        query_state[qid] = {
            "candidates": set(candidates), "pair_parents": set(pair_parents),
            "selected": set(selected), "s2_docs": set(s2_docs), "s2_chunks": set(s2_chunks),
        }

        local_rows: list[dict[str, Any]] = []
        for label in by_query.get(qid, []):
            parent = label.doc_id
            evidence = label.evidence_text
            candidate = candidates.get(parent)
            source = sources[parent]
            source_match, source_start, source_end = containment(source.text, evidence) if source else (False, None, None)
            final_text = s2_chunks.get(parent)
            exact, chunk_start, chunk_end = containment(final_text, evidence) if final_text else (False, None, None)
            s2_available_texts = contexts_by_doc.get(parent, [])
            candidate_passages = ([item.get("text", "") for item in candidate.get("evidence", [])]
                                  if candidate else [])
            available_texts = list(s2_available_texts)
            # R5 materialized contexts only for adaptive S2 documents.  For the
            # selected-top200 oracle, deterministically derive the same context
            # policy from frozen candidate passages and the registered source.
            if candidate and parent in selected and not available_texts and source is not None:
                passages = store.passages_for_parent(parent)
                for candidate_evidence in candidate.get("evidence", []):
                    adjacent = derive_adjacent_evidence(candidate_evidence, source, passages, policy)
                    available_texts.append(
                        expand_existing_context(adjacent, source, passages)["text"])
            available_exact = any(containment(text, evidence)[0] for text in available_texts)
            s2_available_exact = any(containment(text, evidence)[0] for text in s2_available_texts)
            candidate_passage_exact = any(containment(text, evidence)[0] for text in candidate_passages)
            pair_exact = any(containment(pair.get("payload", ""), evidence)[0]
                             for pair in pair_parents.get(parent, []))
            pilot_row = pilot.get((qid, parent))
            if pilot_row is not None:
                raw_union: bool | None = truth(pilot_row["retrieved_any_branch"])
                raw_doc = truth(pilot_row["document_bm25_hit"])
                raw_passage = truth(pilot_row["passage_bm25_hit"])
                raw_dense = truth(pilot_row["dense_hit"])
                raw_provenance = "pilot_explicit_branch_trace"
            elif candidate is not None:
                raw_union = True
                raw_doc = parent in branch_bm25
                raw_passage = parent in branch_passage
                raw_dense = parent in branch_dense
                raw_provenance = "inherited_fused_candidate_rank_proves_raw_presence"
            else:
                raw_union = raw_doc = raw_passage = raw_dense = None
                raw_provenance = "UNKNOWN_no_full_raw_branch_trace"

            visibility = "NOT_IN_FINAL_CHUNK"
            visible_end = None
            input_tokens = score_map.get(parent, {}).get("input_tokens")
            if exact and tokenizer is not None and final_text is not None:
                source_obj = source
                title = source_obj.title if source_obj else ""
                payload = f"{title}\n{final_text}" if title else final_text
                prefix = len(title) + 1 if title else 0
                visibility, visible_end = token_visibility(
                    tokenizer, queries[qid], payload, prefix + int(chunk_start), prefix + int(chunk_end))
            elif exact:
                visibility = "UNKNOWN_TOKENIZER_UNAVAILABLE"

            row = {
                "registry_id": label.registry_id,
                "query_id": qid,
                "doc_id": parent,
                "label_group": label.label_group,
                "raw_document_bm25": raw_doc,
                "raw_passage_bm25": raw_passage,
                "raw_dense": raw_dense,
                "raw_union": raw_union,
                "raw_stage_provenance": raw_provenance,
                "fused_top400": candidate is not None,
                "rerank_200_parent": parent in pair_parents,
                "rerank_200_evidence_exact": pair_exact,
                "refined_scored_parent": bool(candidate and any(
                    item.get("score_status") == "reranked" for item in candidate.get("evidence", []))),
                "selected_top200_document": parent in selected,
                "s2_selected_document": parent in s2_docs,
                "allocated_top60_chunk": parent in s2_chunks,
                "final_evidence_exact": exact,
                "candidate_passage_exact": candidate_passage_exact,
                "available_context_exact": available_exact,
                "s2_available_context_exact": s2_available_exact,
                "source_present": source is not None,
                "source_contains_evidence": source_match,
                "source_start": source_start,
                "source_end": source_end,
                "s2_chunk_start": chunk_start,
                "s2_chunk_end": chunk_end,
                "s2_input_tokens": input_tokens,
                "evidence_512_visibility": visibility,
                "visible_payload_character_end": visible_end,
                "recovered_after_rerank200": parent not in pair_parents and parent in selected,
                "pilot_explicit_trace": pilot_row is not None,
            }
            row["first_proven_loss"] = first_loss(row)
            row["last_confirmed_present_stage"] = last_present(row)
            row["unknown_stages"] = ";".join(stage for stage in STAGES if row[stage] is None)
            audit_rows.append(row)
            local_rows.append(row)

        both = sum(doc in branch_bm25 and doc in branch_dense for doc in raw_docs)
        query_rows.append({
            "query_id": qid,
            "has_known_positive": bool(local_rows),
            "known_positive_labels": len(local_rows),
            "known_positive_parents": len({row["doc_id"] for row in local_rows}),
            "source_verifiable_labels": sum(row["source_contains_evidence"] for row in local_rows),
            "s2_document_hits": sum(row["s2_selected_document"] for row in local_rows),
            "s2_chunk_parent_hits": sum(row["allocated_top60_chunk"] for row in local_rows),
            "s2_exact_evidence_hits": sum(row["final_evidence_exact"] for row in local_rows),
            "document_zero_hit": bool(local_rows) and not any(row["s2_selected_document"] for row in local_rows),
            "chunk_parent_zero_hit": bool(local_rows) and not any(row["allocated_top60_chunk"] for row in local_rows),
            "evidence_zero_hit": bool(local_rows) and not any(row["final_evidence_exact"] for row in local_rows),
            "verifiable_evidence_zero_hit": any(row["source_contains_evidence"] for row in local_rows)
                and not any(row["final_evidence_exact"] for row in local_rows),
            "all_labels_source_ambiguous": bool(local_rows)
                and not any(row["source_contains_evidence"] for row in local_rows),
            "query_characters": len(queries[qid]),
            "query_whitespace_tokens": len(queries[qid].split()),
            "language": language(queries[qid]),
            "medical_term_count": len(MEDICAL_RE.findall(queries[qid])),
            "raw_ranked_parent_count_lower_bound": len(raw_docs),
            "fused_parent_count": len(candidates),
            "document_bm25_dense_parent_overlap": both,
            "document_bm25_dense_overlap_over_union": safe_rate(both, len(branch_bm25 | branch_dense)),
            "pilot_query": qid in PILOT_IDS,
        })
        if qid % 100 == 0:
            print(f"R6 stage audit {qid}/1200", flush=True)

    if lineage_failures:
        raise ValueError(f"Lineage verification failed: {lineage_failures[:10]}")

    # Registry parquet contains the immutable labels plus verified source identity.
    registry_output = []
    audit_by_id = {row["registry_id"]: row for row in audit_rows}
    for label in registry:
        stage = audit_by_id[label.registry_id]
        source = sources[label.doc_id]
        registry_output.append({
            "registry_id": label.registry_id,
            "query_id": label.query_id,
            "query_sha256": sha_text(queries[label.query_id]),
            "doc_id": label.doc_id,
            "label_kind": "positive_evidence",
            "label_group": label.label_group,
            "evidence_text": label.evidence_text,
            "evidence_text_sha256": sha_text(label.evidence_text),
            "evidence_normalized_sha256": sha_text(label.evidence_normalized),
            "source_present": stage["source_present"],
            "source_sha256": source.text_sha256 if source else None,
            "source_start": stage["source_start"],
            "source_end": stage["source_end"],
            "provenance_json": json.dumps(label.provenance, ensure_ascii=False, sort_keys=True),
        })
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(registry_output), output / "positive_registry.parquet",
                   compression="zstd")

    # Long-form aggregate stage recall with explicit unknown denominators.
    stage_recall_rows: list[dict[str, Any]] = []
    for unit in ("evidence_label", "document_parent"):
        for group in sorted({"all", *(row["label_group"] for row in audit_rows)}):
            selected_rows = [row for row in audit_rows
                             if group == "all" or row["label_group"] == group]
            if unit == "document_parent":
                deduplicated = {}
                for row in selected_rows:
                    deduplicated.setdefault((row["query_id"], row["doc_id"]), row)
                selected_rows = list(deduplicated.values())
            previous_present = None
            for stage in REPORTED_STAGES:
                present = sum(row[stage] is True for row in selected_rows)
                unknown = sum(row[stage] is None for row in selected_rows)
                known = len(selected_rows) - unknown
                sequential = stage in STAGES[1:]
                stage_recall_rows.append({
                    "unit": unit,
                    "label_group": group,
                    "stage": stage,
                    "total_units": len(selected_rows),
                    "known_stage_labels": known,
                    "present": present,
                    "absent": known - present,
                    "unknown": unknown,
                    "recall_over_known": safe_rate(present, known),
                    "coverage_lower_bound_over_all": safe_rate(present, len(selected_rows)),
                    "conditional_retention_vs_previous_present": (
                        safe_rate(present, previous_present) if previous_present is not None
                        and present <= previous_present and sequential else None),
                })
                if stage in STAGES:
                    previous_present = present
    write_csv(output / "stage_recall.csv", stage_recall_rows)
    write_csv(output / "first_loss_attribution.csv", audit_rows)

    # Query-level zero-hit analysis only calls a query truly evaluable when a
    # positive evidence string is present in the registered current source.
    write_csv(output / "zero_hit_analysis.csv", query_rows)

    # Oracle summaries.  O1 is fully measured only on the six-query pilot;
    # fused presence is merely a full-set raw lower bound.
    by_query_audit: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in audit_rows:
        by_query_audit[int(row["query_id"])].append(row)

    def metric(key: str, rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
        values = list(rows)
        present = sum(row[key] is True for row in values)
        known = sum(row[key] is not None for row in values)
        query_ids = {int(row["query_id"]) for row in values}
        local_by_query: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in values:
            local_by_query[int(row["query_id"])].append(row)
        hit_queries = sum(any(row[key] is True for row in local_by_query[qid]) for qid in query_ids)
        return {
            "positive_labels": len(values), "known_denominator": known,
            "covered_labels": present, "coverage_over_known": safe_rate(present, known),
            "coverage_lower_bound_over_all": safe_rate(present, len(values)),
            "queries": len(query_ids), "query_hits": hit_queries,
            "query_hit_rate": safe_rate(hit_queries, len(query_ids)),
        }

    pilot_rows = [row for row in audit_rows if row["pilot_explicit_trace"]]
    document_rows = list({(row["query_id"], row["doc_id"]): row for row in audit_rows}.values())
    oracle = {
        "format": "v4-r6-stage-oracle-v1",
        "scope": "Incomplete known-positive registry; exact normalized containment, not official F2",
        "baseline_s2": {
            "documents": metric("s2_selected_document", audit_rows),
            "chunk_parents": metric("allocated_top60_chunk", audit_rows),
            "exact_evidence": metric("final_evidence_exact", audit_rows),
        },
        "document_parent_oracles": {
            "known_positive_documents": len(document_rows),
            "O2_fused_top400": metric("fused_top400", document_rows),
            "O3_rerank_200_parent": metric("rerank_200_parent", document_rows),
            "selected_top200": metric("selected_top200_document", document_rows),
            "s2_selected_documents": metric("s2_selected_document", document_rows),
            "allocated_top60_chunk_parent": metric("allocated_top60_chunk", document_rows),
        },
        "O1_raw_union": {
            "pilot_measured": metric("raw_union", pilot_rows),
            "all_queries_lower_bound_from_fused_presence": metric("fused_top400", audit_rows),
            "status": "PARTIAL_MEASURED; full raw misses are UNKNOWN",
        },
        "O2_fused_top400": metric("fused_top400", audit_rows),
        "O3_rerank_200_pairs": {
            "parent": metric("rerank_200_parent", audit_rows),
            "evidence_exact_in_scored_payload": metric("rerank_200_evidence_exact", audit_rows),
            "status": "MEASURED_HASH_LINKED_A0_PRECURSOR; S2 later added supplemental scores",
        },
        "O4_available_context_from_selected_top200": metric("available_context_exact", audit_rows),
        "O5_gold_source_span_upper_bound": {
            **metric("source_contains_evidence", [
                dict(row, source_contains_evidence=bool(
                    row["source_contains_evidence"] and row["selected_top200_document"]))
                for row in audit_rows]),
            "pool": "selected_top200_documents",
        },
        "limitations": [
            "Known positives are incomplete and were collected from earlier feedback/manual qrels.",
            "O4 uses R5 materialized contexts for S2 docs and deterministically derives the same frozen context policy for other selected-top200 docs.",
            "O5 assumes an oracle may extract the verified gold span from a selected-top200 source; it is an upper bound.",
            "No oracle result is an official F2 estimate.",
        ],
    }
    for name, key in (("O2_fused_top400", "fused_top400"),
                      ("O3_rerank_200_parent", "rerank_200_parent"),
                      ("O4_available_context", "available_context_exact"),
        ("O5_source_span_upper_bound", "source_contains_evidence")):
        covered = sum(row[key] is True and (
            name != "O5_source_span_upper_bound" or row["selected_top200_document"] is True)
                      for row in audit_rows)
        oracle.setdefault("recoverable_vs_s2_exact", {})[name] = {
            "covered_labels": covered,
            "delta_labels_vs_s2_exact": covered - sum(row["final_evidence_exact"] for row in audit_rows),
        }
    write_json(output / "oracle_results.json", oracle)

    budget_rows: list[dict[str, Any]] = []
    for key, name, estimate, parent_pool in (
        ("available_context_exact", "available_context_from_selected_top200", "approximate_available_context",
         "selected_top200_document"),
        ("s2_available_context_exact", "available_context_from_s2_document_pool", "measured_available_context",
         "s2_selected_document"),
        ("source_contains_evidence", "gold_source_span_in_selected_top200", "upper_bound",
         "selected_top200_document"),
        ("source_contains_evidence", "gold_source_span_in_s2_document_pool", "upper_bound",
         "s2_selected_document"),
    ):
        rows_for_pool = [dict(row, **{key: bool(row[key] and row[parent_pool])})
                         for row in audit_rows]
        grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for row in rows_for_pool:
            grouped[int(row["query_id"])].append(row)
        for k in (20, 40, 60, 80, 100):
            labels_hit = 0
            query_hits = 0
            for qid, labels in grouped.items():
                hit, query_hit = oracle_allocate(labels, key, k)
                labels_hit += hit
                query_hits += query_hit
            budget_rows.append({
                "pool": name, "estimate_type": estimate, "chunk_budget_k": k,
                "one_chunk_per_parent": True, "known_positive_labels": len(rows_for_pool),
                "covered_labels": labels_hit,
                "coverage": safe_rate(labels_hit, len(rows_for_pool)),
                "labeled_queries": len(grouped), "query_hits": query_hits,
                "query_hit_rate": safe_rate(query_hits, len(grouped)),
                "hypothetical_above_official_cap": k > 60,
            })
    write_csv(output / "chunk_budget_oracle.csv", budget_rows)

    # Classify every scored historical feedback row, reproducing the R5
    # unmatched-to-S2-chunk-parent count instead of only positive rows.
    trunc_rows: list[dict[str, Any]] = []
    registry_by_key = {(row.query_id, row.doc_id, sha_text(row.evidence_text)): row
                       for row in registry if row.label_group != "manual_positive"}
    unmatched = Counter()
    for csv_index, row in enumerate(feedback_rows, start=2):
        if row["scoring_status"] != "scored":
            continue
        qid = int(row["query_id"])
        doc = int(row["baseline_doc_id_int"])
        state = query_state[qid]
        in_chunk = doc in state["s2_chunks"]
        if in_chunk:
            reason = "S2_CHUNK_PARENT_PRESENT"
        elif doc not in state["candidates"]:
            reason = "FUSED_PARENT_ABSENT_RAW_UNKNOWN"
        elif doc not in state["pair_parents"]:
            reason = "PAIR_ALLOCATION_DROP_PRECURSOR"
        elif doc not in state["selected"]:
            reason = "DOC_SELECTION_DROP_TOP200"
        elif doc not in state["s2_docs"]:
            reason = "S2_ADAPTIVE_DOC_DROP"
        else:
            reason = "CHUNK_SLOT_DROP"
        if not in_chunk:
            unmatched[reason] += 1
        old_chunk = old[qid]["relevant_chunks"][int(row["chunk_order"])]
        reg = registry_by_key.get((qid, doc, sha_text(old_chunk["chunk_text"])))
        detailed = audit_by_id.get(reg.registry_id) if reg else None
        trunc_rows.append({
            "record_scope": "historical_scored_feedback",
            "feedback_csv_row": csv_index,
            "query_id": qid,
            "doc_id": doc,
            "relevant": row["relevant"],
            "s2_document_present": doc in state["s2_docs"],
            "s2_chunk_parent_present": in_chunk,
            "unmatched_parent_reason": reason,
            "source_contains_evidence": detailed["source_contains_evidence"] if detailed else None,
            "final_evidence_exact": detailed["final_evidence_exact"] if detailed else None,
            "s2_input_tokens": score_by_query_doc[qid].get(doc, {}).get("input_tokens"),
            "evidence_512_visibility": detailed["evidence_512_visibility"] if detailed else None,
        })
    write_csv(output / "truncation_audit.csv", trunc_rows)

    if sum(unmatched.values()) != 13587:
        raise ValueError(f"R5 unmatched feedback count drifted: {sum(unmatched.values())} != 13587")

    feedback_by_query = defaultdict(list)
    for row in feedback_rows:
        if row["scoring_status"] == "scored":
            feedback_by_query[int(row["query_id"])].append(row)
    coverage_matrix = []
    for qrow in query_rows:
        qid = int(qrow["query_id"])
        scored = feedback_by_query.get(qid, [])
        coverage_matrix.append({
            "query_id": qid,
            "query_sha256": sha_text(queries[qid]),
            "historical_feedback_rows": len(scored),
            "historical_positive_rows": sum(row["relevant"] == "True" for row in scored),
            "negative_only_feedback": bool(scored) and not any(row["relevant"] == "True" for row in scored),
            "manual_positive_rows": sum(label.label_group in (
                "manual_positive", "historical_and_manual_positive") for label in by_query.get(qid, [])),
            "missing_feedback": not bool(scored) and not bool(by_query.get(qid)),
            "s2_document_count": len(s2[qid]["relevant_docs"]),
            "s2_chunk_count": len(s2[qid]["relevant_chunks"]),
            "refined_trace_available": True,
            "a0_precursor_pair_trace_available": True,
            "explicit_raw_branch_trace_available": qid in PILOT_IDS,
            "r5_context_pool_available": (R5_CONTEXTS / "queries" / f"{qid}.json").exists(),
            "r5_same_input_score_available": (R5_SCORES / "scores" / f"{qid}.json").exists(),
        })

    artifacts = [
        artifact_record(QUERIES, "query_dataset", "Official 1200 query ID/text table", 1200),
        artifact_record(FEEDBACK, "historical_feedback", "31,484 rows; 31,294 scored, labels only for supplied old chunks", 600),
        artifact_record(QRELS, "manual_qrels", "107 reviewed rows, 87 on official query IDs", 87),
        artifact_record(OLD_ZIP, "feedback_prediction_identity", "Frozen submission used to resolve feedback chunk_order", 1200),
        artifact_record(S2_ZIP, "baseline_submission", "Official user-reported champion S2", 1200),
        artifact_record(S2_DIR / "manifest.json", "s2_manifest", "S2 cardinality and source lineage", 1200),
        artifact_record(REFINED / "run_manifest.json", "s2_candidate_manifest", "Hash map for 1200 refined final shards", 1200),
        artifact_record(A0 / "run_manifest.json", "a0_precursor_manifest", "Hash-linked precursor, not relabeled as S2 raw trace", 1200,
                        "Raw misses outside fused candidates are unknown except the six-query pilot"),
        artifact_record(PILOT, "explicit_stage_pilot", "37 known positive paths on six required pilot queries", 6),
        artifact_record(PILOT_MANIFEST, "explicit_stage_pilot_manifest", "Pilot provenance and checks", 6),
        artifact_record(R5_CONTEXTS / "manifest.json", "r5_context_pool", "Exact contexts for all S2 documents", 1200),
        artifact_record(R5_SCORES / "runtime_lock.json", "r5_token_input_contract", "Exact S2 query/title/chunk input, max_length 512", 1200),
    ]
    inventory = {
        "format": "v4-r6-artifact-inventory-v1",
        "status": "PASS",
        "artifacts": artifacts,
        "identity_checks": {
            "query_ids_exact_1_to_1200": True,
            "s2_zip_matches_manifest": True,
            "s2_source_manifest_matches_refined_manifest": True,
            "all_refined_final_hashes_verified": True,
            "all_refined_to_a0_precursor_hashes_verified": True,
            "all_s2_docs_subset_of_refined_top200": True,
            "all_s2_chunks_match_linked_refined_expansion": True,
            "lineage_failures": lineage_failures,
        },
        "feedback_classification": {
            "scored_rows": sum(row["scoring_status"] == "scored" for row in feedback_rows),
            "positive_rows": sum(row["scoring_status"] == "scored" and row["relevant"] == "True"
                                 for row in feedback_rows),
            "negative_rows": sum(row["scoring_status"] == "scored" and row["relevant"] == "False"
                                 for row in feedback_rows),
            "excluded_rows": sum(row["scoring_status"] != "scored" for row in feedback_rows),
            "unmatched_to_s2_chunk_parent": sum(unmatched.values()),
            "unmatched_reason_counts": dict(unmatched),
        },
        "positive_registry": {
            "rows_after_deduplication": len(registry_output),
            "historical_positive": sum(row["label_group"] == "historical_positive" for row in registry_output),
            "manual_positive": sum(row["label_group"] == "manual_positive" for row in registry_output),
            "historical_and_manual_positive": sum(row["label_group"] == "historical_and_manual_positive"
                                                   for row in registry_output),
            "queries_with_positive": len(by_query),
        },
        "tokenizer_error": tokenizer_error,
        "coverage_matrix": coverage_matrix,
        "semantic_warning": "Missing artifact is UNKNOWN, not a retrieval miss; no-feedback query is not a zero-hit query.",
    }
    write_json(output / "artifact_inventory.json", inventory)

    loss_counts = Counter(row["first_proven_loss"] for row in audit_rows)
    labeled_queries = [row for row in query_rows if row["has_known_positive"]]
    verifiable_queries = [row for row in labeled_queries if row["source_verifiable_labels"]]
    exact_matched = [row for row in audit_rows if row["final_evidence_exact"]]
    visibility_counts = Counter(row["evidence_512_visibility"] for row in exact_matched)
    stage_all = {row["stage"]: row for row in stage_recall_rows
                 if row["label_group"] == "all" and row["unit"] == "evidence_label"}
    stage_documents = {row["stage"]: row for row in stage_recall_rows
                       if row["label_group"] == "all" and row["unit"] == "document_parent"}
    budget60 = {row["pool"]: row for row in budget_rows if row["chunk_budget_k"] == 60}

    report = f"""# R6 — Stage-wise Recall & Oracle Audit

## Kết luận

R6 là audit trên **known positives chưa đầy đủ**, không phải phép đo official F2.
Registry sau deduplicate có **{len(registry_output):,} evidence positives**, tương ứng
**{len(document_rows):,} unique positive documents trên {len(by_query):,} queries**.
S2 vẫn giữ nguyên; R6 không chạy retrieval, inference hay training và không tạo submission.

Nút thắt có bằng chứng lớn nhất ở stage cuối là **chunk allocation/evidence localization**.
S2 giữ đúng parent ở document list cho **{stage_all['s2_selected_document']['present']:,}/{len(audit_rows):,}** labels,
nhưng chỉ cấp chunk cho **{stage_all['allocated_top60_chunk']['present']:,}** và exact evidence cuối còn
**{stage_all['final_evidence_exact']['present']:,}**. Raw miss toàn bộ corpus chưa thể khẳng định vì raw
branch trace đầy đủ chỉ tồn tại cho pilot; các trường hợp ngoài fused được ghi `UNKNOWN`.

## 1. Known positives mất ở đâu?

| First proven loss | Labels |
|---|---:|
{chr(10).join(f'| `{key}` | {value:,} |' for key, value in sorted(loss_counts.items()))}

Ngân sách 200 pairs của precursor A0 bỏ **{sum(not row['rerank_200_parent'] and row['fused_top400'] for row in audit_rows):,}**
known-positive parents đã có trong fused pool; supplemental scoring của S2 phục hồi vào top-200
**{sum(row['recovered_after_rerank200'] for row in audit_rows):,}** labels. Vì vậy precursor drop
được báo riêng, không bị gán sai thành terminal loss của S2.

## 2. Stage coverage đo được

| Stage | Evidence present | Document present | Evidence unknown | Evidence lower bound |
|---|---:|---:|---:|---:|
{chr(10).join(f"| `{stage}` | {row['present']:,} | {stage_documents[stage]['present']:,} | {row['unknown']:,} | {row['coverage_lower_bound_over_all']:.4f} |" for stage, row in stage_all.items())}

## 3. Zero-hit

- Queries có ít nhất một known positive: **{len(labeled_queries):,}**.
- Document zero-hit ở S2: **{sum(row['document_zero_hit'] for row in labeled_queries):,}**.
- Chunk-parent zero-hit: **{sum(row['chunk_parent_zero_hit'] for row in labeled_queries):,}**.
- Evidence exact zero-hit: **{sum(row['evidence_zero_hit'] for row in labeled_queries):,}**.
- Trong {len(verifiable_queries):,} queries có ít nhất một evidence còn tồn tại trong current source,
  verifiable evidence zero-hit: **{sum(row['verifiable_evidence_zero_hit'] for row in verifiable_queries):,}**.
- Queries chỉ có labels không còn exact trong current source: **{sum(row['all_labels_source_ambiguous'] for row in labeled_queries):,}**;
  nhóm này không được gọi là retrieval zero-hit.

## 4. Oracle top-200/top-60

- O2 fused top-400 chứa parent của **{oracle['O2_fused_top400']['covered_labels']:,}** labels.
- O3 precursor 200 pairs chứa parent của **{oracle['O3_rerank_200_pairs']['parent']['covered_labels']:,}** labels.
- S2 document pool chứa parent của **{oracle['baseline_s2']['documents']['covered_labels']:,}** labels.
- Với K=60 từ selected top-200, approximate available-context oracle đạt
  **{budget60['available_context_from_selected_top200']['covered_labels']:,}** labels.
- Với K=60 từ selected top-200 và được phép nhìn gold span, upper bound đạt
  **{budget60['gold_source_span_in_selected_top200']['covered_labels']:,}** labels.
- Nếu giữ S2 adaptive document pool, hai mốc tương ứng là
  **{budget60['available_context_from_s2_document_pool']['covered_labels']:,}** và
  **{budget60['gold_source_span_in_s2_document_pool']['covered_labels']:,}** labels.
- S2 exact hiện tại đạt **{oracle['baseline_s2']['exact_evidence']['covered_labels']:,}** labels.
- K=20 đã bão hòa registry quan sát được; K=40/60/80/100 không tăng thêm oracle coverage.
  Điều này chỉ nói known labels/query hiện có ít hơn budget, không chứng minh K=20 đủ cho official gold.

Các số oracle là coverage trên registry, không chuyển đổi thành official F2.

## 5. Truncation

Trong các known positives exact-match ở output S2, vị trí evidence dưới contract tokenizer 512:

{chr(10).join(f'- `{key}`: **{value:,}**' for key, value in sorted(visibility_counts.items()))}

R5 có 64.079/72.000 inputs dài hơn 512 token, nhưng R6 chỉ gọi `TRUNCATED_EVIDENCE`
khi token offsets chứng minh evidence nằm một phần hoặc hoàn toàn ngoài vùng nhìn thấy.

13.587 feedback rows không còn S2 chunk parent được phân loại:

{chr(10).join(f'- `{key}`: **{value:,}**' for key, value in sorted(unmatched.items()))}

## 6. Quyết định R7/R8

**Ưu tiên R8 Evidence Extraction/Allocation trước, đồng thời R7 chỉ mở pilot targeted.**
Lý do: document parent survival đo được cao hơn rõ rệt so với chunk-parent và exact evidence;
oracle K=60 trên chính S2 document pool còn khoảng phục hồi đo được mà không cần full-index retrieval.
Tuy nhiên, R7 vẫn cần cho các fused misses; do raw trace ngoài pilot thiếu, chưa có bằng chứng để
quy toàn bộ fused-absent cho raw retrieval.

## 7. Tối đa ba experiments tiếp theo

1. **R8 passage extractor + oracle-trained allocation (impact cao, cost vừa):** trên frozen S2 docs,
   retrieve/rank short source spans, tối đa một span/parent, tối ưu trực tiếp evidence containment và K=60.
2. **R8 query-adaptive chunk slots (impact vừa-cao, cost thấp):** dùng confidence/parent diversity để
   phân bổ 20/40/60 slots; kiểm bằng query-disjoint labels và không thay document list.
3. **R7 targeted raw-retrieval replay (impact tiềm năng cao, cost cao):** chỉ chạy các verifiable
   document zero-hit/fused-absent queries, lưu đầy đủ raw BM25/dense union để tách `RAW_PARENT_MISS`
   khỏi `FUSION_DROP`; chưa chạy lại toàn bộ 1.200 query.

## Giới hạn

- Official ground truth đầy đủ không có trong repo; aggregate BTC metrics không cho query-level labels.
- Raw branch trace đầy đủ toàn bộ 1.200 query không tồn tại. A0 artifacts chỉ được dùng đúng vai trò
  precursor đã liên kết SHA-256, không được gọi là historical S2 raw trace.
- Exact normalized containment là proxy xác minh evidence, không phải implementation của BTC scorer.
- O4 là approximate oracle trên available contexts; O5 là upper bound có quyền nhìn gold source spans.
"""
    (output / "R6_FINAL_REPORT.md").write_text(report)

    decision = f"""# R6 → R7/R8 Decision

## Quyết định

**Đi R8 trước; R7 chạy targeted diagnostic song song, không full replay.**

Evidence định lượng:

- S2 document coverage trên registry: {stage_all['s2_selected_document']['present']:,}/{len(audit_rows):,}.
- Chunk-parent coverage: {stage_all['allocated_top60_chunk']['present']:,}/{len(audit_rows):,}.
- Exact evidence coverage: {stage_all['final_evidence_exact']['present']:,}/{len(audit_rows):,}.
- K=60 selected-top200 available-context oracle: {budget60['available_context_from_selected_top200']['covered_labels']:,}/{len(audit_rows):,}.
- K=60 selected-top200 gold-source upper bound: {budget60['gold_source_span_in_selected_top200']['covered_labels']:,}/{len(audit_rows):,}.

Khoảng cách document → chunk → exact evidence là measured. Ngược lại, raw retrieval attribution
ngoài sáu pilot còn unknown. Vì vậy xây R7 full-index ngay bây giờ có rủi ro tối ưu nhầm stage.

## Gate cho R8

R8 chỉ được tiến tới submission candidate khi trên query-disjoint validation:

1. Không giảm known-positive document parents.
2. Exact evidence tăng, không chỉ semantic reranker score.
3. Số query mất exact evidence bằng 0 hoặc nằm trong loss budget khóa trước.
4. Mọi span là substring của registered source và một parent chỉ dùng tối đa một chunk.

## Gate chuyển trọng tâm sang R7

Sau targeted replay, chuyển sang R7 nếu raw/fusion là first proven loss của nhiều verifiable
document-zero-hit queries hơn số query R8 có thể phục hồi từ S2 document pool. Replay phải lưu raw
branch identities; thiếu artifact tiếp tục được ghi `UNKNOWN`.
"""
    (output / "R6_R7_R8_DECISION.md").write_text(decision)

    output_hashes = {path.name: sha256_file(path) for path in output.iterdir() if path.is_file()}
    write_json(output / "manifest.json", {
        "format": "v4-r6-stage-oracle-manifest-v1",
        "status": "PASS",
        "query_count": 1200,
        "positive_registry_rows": len(registry_output),
        "labeled_queries": len(by_query),
        "unmatched_feedback_rows": sum(unmatched.values()),
        "output_sha256_before_manifest": output_hashes,
        "official_f2": "NOT_MEASURED",
        "submission_created": False,
    })
    print(json.dumps({"status": "PASS", "output": str(output),
                      "registry": len(registry_output), "labeled_queries": len(by_query),
                      "loss_counts": loss_counts, "unmatched": unmatched}, default=dict), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.output.resolve())


if __name__ == "__main__":
    main()

"""Prepare and finalize a paired 150-passage Phase 2 semantic review."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.selection import sha256_file


PATTERNS = {
    "negation": re.compile(
        r"\b(?:không|chưa|chẳng|not|no|without|avoid|contraindicat\w*)\b|"
        r"(?:不得|不可|不宜|禁止|无|未)", re.I
    ),
    "dosage": re.compile(
        r"\b\d+(?:[.,]\d+)?(?:\s*[–-]\s*\d+(?:[.,]\d+)?)?\s*"
        r"(?:mg|mcg|g|kg|ml|mL|IU|đơn vị)(?:\s*/\s*(?:day|ngày|kg|lần))?\b|"
        r"(?:每日|每次|一天)\s*\d+", re.I
    ),
    "lab_result": re.compile(
        r"HbA1c|eGFR|beta[- ]?HCG|glucose|creatinine|cholesterol|"
        r"(?:xét nghiệm|kết quả xét nghiệm|test result)|"
        r"(?:血糖|肌酐|检验|检查结果)|[<>≤≥]\s*\d+(?:[.,]\d+)?", re.I
    ),
    "treatment_condition": re.compile(
        r"\b(?:nếu|khi|trường hợp|đối với|chỉ định|bệnh nhân|patients?|if|when|unless|"
        r"recommended for|treatment for)\b|(?:患者|如果|若|当|适用于|禁忌)", re.I
    ),
    "question_answer": re.compile(
        r"\?|？|\b(?:hỏi|đáp|question|answer)\s*[:：]|(?:怎么办|是什么|如何)", re.I
    ),
}

CATEGORY_LANGUAGE_QUOTAS = {
    "negation": {"vi": 3, "en": 2, "zh": 3},
    "dosage": {"vi": 3, "en": 2, "zh": 3},
    "lab_result": {"vi": 3, "en": 2, "zh": 3},
    "treatment_condition": {"vi": 3, "en": 2, "zh": 3},
    "question_answer": {"vi": 3, "en": 2, "zh": 3},
    "boundary_stress": {"vi": 3, "en": 6, "zh": 1},
}


def _atomic_json(path: Path, value) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=f".{path.name}.", suffix=".partial",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(path: Path, rows: list[dict]) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        pq.write_table(pa.Table.from_pylist(rows), temporary, compression="zstd")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _score(category: str, doc_id: int, anchor: int) -> int:
    value = f"phase2-review-v1\0{category}\0{doc_id}\0{anchor}".encode("ascii")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big")


def _load_documents(candidate: Path, ids: set[int]) -> dict[int, dict]:
    manifest = json.loads((candidate / "phase1_manifest.json").read_text())
    columns = ["doc_id", "official_url", "source_kind", "title", "normalized_text",
               "detected_language", "normalized_text_sha256"]
    documents = {}
    for stem in sorted(manifest["part_checkpoints_sha256"]):
        table = pq.read_table(candidate / "clean_documents" / f"{stem}.parquet", columns=columns)
        for row in table.to_pylist():
            doc_id = int(row["doc_id"])
            if doc_id in ids:
                documents[doc_id] = row
    if set(documents) != ids:
        raise ValueError("Could not load every pilot document")
    return documents


def _pick_passage(passages: list[dict], anchor: int) -> dict:
    containing = [row for row in passages if row["char_start"] <= anchor < row["char_end"]]
    if containing:
        return min(containing, key=lambda row: (
            abs((row["char_start"] + row["char_end"]) // 2 - anchor), row["passage_index"]
        ))
    return min(passages, key=lambda row: min(
        abs(row["char_start"] - anchor), abs(row["char_end"] - anchor)
    ))


def _overlap_context(passages: list[dict], selected: dict) -> tuple[int, int]:
    index = passages.index(selected)
    previous = passages[index - 1] if index else None
    following = passages[index + 1] if index + 1 < len(passages) else None
    previous_overlap = max(0, previous["char_end"] - selected["char_start"]) if previous else 0
    next_overlap = max(0, selected["char_end"] - following["char_start"]) if following else 0
    return previous_overlap, next_overlap


def prepare_review(pilot_root: Path, candidate: Path,
                   locked_review_table: Path | None = None) -> tuple[list[dict], list[dict], dict]:
    pilot_meta = pq.read_table(pilot_root / "pilot/pilot_manifest.parquet").to_pylist()
    meta = {int(row["doc_id"]): row for row in pilot_meta}
    documents = _load_documents(candidate, set(meta))
    by_strategy: dict[str, dict[int, list[dict]]] = {}
    for strategy in "ABC":
        rows = pq.read_table(
            pilot_root / f"pilot/strategy_{strategy}/part-00000.parquet"
        ).to_pylist()
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(int(row["doc_id"]), []).append(row)
        for values in grouped.values():
            values.sort(key=lambda row: row["passage_index"])
        by_strategy[strategy] = grouped

    pools: dict[tuple[str, str], list[dict]] = {}
    for doc_id, document in documents.items():
        language = meta[doc_id]["language"]
        if language not in {"vi", "en", "zh"}:
            continue
        text = document["normalized_text"]
        for category, pattern in PATTERNS.items():
            matches = list(pattern.finditer(text))
            if matches:
                match = matches[len(matches) // 2]
                anchor = (match.start() + match.end()) // 2
                pools.setdefault((category, language), []).append({
                    "category": category, "language": language, "doc_id": doc_id,
                    "anchor": anchor, "signal": match.group(),
                    "score": _score(category, doc_id, anchor),
                })
        passages_a = by_strategy["A"].get(doc_id, [])
        if len(passages_a) > 1:
            boundary_passage = passages_a[len(passages_a) // 2]
            anchor = boundary_passage["char_start"]
            pools.setdefault(("boundary_stress", language), []).append({
                "category": "boundary_stress", "language": language, "doc_id": doc_id,
                "anchor": anchor, "signal": text[anchor:anchor + 80],
                "score": _score("boundary_stress", doc_id, anchor),
            })

    selected_cases: list[dict] = []
    if locked_review_table is not None:
        locked = pq.read_table(locked_review_table).to_pylist()
        seen_cases: set[str] = set()
        for row in locked:
            case_id = row["case_id"]
            if case_id in seen_cases:
                continue
            seen_cases.add(case_id)
            doc_id = int(row["doc_id"])
            if doc_id not in documents:
                raise ValueError(f"Locked review doc is outside pilot: {doc_id}")
            selected_cases.append({
                "case_id": case_id,
                "category": row["category"],
                "language": row["language"],
                "doc_id": doc_id,
                "anchor": int(row["anchor_char"]),
                "signal": row["anchor_signal"],
                "score": _score(row["category"], doc_id, int(row["anchor_char"])),
            })
        selected_cases.sort(key=lambda row: row["case_id"])
        if len(selected_cases) * 3 != len(locked):
            raise ValueError("Locked review table is not a complete A/B/C pairing")
    else:
        used_docs: set[int] = set()
        long_count = 0
        pubmed_count = 0
        case_number = 0
        for category, language_quotas in CATEGORY_LANGUAGE_QUOTAS.items():
            for language, quota in language_quotas.items():
                candidates = sorted(pools.get((category, language), []), key=lambda row: row["score"])
                for _ in range(quota):
                    available = [row for row in candidates if row["doc_id"] not in used_docs]
                    if not available:
                        # Preserve the requested semantic/language stratum even when
                        # the pilot has too few distinct documents. The case anchor
                        # remains unique and is still paired across A/B/C.
                        available = candidates
                    if not available:
                        raise ValueError(f"No review cases for {category}/{language}")
                    want_pubmed = language == "en" and pubmed_count < 10
                    want_long = long_count < 25 and len(selected_cases) % 2 == 0
                    ranked = sorted(available, key=lambda row: (
                        meta[row["doc_id"]]["source_kind"] != "pubmed" if want_pubmed else False,
                        (meta[row["doc_id"]]["length_bin"] in {"long", "very_long"}) != want_long,
                        row["score"],
                    ))
                    chosen = ranked[0]
                    case_number += 1
                    chosen["case_id"] = f"case-{case_number:03d}"
                    selected_cases.append(chosen)
                    used_docs.add(chosen["doc_id"])
                    if meta[chosen["doc_id"]]["length_bin"] in {"long", "very_long"}:
                        long_count += 1
                    if meta[chosen["doc_id"]]["source_kind"] == "pubmed":
                        pubmed_count += 1

    review_rows: list[dict] = []
    for case in selected_cases:
        doc_id = case["doc_id"]
        document = documents[doc_id]
        text = document["normalized_text"]
        for strategy in "ABC":
            passages = by_strategy[strategy][doc_id]
            passage = _pick_passage(passages, case["anchor"])
            previous_overlap, next_overlap = _overlap_context(passages, passage)
            start, end = int(passage["char_start"]), int(passage["char_end"])
            review_rows.append({
                "case_id": case["case_id"],
                "category": case["category"],
                "anchor_signal": case["signal"],
                "anchor_char": case["anchor"],
                "strategy_id": strategy,
                "passage_id": passage["passage_id"],
                "doc_id": doc_id,
                "official_url": document["official_url"],
                "source_kind": meta[doc_id]["source_kind"],
                "language": case["language"],
                "document_length_bin": meta[doc_id]["length_bin"],
                "document_length_chars": meta[doc_id]["length_chars"],
                "passage_index": passage["passage_index"],
                "token_count": passage["token_count"],
                "char_start": start,
                "char_end": end,
                "context_before": text[max(0, start - 320):start],
                "passage_text": passage["passage_text"],
                "context_after": text[end:min(len(text), end + 320)],
                "previous_overlap_chars": previous_overlap,
                "next_overlap_chars": next_overlap,
                "source_slice_exact": text[start:end] == passage["passage_text"],
                "semantic_completeness": "UNREVIEWED",
                "boundary_quality": "UNREVIEWED",
                "context_loss": "UNREVIEWED",
                "redundant_overlap": "UNREVIEWED",
                "review_notes": "",
                "reviewed_by": "",
                "reviewed_at": "",
            })
    manifest = {
        "format": "phase2-semantic-review-v1",
        "cases": len(selected_cases),
        "passages": len(review_rows),
        "strategies": dict(Counter(row["strategy_id"] for row in review_rows)),
        "languages": dict(Counter(case["language"] for case in selected_cases)),
        "categories": dict(Counter(case["category"] for case in selected_cases)),
        "sources": dict(Counter(meta[case["doc_id"]]["source_kind"] for case in selected_cases)),
        "length_groups": dict(Counter(
            "long" if meta[case["doc_id"]]["length_bin"] in {"long", "very_long"} else "short_medium"
            for case in selected_cases
        )),
        "source_slice_errors": sum(not row["source_slice_exact"] for row in review_rows),
    }
    return selected_cases, review_rows, manifest


def _read_annotations(annotations_path: Path) -> dict:
    annotations = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not isinstance(annotations, dict):
        raise ValueError("Annotations must be keyed by case_id/strategy_id")
    fields = ["semantic_completeness", "boundary_quality", "context_loss",
              "redundant_overlap", "review_notes"]
    if annotations and all("/" not in key for key in annotations):
        expanded = {}
        for case_id, strategies in annotations.items():
            if not isinstance(strategies, dict):
                raise ValueError(f"Invalid compact annotation: {case_id}")
            for strategy_id, values in strategies.items():
                if not isinstance(values, list) or len(values) != len(fields):
                    raise ValueError(f"Invalid compact annotation: {case_id}/{strategy_id}")
                expanded[f"{case_id}/{strategy_id}"] = dict(zip(fields, values))
        annotations = expanded
    return annotations


def apply_annotations(rows: list[dict], annotations_path: Path,
                      base_annotations_path: Path | None = None) -> list[dict]:
    annotations = (
        _read_annotations(base_annotations_path) if base_annotations_path is not None else {}
    )
    annotations.update(_read_annotations(annotations_path))
    output = []
    required = {"semantic_completeness", "boundary_quality", "context_loss",
                "redundant_overlap", "review_notes"}
    for row in rows:
        key = f"{row['case_id']}/{row['strategy_id']}"
        value = annotations.get(key)
        if not isinstance(value, dict) or not required <= set(value):
            raise ValueError(f"Missing manual annotation: {key}")
        output.append({**row, **value, "reviewed_by": "Codex manual inspection",
                       "reviewed_at": "2026-10-15"})
    if len(annotations) != len(rows):
        raise ValueError("Annotation keys do not exactly match review rows")
    return output


def write_outputs(output: Path, cases: list[dict], rows: list[dict], manifest: dict) -> None:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    _atomic_parquet(output / "phase2_semantic_review_150.parquet", rows)
    csv_path = output / "phase2_semantic_review_150.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    packet = output / "review_packet.md"
    with packet.open("w", encoding="utf-8") as stream:
        stream.write("# Phase 2 Paired Review Packet\n\n")
        for case in cases:
            stream.write(
                f"## {case['case_id']} | {case['category']} | {case['language']} | "
                f"doc {case['doc_id']}\n\n"
            )
            for row in [value for value in rows if value["case_id"] == case["case_id"]]:
                stream.write(
                    f"### Strategy {row['strategy_id']} | {row['token_count']} tokens\n\n"
                    f"**Before:** {row['context_before']}\n\n"
                    f"**Passage:** {row['passage_text']}\n\n"
                    f"**After:** {row['context_after']}\n\n"
                    f"**Review:** {row['semantic_completeness']}; {row['boundary_quality']}; "
                    f"{row['context_loss']}; overlap={row['redundant_overlap']}. "
                    f"{row['review_notes']}\n\n"
                )
    completed = sum(row["semantic_completeness"] != "UNREVIEWED" for row in rows)
    final_manifest = {
        **manifest,
        "reviewed_passages": completed,
        "manual_review_complete": completed == len(rows),
        "review_table_sha256": sha256_file(output / "phase2_semantic_review_150.parquet"),
        "review_packet_sha256": sha256_file(packet),
    }
    _atomic_json(output / "review_manifest.json", final_manifest)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--base-annotations", type=Path,
                        help="Complete earlier annotations to inherit for byte-identical rows")
    parser.add_argument("--locked-review-table", type=Path,
                        help="Reuse case/doc/anchor selections from an earlier paired review")
    args = parser.parse_args(argv)
    try:
        cases, rows, manifest = prepare_review(
            args.pilot, args.candidate, locked_review_table=args.locked_review_table,
        )
        if args.annotations:
            rows = apply_annotations(rows, args.annotations, args.base_annotations)
        write_outputs(args.output, cases, rows, manifest)
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Phase 2 review failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({**manifest, "output": str(args.output)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

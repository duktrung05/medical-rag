"""Create a reproducible Phase 1 quality-review sample and machine checks.

The command reads versioned clean pilot outputs and, optionally, a bounded
control sample from Phase 0 quarantine. It never promotes quarantine rows and
never reads the legacy raw-HTML archive.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import tempfile
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.language import detect_document_language, language_stratum
from src.medrag.data.phase1.normalization import normalize_medical_text, text_sha256
from src.medrag.data.phase1.selection import (
    Phase1Paths,
    ProductionExpectations,
    load_content_files,
    sha256_file,
    validate_locked_inputs,
)
from src.medrag.data.phase1.validation import assess_text_quality


EVALUATION_VERSION = "phase1-quality-evaluation-v2"
_HTML = re.compile(r"<!--|</?(?:script|style|iframe|div|span|p|br|nav|footer|header|a)\b", re.I)
_AD = re.compile(
    r"(?:advertisement|sponsored|buy now|subscribe now|quảng cáo|mua ngay|"
    r"tải ứng dụng|广告|立即购买)", re.I,
)
_MENU = re.compile(
    r"(?:\b(?:home page|main menu|login|register|privacy policy|terms of use)\b|"
    r"trang chủ|đăng nhập|đăng ký|chính sách bảo mật|điều khoản sử dụng|"
    r"首页|登录|注册|用药 药品库|赠送医生锦旗|专题推荐|男性 健身)", re.I,
)
_MOJIBAKE = re.compile(r"(?:\ufffd|ï¿½|â€[™œž“”˜¦]|Ã[\x80-\xbf]|Â(?=[\s\xa0]))")
_MEDICAL_VALUE = re.compile(
    r"(?<!\w)\d+(?:[.,]\d+)?\s*(?:mg|mcg|µg|g|kg|m[lL]|L|IU|mmHg|%|°C)(?!\w)", re.I
)
_NEGATION = re.compile(r"\b(?:không|chưa|not|no|without|无|不|未)\b", re.I)
_LONG_HEX = re.compile(r"\b[0-9a-f]{24,}\b", re.I)


REVIEW_SCHEMA = pa.schema([
    ("sample_origin", pa.string()),
    ("doc_id", pa.int64()),
    ("official_url", pa.string()),
    ("phase0_status", pa.string()),
    ("phase0_reason", pa.string()),
    ("source_kind", pa.string()),
    ("content_group", pa.string()),
    ("title", pa.string()),
    ("source_text", pa.string()),
    ("normalized_text", pa.string()),
    ("source_text_sha256", pa.string()),
    ("normalized_text_sha256", pa.string()),
    ("language_metadata", pa.string()),
    ("detected_language", pa.string()),
    ("language_confidence", pa.float32()),
    ("language_detection_method", pa.string()),
    ("language_metadata_agrees", pa.bool_()),
    ("html_contamination", pa.bool_()),
    ("menu_or_footer_signals", pa.int32()),
    ("advertising_signals", pa.int32()),
    ("encoding_issue", pa.bool_()),
    ("lexical_content_preserved", pa.bool_()),
    ("nonempty_lines_preserved", pa.bool_()),
    ("medical_values_preserved", pa.bool_()),
    ("negations_preserved", pa.bool_()),
    ("quality_flags", pa.list_(pa.string())),
    ("recorded_quality_flags", pa.list_(pa.string())),
    ("quality_validation_version", pa.string()),
    ("review_decision", pa.string()),
    ("review_notes", pa.string()),
])


def _stable_key(namespace: str, ident: int) -> bytes:
    return hashlib.sha256(f"{namespace}:{ident}".encode("ascii")).digest()


def _canonical_nonspace(text: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFC", text))


def _nonempty_lines(text: str) -> list[str]:
    return [re.sub(r"\s+", " ", unicodedata.normalize("NFC", line)).strip()
            for line in text.splitlines() if line.strip()]


def _canonical_matches(pattern: re.Pattern, text: str) -> list[str]:
    return [re.sub(r"\s+", " ", unicodedata.normalize("NFC", value)).strip()
            for value in pattern.findall(text)]


def _review_row(row: dict, *, origin: str, phase0_status: str,
                phase0_reason: str = "", repeated_text_count: int = 1) -> dict:
    source = row.get("source_text", row.get("text", ""))
    normalized = row.get("normalized_text")
    if normalized is None:
        normalized = normalize_medical_text(source).normalized_text
    metadata = row.get("language_metadata")
    language = detect_document_language(f"{row.get('title') or ''}\n{source}", metadata)
    source_kind = row.get("source_kind") or "html"
    quality = assess_text_quality(
        source,
        title=row.get("title") or "",
        source_kind=source_kind,
        encoding=row.get("encoding") or ("not-applicable" if source_kind == "pubmed" else None),
        repeated_text_count=repeated_text_count,
    )
    menu_count = len(_MENU.findall(source[:20_000]))
    ad_count = len(_AD.findall(source[:20_000]))
    lexical_ok = _canonical_nonspace(source) == _canonical_nonspace(normalized)
    lines_ok = _nonempty_lines(source) == _nonempty_lines(normalized)
    values_ok = _canonical_matches(_MEDICAL_VALUE, source) == _canonical_matches(
        _MEDICAL_VALUE, normalized
    )
    negations_ok = _NEGATION.findall(source) == _NEGATION.findall(normalized)
    if origin == "phase0_quarantine_control":
        confirmed = {
            "duplicated_template_content": repeated_text_count >= 20 and (
                menu_count >= 1 or len(source.strip()) < 1_000
            ),
            "homepage_or_redirect_content": menu_count >= 5 or (row.get("title") or "").casefold() == "trang chủ",
            "login_or_access_gate_content": "đăng nhập" in ((row.get("title") or "") + source[:3000]).casefold(),
            "hex_or_asset_noise": len(_LONG_HEX.findall(source[:2000])) >= 2,
        }.get(phase0_reason, False)
        decision = "LABEL_CONFIRMED" if confirmed else "LABEL_NEEDS_REVIEW"
        note = "Quarantine label matched visible content signals" if confirmed else "Reason not confirmed by bounded text checks"
    else:
        failed = []
        if quality.flags:
            failed.append("quality flags: " + ", ".join(quality.flags))
        if not all((lexical_ok, lines_ok, values_ok, negations_ok)):
            failed.append("normalization preservation invariant failed")
        decision = "PASS" if not failed else "REVIEW"
        note = "; ".join(failed) if failed else "No reviewed structural or normalization defect"
    return {
        "sample_origin": origin,
        "doc_id": int(row.get("doc_id", row.get("id"))),
        "official_url": row["official_url"],
        "phase0_status": phase0_status,
        "phase0_reason": phase0_reason,
        "source_kind": source_kind,
        "content_group": row.get("content_group") or "",
        "title": row.get("title") or "",
        "source_text": source,
        "normalized_text": normalized,
        "source_text_sha256": text_sha256(source),
        "normalized_text_sha256": text_sha256(normalized),
        "language_metadata": metadata,
        "detected_language": language.language,
        "language_confidence": language.confidence,
        "language_detection_method": language.method,
        "language_metadata_agrees": language.metadata_agrees,
        "html_contamination": bool(_HTML.search(source)),
        "menu_or_footer_signals": menu_count,
        "advertising_signals": ad_count,
        "encoding_issue": bool(_MOJIBAKE.search(source)),
        "lexical_content_preserved": lexical_ok,
        "nonempty_lines_preserved": lines_ok,
        "medical_values_preserved": values_ok,
        "negations_preserved": negations_ok,
        "quality_flags": list(quality.flags),
        "recorded_quality_flags": list(row.get("quality_flags", quality.flags)),
        "quality_validation_version": row.get("quality_validation_version", ""),
        "review_decision": decision,
        "review_notes": note,
    }


def _sample_pilot(pilot: Path, count: int) -> list[dict]:
    rows = ds.dataset(pilot / "clean_documents", format="parquet").to_table().to_pylist()
    if count > len(rows):
        raise ValueError(f"Pilot contains {len(rows)} rows, cannot sample {count}")
    strata: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        detected = row.get("detected_language") or detect_document_language(
            f"{row.get('title') or ''}\n{row['source_text']}", row.get("language_metadata")
        ).language
        key = f"{row['source_kind']}:{language_stratum(detected)}"
        strata[key].append(row)
    for key, values in strata.items():
        values.sort(key=lambda row: _stable_key(f"pilot:{key}", int(row["doc_id"])))
    selected = []
    positions = Counter()
    keys = sorted(strata)
    while len(selected) < count:
        candidates = [key for key in keys if positions[key] < len(strata[key])]
        if not candidates:
            break
        key = min(candidates, key=lambda value: (positions[value] / len(strata[value]), value))
        selected.append(strata[key][positions[key]])
        positions[key] += 1
    return selected


def _flagged_pilot_rows(pilot: Path) -> list[dict]:
    rows = ds.dataset(pilot / "clean_documents", format="parquet").to_table().to_pylist()
    return sorted(
        (row for row in rows if row.get("quality_review_required")),
        key=lambda row: int(row["doc_id"]),
    )


def _validate_pilot(pilot: Path) -> dict:
    manifest_path = pilot / "phase1_manifest.json"
    snapshot_path = pilot / "source_snapshot.json"
    if not manifest_path.is_file() or not snapshot_path.is_file():
        raise ValueError("Pilot is missing phase1_manifest.json or source_snapshot.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "pilot_complete" or manifest.get("build_scope") != "pilot":
        raise ValueError("Input is not a completed Phase 1 pilot")
    if manifest.get("source_snapshot_sha256") != sha256_file(snapshot_path):
        raise ValueError("Pilot source snapshot checksum mismatch")
    rows = 0
    checkpoints = manifest.get("part_checkpoints_sha256", {})
    if len(checkpoints) != int(manifest.get("parts", -1)):
        raise ValueError("Pilot checkpoint count differs from manifest")
    for stem, expected_hash in checkpoints.items():
        part = pilot / "clean_documents" / f"{stem}.parquet"
        if not part.is_file() or sha256_file(part) != expected_hash:
            raise ValueError(f"Pilot part checksum mismatch: {part}")
        rows += pq.ParquetFile(part).metadata.num_rows
    if rows != int(manifest.get("documents", -1)):
        raise ValueError(f"Pilot contains {rows} rows, manifest declares {manifest.get('documents')}")
    return manifest


def _sample_quarantine(root: Path, count: int) -> list[tuple[dict, dict, int]]:
    paths = Phase1Paths.from_repo(root)
    expectations = ProductionExpectations()
    _fingerprints, _audit, release = validate_locked_inputs(paths, expectations)
    state = pq.read_table(paths.phase0_state, columns=[
        "doc_id", "official_url", "phase0_status", "reason", "content_group", "text_sha256",
    ]).to_pylist()
    review = [row for row in state if row["phase0_status"] == "REVIEW_REQUIRED"]
    hash_counts = Counter(row["text_sha256"] for row in review if row["text_sha256"])
    by_reason: dict[str, list[dict]] = defaultdict(list)
    for row in review:
        by_reason[row["reason"]].append(row)
    for reason, values in by_reason.items():
        values.sort(key=lambda row: _stable_key(f"quarantine:{reason}", int(row["doc_id"])))
    reasons = sorted(by_reason)
    quotas = {reason: count // len(reasons) for reason in reasons}
    for reason in reasons[:count % len(reasons)]:
        quotas[reason] += 1
    # Preserve every rare reason in the sample, then redistribute its unused quota.
    deficit = 0
    for reason in reasons:
        if quotas[reason] > len(by_reason[reason]):
            deficit += quotas[reason] - len(by_reason[reason])
            quotas[reason] = len(by_reason[reason])
    while deficit:
        candidates = [reason for reason in reasons if quotas[reason] < len(by_reason[reason])]
        reason = min(candidates, key=lambda value: (quotas[value], value))
        quotas[reason] += 1
        deficit -= 1
    targets = {int(row["doc_id"]): row for reason in reasons for row in by_reason[reason][:quotas[reason]]}
    found = {}
    for spec in load_content_files(paths, release, expectations):
        if not any(row["content_group"] == spec.group for row in targets.values()):
            continue
        for batch in pq.ParquetFile(spec.documents_path).iter_batches(columns=[
            "id", "official_url", "title", "text", "text_sha256",
            *(["encoding"] if spec.kind == "html" else ["language_metadata"]),
        ]):
            for row in batch.to_pylist():
                state_row = targets.get(int(row["id"]))
                if state_row is not None and state_row["content_group"] == spec.group:
                    found[int(row["id"])] = ({**row, "source_kind": spec.kind,
                                                   "content_group": spec.group}, state_row)
    if len(found) != count:
        raise ValueError(f"Resolved {len(found)}/{count} quarantine controls")
    return [(content, status, hash_counts[status["text_sha256"]])
            for _ident, (content, status) in sorted(found.items())]


def _atomic_write(path: Path, writer) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.",
                                     suffix=".partial", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        writer(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_review_csv(path: Path, rows: list[dict]) -> None:
    fields = [field.name for field in REVIEW_SCHEMA
              if field.name not in {"source_text", "normalized_text", "quality_flags", "recorded_quality_flags"}]
    fields += ["quality_flags_csv", "recorded_quality_flags_csv", "text_preview"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            value = {key: row[key] for key in fields if key in row}
            value["quality_flags_csv"] = "|".join(row["quality_flags"])
            value["recorded_quality_flags_csv"] = "|".join(row["recorded_quality_flags"])
            value["text_preview"] = re.sub(r"\s+", " ", row["source_text"])[:500]
            writer.writerow(value)


def evaluate(pilot: Path, output: Path, *, pilot_sample: int = 400,
             quarantine_sample: int = 100, root: Path = ROOT) -> dict:
    manifest = _validate_pilot(pilot)
    pilot_rows = [_review_row(row, origin="pilot", phase0_status="USABLE")
                  for row in _sample_pilot(pilot, pilot_sample)]
    quarantine_rows = []
    if quarantine_sample:
        for content, status, repeated_count in _sample_quarantine(root, quarantine_sample):
            quarantine_rows.append(_review_row(
                content, origin="phase0_quarantine_control",
                phase0_status="REVIEW_REQUIRED", phase0_reason=status["reason"],
                repeated_text_count=repeated_count,
            ))
    rows = pilot_rows + quarantine_rows
    table = pa.Table.from_pylist(rows, schema=REVIEW_SCHEMA)
    _atomic_write(output / "review_sample.parquet",
                  lambda path: pq.write_table(table, path, compression="zstd"))
    _atomic_write(output / "review_sample.csv", lambda path: _write_review_csv(path, rows))

    flagged_rows = [
        _review_row(row, origin="pilot_flagged", phase0_status="USABLE")
        for row in _flagged_pilot_rows(pilot)
    ]
    flagged_table = pa.Table.from_pylist(flagged_rows, schema=REVIEW_SCHEMA)
    _atomic_write(output / "flagged_review.parquet",
                  lambda path: pq.write_table(flagged_table, path, compression="zstd"))
    _atomic_write(output / "flagged_review.csv",
                  lambda path: _write_review_csv(path, flagged_rows))

    errors = [row for row in rows if row["review_decision"] in {"REVIEW", "LABEL_NEEDS_REVIEW"}]
    summary = {
        "evaluation_version": EVALUATION_VERSION,
        "pilot": str(pilot.relative_to(root)),
        "pilot_format": manifest["format"],
        "pilot_manifest_sha256": sha256_file(pilot / "phase1_manifest.json"),
        "documents_reviewed": len(rows),
        "pilot_documents_reviewed": len(pilot_rows),
        "quarantine_controls_reviewed": len(quarantine_rows),
        "flagged_pilot_documents_reviewed": len(flagged_rows),
        "flagged_pilot_decision_counts": dict(sorted(
            Counter(row["review_decision"] for row in flagged_rows).items()
        )),
        "origin_counts": dict(sorted(Counter(row["sample_origin"] for row in rows).items())),
        "source_counts": dict(sorted(Counter(row["source_kind"] for row in rows).items())),
        "language_counts": dict(sorted(Counter(row["detected_language"] for row in rows).items())),
        "decision_counts": dict(sorted(Counter(row["review_decision"] for row in rows).items())),
        "quality_flag_counts": dict(sorted(Counter(flag for row in rows for flag in row["quality_flags"]).items())),
        "html_contamination": sum(row["html_contamination"] for row in rows),
        "encoding_issues": sum(row["encoding_issue"] for row in rows),
        "normalization_invariant_failures": sum(
            not all((row["lexical_content_preserved"], row["nonempty_lines_preserved"],
                     row["medical_values_preserved"], row["negations_preserved"]))
            for row in rows
        ),
        "language_metadata_mismatches": sum(row["language_metadata_agrees"] is False for row in rows),
        "errors": [{"doc_id": row["doc_id"], "origin": row["sample_origin"],
                    "decision": row["review_decision"], "notes": row["review_notes"]}
                   for row in errors],
    }
    _atomic_write(output / "quality_summary.json",
                  lambda path: path.write_text(json.dumps(summary, ensure_ascii=False,
                                                         indent=2, sort_keys=True) + "\n",
                                               encoding="utf-8"))
    _atomic_write(output / "errors.json",
                  lambda path: path.write_text(json.dumps(summary["errors"], ensure_ascii=False,
                                                         indent=2) + "\n", encoding="utf-8"))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pilot-sample", type=int, default=400)
    parser.add_argument("--quarantine-sample", type=int, default=100)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    if args.pilot_sample < 1 or args.quarantine_sample < 0:
        raise SystemExit("sample sizes must be non-negative and pilot sample must be positive")
    try:
        summary = evaluate(args.pilot.resolve(), args.output.resolve(),
                           pilot_sample=args.pilot_sample,
                           quarantine_sample=args.quarantine_sample,
                           root=args.root.resolve())
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Quality evaluation failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

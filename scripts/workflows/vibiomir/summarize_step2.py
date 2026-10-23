"""Write an honest, resumable coverage snapshot for ViBioMIR step 2."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/vibiomir_step2_20261013"


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def part_totals(folder: Path) -> tuple[int, int, Counter, int]:
    status, passages, parts = Counter(), 0, 0
    for path in folder.glob("part_*.json"):
        value = read(path)
        status.update(value["status_counts"])
        passages += value["passages"]
        parts += 1
    return sum(status.values()), passages, status, parts


def recovery_files(root: Path) -> list[tuple[Path, Path]]:
    single_doc = root / "documents.parquet"
    if single_doc.exists():
        return [(single_doc, root / "passages.parquet")]
    return [(path, root / "passages" / path.name)
            for path in sorted((root / "documents").glob("part_*.parquet"))]


def unique_recovery_totals(roots: list[Path]) -> dict:
    seen: set[int] = set()
    statuses = Counter()
    passages = 0
    duplicates = 0
    for root in roots:
        for documents_path, passages_path in recovery_files(root):
            rows = pq.read_table(documents_path, columns=["id", "status"]).to_pylist()
            document_ids = {row["id"] for row in rows}
            accepted = set()
            for row in rows:
                if row["id"] in seen:
                    duplicates += 1
                    continue
                seen.add(row["id"])
                accepted.add(row["id"])
                statuses[row["status"]] += 1
            if passages_path.exists():
                for ident in pq.read_table(passages_path, columns=["id"])["id"].to_pylist():
                    if ident not in document_ids:
                        raise ValueError(f"Passage without recovery document: {ident}")
                    passages += ident in accepted
    return {"documents": len(seen), "passages": passages,
            "status_counts": dict(statuses), "duplicate_rows_skipped": duplicates}


def main() -> None:
    ledger = read(BASE / "ledger_manifest.json")
    pubmed = read(BASE / "content/pubmed/manifest.json")
    need_extract = read(BASE / "content/needs_extract/progress.json")
    legacy_count, legacy_passages, legacy_status, legacy_parts = part_totals(
        BASE / "content/legacy_chunk_candidate/parts")
    baotayninh_fallback = read(BASE / "content/baotayninh_css_fallback/manifest.json")
    fallback_usable = (baotayninh_fallback or {}).get("status_counts", {}).get("usable", 0)
    fallback_passages = (baotayninh_fallback or {}).get("passages", 0)
    recovered_count, recovered_passages, recovered_status, recovered_parts = part_totals(
        BASE / "content/recovery_longchau/parts")
    healthy3_count, healthy3_passages, healthy3_status, healthy3_parts = part_totals(
        BASE / "content/recovery_healthy3/parts")
    misc_count, misc_passages, misc_status, misc_parts = part_totals(
        BASE / "content/recovery_misc_healthy/parts")
    misc_stable_count, misc_stable_passages, misc_stable_status, misc_stable_parts = part_totals(
        BASE / "content/recovery_misc_stable/parts")
    subs39_count, subs39_passages, subs39_status, subs39_parts = part_totals(
        BASE / "content/recovery_39subdomains/parts")
    first_pilot = read(BASE / "content/recovery_pilot/manifest.json")
    healthy_pilot = read(BASE / "content/recovery_healthy_pilot/manifest.json")
    misc_pilot = read(BASE / "content/recovery_misc_probe/manifest.json")
    subs39_pilot = read(BASE / "content/recovery_39subdomains_probe/manifest.json")
    partial_longchau = read(BASE / "content/recovery_longchau_partial_00000/manifest.json")
    partial_healthy = read(BASE / "content/recovery_healthy_partial_00000/manifest.json")
    if (BASE / "content/recovery_longchau/parts/part_00000.json").exists():
        partial_longchau = None
    if (BASE / "content/recovery_healthy_batches/parts/part_00000.json").exists():
        partial_healthy = None
    pilot_docs = sum((x or {}).get("fetched_success", 0)
                     for x in (first_pilot, healthy_pilot, misc_pilot, subs39_pilot,
                               partial_longchau, partial_healthy))
    pilot_usable = sum((x or {}).get("status_counts", {}).get("usable", 0)
                       for x in (first_pilot, healthy_pilot, misc_pilot, subs39_pilot,
                                 partial_longchau, partial_healthy))
    pilot_passages = sum((x or {}).get("passages", 0)
                         for x in (first_pilot, healthy_pilot, misc_pilot, subs39_pilot,
                                   partial_longchau, partial_healthy))
    recovery_roots = [
        BASE / "content/recovery_pilot",
        BASE / "content/recovery_healthy_pilot",
        BASE / "content/recovery_misc_probe",
        BASE / "content/recovery_39subdomains_probe",
    ]
    if partial_longchau:
        recovery_roots.append(BASE / "content/recovery_longchau_partial_00000")
    if partial_healthy:
        recovery_roots.append(BASE / "content/recovery_healthy_partial_00000")
    recovery_roots.extend(BASE / "content" / name for name in (
        "recovery_longchau", "recovery_healthy3", "recovery_misc_healthy",
        "recovery_misc_stable", "recovery_39subdomains"))
    recovery_unique = unique_recovery_totals(recovery_roots)
    usable = (pubmed["documents"] + need_extract["status_counts"].get("usable", 0)
              + legacy_status["usable"] + fallback_usable
              + recovery_unique["status_counts"].get("usable", 0))
    processed = (240000 + need_extract["processed_rows"] + legacy_count
                 + recovery_unique["documents"])
    total = ledger["rows"]
    report = {
        "status": "complete" if processed == total else "in_progress",
        "generated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "corpus_revision": ledger["corpus_revision"],
        "official_ids": total,
        "raw_or_content_processed_ids": processed,
        "remaining_ids_without_step2_content_decision": total - processed,
        "usable_materialized_ids": usable,
        "usable_materialized_fraction": round(usable / total, 6),
        "pubmed": {"input_ids": 240000, "usable": pubmed["documents"],
                   "passages": pubmed["passages"],
                   "metadata_only": ledger["states"]["pubmed_metadata_only"],
                   "unavailable": ledger["states"]["pubmed_unavailable"]},
        "legacy_html": {"planned": ledger["states"]["legacy_chunk_candidate"],
                        "processed": legacy_count, "parts": legacy_parts,
                        "passages": legacy_passages,
                        "status_counts": dict(legacy_status)},
        "baotayninh_css_fallback": baotayninh_fallback,
        "saved_raw_without_chunks": need_extract,
        "recovery_first_pilot": first_pilot,
        "recovery_healthy_pilot": healthy_pilot,
        "recovery_misc_probe": misc_pilot,
        "recovery_39subdomains_probe": subs39_pilot,
        "recovery_longchau_partial_batch0": partial_longchau,
        "recovery_healthy_partial_batch0": partial_healthy,
        "recovery_longchau": {"planned": 77975, "materialized": recovered_count,
                               "parts": recovered_parts, "passages": recovered_passages,
                               "status_counts": dict(recovered_status)},
        "recovery_healthy3": {"planned": 7653, "materialized": healthy3_count,
                              "parts": healthy3_parts, "passages": healthy3_passages,
                              "status_counts": dict(healthy3_status)},
        "recovery_misc_healthy": {"planned": 1025, "materialized": misc_count,
                                  "parts": misc_parts, "passages": misc_passages,
                                  "status_counts": dict(misc_status)},
        "recovery_misc_stable": {"planned": 69,
                                 "materialized": misc_stable_count,
                                 "parts": misc_stable_parts,
                                 "passages": misc_stable_passages,
                                 "status_counts": dict(misc_stable_status)},
        "recovery_39subdomains": {"planned": 7605,
                                  "materialized": subs39_count, "parts": subs39_parts,
                                  "passages": subs39_passages,
                                  "status_counts": dict(subs39_status)},
        "recovery_unique": recovery_unique,
        "total_passages": (pubmed["passages"] + need_extract["passages"]
                           + legacy_passages + fallback_passages
                           + recovery_unique["passages"]),
        "legacy_chunk_structural_audit": read(BASE / "legacy_chunk_quality.summary.json"),
        "quality_limit": "HTML usable means automated text extraction and exact offsets passed; page relevance, boilerplate, clinical accuracy, and language are not fully reviewed. Old chunks have only structural audit.",
    }
    atomic_json(BASE / "coverage_progress.json", report)
    with (BASE / "coverage_report.csv.partial").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["group", "official_ids", "processed_ids", "usable_ids", "passages"])
        writer.writerow(["PubMed", 240000, 240000, pubmed["documents"], pubmed["passages"]])
        writer.writerow(["Legacy HTML", ledger["states"]["legacy_chunk_candidate"],
                         legacy_count, legacy_status["usable"] + fallback_usable,
                         legacy_passages + fallback_passages])
        writer.writerow(["Saved raw", ledger["states"]["needs_extract"],
                         need_extract["processed_rows"], need_extract["status_counts"].get("usable", 0),
                         need_extract["passages"]])
        writer.writerow(["Missing raw recovery", ledger["states"]["needs_fetch"],
                         recovery_unique["documents"],
                         recovery_unique["status_counts"].get("usable", 0),
                         recovery_unique["passages"]])
        writer.writerow(["TOTAL", total, processed, usable, report["total_passages"]])
    os.replace(BASE / "coverage_report.csv.partial", BASE / "coverage_report.csv")
    print(json.dumps({k: report[k] for k in ("status", "official_ids",
                                          "raw_or_content_processed_ids",
                                          "remaining_ids_without_step2_content_decision",
                                          "usable_materialized_ids", "total_passages")}, indent=2))


if __name__ == "__main__":
    main()

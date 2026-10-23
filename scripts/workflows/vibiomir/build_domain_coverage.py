"""Build per-domain official-ID coverage from completed content checkpoints."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

from scripts.workflows.vibiomir.materialize_html_content import BASE


def read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def completed_part_documents(root: Path):
    for sidecar in sorted((root / "parts").glob("part_*.json")):
        path = root / "documents" / f"{sidecar.stem}.parquet"
        if path.exists():
            yield path


def main() -> None:
    ledger = read(BASE / "ledger_manifest.json")
    data = defaultdict(lambda: Counter())
    for domain, states in ledger["domains"].items():
        data[domain]["official_ids"] = sum(states.values())
        data[domain]["pubmed_final"] = sum(value for state, value in states.items()
                                             if state.startswith("pubmed_"))
        data[domain]["pubmed_usable"] = states.get("pubmed_usable", 0)

    duplicate_recovery_rows = 0

    def add(path: Path, group: str, seen: set[int] | None = None) -> None:
        nonlocal duplicate_recovery_rows
        for row in pq.read_table(path, columns=["id", "hostname", "status"]).to_pylist():
            if seen is not None:
                if row["id"] in seen:
                    duplicate_recovery_rows += 1
                    continue
                seen.add(row["id"])
            counts = data[row["hostname"]]
            counts[group + "_processed"] += 1
            counts[group + "_usable"] += row["status"] == "usable"
            counts[group + "_review"] += row["status"] != "usable"

    for path in completed_part_documents(BASE / "content/legacy_chunk_candidate"):
        add(path, "legacy")
    for path in completed_part_documents(BASE / "content/needs_extract"):
        add(path, "saved_raw")

    fallback = read(BASE / "content/baotayninh_css_fallback/manifest.json")
    if fallback:
        if fallback["prior_status"] != "quality_review_empty":
            raise ValueError("Unknown fallback prior status")
        domain = fallback["domain"]
        rescued = fallback["status_counts"].get("usable", 0)
        if data[domain]["legacy_review"] < rescued:
            raise ValueError("Fallback exceeds prior review count")
        data[domain]["legacy_usable"] += rescued
        data[domain]["legacy_review"] -= rescued

    seen_recovery = set()
    for name in ("recovery_pilot", "recovery_healthy_pilot", "recovery_misc_probe",
                 "recovery_39subdomains_probe"):
        root = BASE / "content" / name
        if (root / "manifest.json").exists():
            add(root / "documents.parquet", "recovery", seen_recovery)
    partials = (
        ("recovery_longchau_partial_00000", "recovery_longchau"),
        ("recovery_healthy_partial_00000", "recovery_healthy_batches"),
    )
    for name, full in partials:
        root = BASE / "content" / name
        if (root / "manifest.json").exists() and not (
                BASE / "content" / full / "parts/part_00000.json").exists():
            add(root / "documents.parquet", "recovery", seen_recovery)
    for name in ("recovery_longchau", "recovery_healthy_batches",
                 "recovery_healthy3", "recovery_misc_healthy",
                 "recovery_misc_stable", "recovery_39subdomains"):
        for path in completed_part_documents(BASE / "content" / name):
            add(path, "recovery", seen_recovery)

    fields = ["domain", "official_ids", "pubmed_final", "pubmed_usable",
              "legacy_processed", "legacy_usable", "legacy_review",
              "saved_raw_processed", "saved_raw_usable", "saved_raw_review",
              "recovery_processed", "recovery_usable", "recovery_review",
              "total_processed", "total_usable", "pending_content"]
    destination = BASE / "coverage_by_domain.csv"
    temporary = destination.with_suffix(".csv.partial")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for domain in sorted(data):
            counts = data[domain]
            processed = sum(counts[name] for name in ("pubmed_final", "legacy_processed",
                                                      "saved_raw_processed", "recovery_processed"))
            usable = sum(counts[name] for name in ("pubmed_usable", "legacy_usable",
                                                   "saved_raw_usable", "recovery_usable"))
            if processed > counts["official_ids"] or usable > processed:
                raise ValueError(f"Invalid official coverage for {domain}")
            writer.writerow({"domain": domain,
                             **{name: counts[name] for name in fields[1:-3]},
                             "total_processed": processed, "total_usable": usable,
                             "pending_content": counts["official_ids"] - processed})
    os.replace(temporary, destination)
    print(json.dumps({"domains": len(data), "recovery_ids": len(seen_recovery),
                      "duplicate_recovery_rows_skipped": duplicate_recovery_rows,
                      "output": str(destination)}, indent=2))


if __name__ == "__main__":
    main()

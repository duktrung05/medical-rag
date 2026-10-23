"""Create the Phase 3 full-run gate from already accepted pilot artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase3.canonical import CANONICAL_POLICY_VERSION
from src.medrag.data.phase3.disk_lsh import DISK_LSH_VERSION, SHINGLING_VERSION
from src.medrag.data.phase3.exact import GROUPING_POLICY_VERSION
from src.medrag.data.phase3.near import NEAR_POLICY_VERSION
from src.medrag.data.phase3.overlap import OVERLAP_POLICY_VERSION
from src.medrag.data.phase3.passage import PASSAGE_GROUPING_POLICY_VERSION


PILOTS = {
    "task2": ROOT / "outputs/vibiomir_phase3_20261015_task2_dry_run_v2",
    "task3": ROOT / "outputs/vibiomir_phase3_20261015_task3_dry_run_v1",
    "task4": ROOT / "outputs/vibiomir_phase3_20261015_task4_dry_run_v2",
    "task5": ROOT / "outputs/vibiomir_phase3_20261015_task5_pilot_v6",
    "task6": ROOT / "outputs/vibiomir_phase3_20261015_task6_pilot_v1",
}


VALIDATIONS = {
    "task2": "validation/task2_exact_document_validation.json",
    "task3": "validation/task3_near_document_validation.json",
    "task4": "validation/task4_passage_content_validation.json",
    "task5": "validation/overlap_validation_report.json",
    "task6": "validation/task6_canonical_validation.json",
}


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected object: {path}")
    return value


def create(test_results: Path) -> dict:
    tests = _load(test_results)
    validations = {name: _load(root / VALIDATIONS[name]) for name, root in PILOTS.items()}
    checks = {
        "all_pilot_validations_pass": all(
            validation.get("status") == "pass" for validation in validations.values()
        ),
        "task5_auto_merge_disabled": not validations["task5"]["checks"]["auto_merge_allowed"],
        "task5_labeled_false_positive_rate_zero": (
            validations["task5"]["evaluation"]["false_positive_rate"] == 0.0
        ),
        "task6_full_corpus_not_started": not validations["task6"]["full_corpus_started"],
        "focused_tests_pass": tests.get("status") == "pass" and tests.get("failed") == 0,
        "scalability_tests_present": tests.get("scalability_tests", 0) > 0,
    }
    status = "accepted" if all(checks.values()) else "not_accepted"
    payload = {
        "format": "vibiomir-phase3-pilot-acceptance-v1",
        "phase": "phase3_pilot",
        "status": status,
        "accepted_scope": "Task 2-6 bounded pilots plus scalability-focused synthetic regression",
        "checks": checks,
        "input_fingerprints": {
            "phase1_release_manifest": sha256_file(
                ROOT / "outputs/vibiomir_phase1_full_20261014_v1/release/phase1_release_manifest.json"
            ),
            "phase2_release_manifest": sha256_file(
                ROOT / "outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json"
            ),
            "phase3_task1_audit": sha256_file(
                ROOT / "outputs/vibiomir_phase3_20261015_task1/reports/PHASE3_DATA_AUDIT.md"
            ),
        },
        "module_versions": {
            "exact_document": GROUPING_POLICY_VERSION,
            "near_document": NEAR_POLICY_VERSION,
            "passage_content": PASSAGE_GROUPING_POLICY_VERSION,
            "passage_overlap": OVERLAP_POLICY_VERSION,
            "canonical_selection": CANONICAL_POLICY_VERSION,
            "disk_lsh": DISK_LSH_VERSION,
            "shingling": SHINGLING_VERSION,
        },
        "config_hashes": {
            "task5_pilot": sha256_file(ROOT / "configs/vibiomir/phase3_task5_pilot_v1.json"),
        },
        "test_results": tests,
        "pilot_artifact_refs": {
            name: {
                "root": str(root.relative_to(ROOT)),
                "validation": VALIDATIONS[name],
                "validation_sha256": sha256_file(root / VALIDATIONS[name]),
            }
            for name, root in PILOTS.items()
        },
        "limitations": [
            "Acceptance does not establish exhaustive MinHash/LSH candidate recall.",
            "Hot buckets filtered by the full runner must be reported as uncovered candidate space.",
            "Near-document and passage-overlap relations remain review-only and never auto-merge.",
        ],
    }
    if status != "accepted":
        failed = sorted(key for key, value in checks.items() if not value)
        raise ValueError(f"Acceptance requirements missing: {failed}")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = create(args.test_results.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=args.output.parent, suffix=".partial", delete=False
    ) as stream:
        temporary = Path(stream.name)
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, args.output)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

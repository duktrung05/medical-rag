from pathlib import Path

from src.medrag.data.phase1.selection import (
    Phase1Paths,
    ProductionExpectations,
    load_selection,
    validate_locked_inputs,
)


ROOT = Path(__file__).resolve().parents[3]


def test_production_phase0_allowlist_is_exact_and_excludes_quarantine():
    paths = Phase1Paths.from_repo(ROOT)
    expected = ProductionExpectations()
    fingerprints, audit, _release = validate_locked_inputs(paths, expected)
    selection = load_selection(paths, expected, fingerprints)
    assert len(selection.documents) == 1_075_599
    assert selection.status_counts["REVIEW_REQUIRED"] == 9_814
    assert not expected.staging_ids.intersection(selection.documents)
    assert audit["staged_recovery_merged"] is False


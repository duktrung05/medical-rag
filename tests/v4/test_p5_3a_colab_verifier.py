"""Regression coverage for restoring frozen P5.3A pairs before score import."""
from __future__ import annotations

from scripts.v4.run_p5_2_split import PILOT_IDS, external_lock, verify_pair_manifest
from scripts.v4.verify_p5_3a_colab_canary import SOURCE, materialize_pair_workspace


def test_materializes_all_pair_shards_before_manifest_validation(tmp_path):
    lock = external_lock(SOURCE, "pilot-live")
    for name in ("lock.json", "pair_manifest.json", "key_contract.sqlite"):
        (tmp_path / name).write_bytes((SOURCE / name).read_bytes())

    restored = materialize_pair_workspace(SOURCE, tmp_path, lock)

    assert tuple(restored) == PILOT_IDS
    assert sum(len(shard["pairs"]) for shard in restored.values()) == 1200
    assert {path.stem for path in (tmp_path / "pairs").glob("*.json")} == {
        str(qid) for qid in PILOT_IDS
    }
    verify_pair_manifest(tmp_path, lock)

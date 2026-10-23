from __future__ import annotations

import hashlib

import pytest

from scripts.workflows.vibiomir.phase3_scalable_candidates import (
    _document_pair_rows,
    _passage_pair_rows,
)
from src.medrag.data.phase3.disk_lsh import (
    DiskLSHIndex,
    ResourceBudgetExceeded,
    ResourceMonitor,
    deterministic_partition,
)


def _record(key: str, signature: tuple[int, ...], *, parent: str | None = None) -> dict:
    return {
        "entity_key": key,
        "parent_key": parent,
        "sequence_index": 0,
        "ordinal": int(key.removeprefix("r")),
        "strategy": "english_word",
        "shingle_size": 5,
        "shingles": frozenset({1, 2, int(key.removeprefix("r")) + 10}),
        "signature": signature,
        "normalized_hash": hashlib.sha256(key.encode()).hexdigest(),
        "payload": {"text": key, "doc_id": int(key.removeprefix("r"))},
    }


def _index(tmp_path, *, maximum_bucket_size=10, partitions=4):
    monitor = ResourceMonitor(
        rss_limit_bytes=16 * 1024**3,
        disk_limit_bytes=1024**3,
        work_dir=tmp_path,
    )
    return DiskLSHIndex(
        tmp_path / "lsh.sqlite", bands=2, rows_per_band=2,
        partitions=partitions, maximum_bucket_size=maximum_bucket_size,
        config_hash="config", input_fingerprint="input", monitor=monitor,
    )


def test_cross_shard_candidates_are_discovered_and_partitioned(tmp_path):
    with _index(tmp_path) as index:
        index.ingest_shard("part-a", [_record("r1", (1, 2, 3, 4))])
        index.ingest_shard("part-b", [_record("r2", (1, 2, 8, 9))])
        stats = index.generate_candidates()
        assert stats["deduplicated_candidates"] == 1
        partitions = list(index.candidate_partitions())
        assert partitions == [deterministic_partition("r1", "r2", 4)]
        pairs = list(index.iter_candidates(partitions[0]))
        assert [(pair[0]["entity_key"], pair[1]["entity_key"]) for pair in pairs] == [("r1", "r2")]


def test_candidate_pairs_are_deduplicated_across_bands(tmp_path):
    with _index(tmp_path) as index:
        index.ingest_shard("a", [_record("r1", (1, 2, 3, 4))])
        index.ingest_shard("b", [_record("r2", (1, 2, 3, 4))])
        stats = index.generate_candidates()
        assert stats["examined_pair_occurrences"] == 2
        assert stats["deduplicated_candidates"] == 1


def test_hot_buckets_are_accounted_and_never_silently_truncated(tmp_path):
    with _index(tmp_path, maximum_bucket_size=2) as index:
        index.ingest_shard("a", [_record(f"r{i}", (1, 2, i, i)) for i in range(1, 4)])
        stats = index.generate_candidates()
        assert stats["hot_buckets"] == 1
        assert index.statistics()["hot_bucket_potential_pairs"] == 3
        row = index.connection.execute(
            "SELECT bucket_size, potential_pairs, examined_pairs, policy FROM hot_buckets"
        ).fetchone()
        assert row == (3, 3, 0, "high_frequency_template_filtered")


def test_completed_shard_resume_does_not_duplicate_or_reprocess(tmp_path):
    records = [_record("r1", (1, 2, 3, 4))]
    with _index(tmp_path) as index:
        assert index.ingest_shard("a", records) == 1
    with _index(tmp_path) as resumed:
        assert resumed.ingest_shard("a", []) == 1
        assert resumed.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 1


def test_interrupted_shard_resume_is_idempotent(tmp_path):
    def interrupted():
        yield _record("r1", (1, 2, 3, 4))
        raise RuntimeError("synthetic interruption")

    with _index(tmp_path) as index:
        with pytest.raises(RuntimeError, match="interruption"):
            index.ingest_shard("a", interrupted(), batch_size=1)
        assert not index.checkpoint_complete("input_shard", "a")
    with _index(tmp_path) as resumed:
        assert resumed.ingest_shard(
            "a", [_record("r1", (1, 2, 3, 4)), _record("r2", (1, 2, 8, 9))],
            batch_size=1,
        ) == 2
        assert resumed.connection.execute("SELECT COUNT(*) FROM records").fetchone()[0] == 2


def test_completed_candidate_generation_resume_is_idempotent(tmp_path):
    with _index(tmp_path) as index:
        index.ingest_shard("a", [_record("r1", (1, 2, 3, 4))])
        index.ingest_shard("b", [_record("r2", (1, 2, 3, 4))])
        first = index.generate_candidates()
    with _index(tmp_path) as resumed:
        second = resumed.generate_candidates()
        assert second["deduplicated_candidates"] == first["deduplicated_candidates"] == 1
        assert resumed.connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1


def test_output_is_deterministic_across_shard_ingestion_order(tmp_path):
    results = []
    for name, order in (("forward", (1, 2, 3)), ("reverse", (3, 2, 1))):
        root = tmp_path / name
        with _index(root, partitions=8) as index:
            for value in order:
                index.ingest_shard(f"part-{value}", [_record(f"r{value}", (1, 2, 3, 4))])
            index.generate_candidates()
            results.append(sorted(
                (left["entity_key"], right["entity_key"])
                for partition in index.candidate_partitions()
                for left, right, _ in index.iter_candidates(partition)
            ))
    assert results[0] == results[1]


def test_disk_full_mode_candidates_match_in_memory_pilot_mode(tmp_path):
    from src.medrag.data.phase3.near import lsh_candidates

    records = [_record("r1", (1, 2, 3, 4)), _record("r2", (1, 2, 8, 9)),
               _record("r3", (7, 7, 3, 4))]
    pilot_records = [{**row, "shingle_strategy": row["strategy"]} for row in records]
    in_memory, overflow = lsh_candidates(
        pilot_records, bands=2, rows_per_band=2, maximum_bucket_size=10
    )
    expected = {
        tuple(sorted((records[left]["entity_key"], records[right]["entity_key"])))
        for left, right in in_memory
    }
    assert overflow == 0
    with _index(tmp_path) as index:
        index.ingest_shard("a", records[:2])
        index.ingest_shard("b", records[2:])
        index.generate_candidates()
        actual = {
            tuple(sorted((left["entity_key"], right["entity_key"])))
            for partition in index.candidate_partitions()
            for left, right, _ in index.iter_candidates(partition)
        }
    assert actual == expected


def test_memory_budget_fails_closed(tmp_path):
    monitor = ResourceMonitor(rss_limit_bytes=1, disk_limit_bytes=1024**3, work_dir=tmp_path)
    with pytest.raises(ResourceBudgetExceeded, match="RSS budget"):
        monitor.preflight()


def _custom_record(key: str, text: str, payload: dict) -> dict:
    from src.medrag.data.phase3.near import hash_shingles, minhash_signature, shingle_text

    shingles, strategy, size = shingle_text(text, "en")
    hashed = hash_shingles(shingles)
    return {
        "entity_key": key, "parent_key": str(payload["doc_id"]),
        "sequence_index": payload.get("passage_index", 0), "ordinal": 0,
        "strategy": strategy, "shingle_size": size, "shingles": hashed,
        "signature": minhash_signature(hashed, 4),
        "normalized_hash": hashlib.sha256(text.encode()).hexdigest(),
        "payload": payload,
    }


def test_known_near_pair_survives_disk_backed_verification(tmp_path):
    left_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda"
    right_text = "alpha beta gamma delta epsilon zeta eta theta iota kappa changed"
    left = _custom_record("1", left_text, {
        "doc_id": 1, "official_url": "https://example.test/1",
        "normalized_text": left_text, "detected_language": "en",
    })
    right = _custom_record("2", right_text, {
        "doc_id": 2, "official_url": "https://example.test/2",
        "normalized_text": right_text, "detected_language": "en",
    })
    with _index(tmp_path, partitions=1) as index:
        index.ingest_shard("left", [left])
        index.ingest_shard("right", [right])
        left_id, right_id = [row[0] for row in index.connection.execute(
            "SELECT record_id FROM records ORDER BY entity_key"
        )]
        index.add_candidate(left_id, right_id, "known_regression")
        index.connection.commit()
        config = {
            "thresholds": [0.5, 0.8], "acceptance_threshold": 0.5,
            "containment_threshold": 0.95, "containment_jaccard_floor": 0.5,
        }
        rows = list(_document_pair_rows(index, 0, config, "release"))
        assert [(row["left_doc_id"], row["right_doc_id"]) for row in rows] == [(1, 2)]
        assert rows[0]["auto_merge_allowed"] is False


def test_medical_dosage_mismatch_is_never_promoted_to_document_pair(tmp_path):
    base = "patient treatment dose is {} daily alpha beta gamma delta epsilon zeta eta"
    left_text, right_text = base.format("5 mg"), base.format("10 mg")
    records = [
        _custom_record("1", left_text, {"doc_id": 1, "official_url": "u1",
                                       "normalized_text": left_text, "detected_language": "en"}),
        _custom_record("2", right_text, {"doc_id": 2, "official_url": "u2",
                                        "normalized_text": right_text, "detected_language": "en"}),
    ]
    with _index(tmp_path, partitions=1) as index:
        index.ingest_shard("both", records)
        ids = [row[0] for row in index.connection.execute("SELECT record_id FROM records ORDER BY entity_key")]
        index.add_candidate(*ids, "known_medical_guard")
        index.connection.commit()
        config = {
            "thresholds": [0.1], "acceptance_threshold": 0.1,
            "containment_threshold": 0.1, "containment_jaccard_floor": 0.0,
        }
        assert list(_document_pair_rows(index, 0, config, "release")) == []


def test_passage_medical_guard_remains_review_only(tmp_path):
    left_text = "treatment dose is 5 mg daily alpha beta gamma delta epsilon zeta"
    right_text = "treatment dose is 10 mg daily alpha beta gamma delta epsilon zeta"
    records = [
        _custom_record("p1", left_text, {
            "passage_id": "p1", "doc_id": 1, "official_url": "u1", "passage_index": 0,
            "passage_text": left_text, "char_start": 0, "char_end": len(left_text),
            "source_text_sha256": "a" * 64, "language": "en",
        }),
        _custom_record("p2", right_text, {
            "passage_id": "p2", "doc_id": 2, "official_url": "u2", "passage_index": 0,
            "passage_text": right_text, "char_start": 0, "char_end": len(right_text),
            "source_text_sha256": "b" * 64, "language": "en",
        }),
    ]
    with _index(tmp_path, partitions=1) as index:
        index.ingest_shard("both", records)
        ids = [row[0] for row in index.connection.execute("SELECT record_id FROM records ORDER BY entity_key")]
        index.add_candidate(*ids, "known_medical_guard")
        index.connection.commit()
        config = {
            "minimum_shared_shingles": 1, "jaccard_threshold": 0.1,
            "containment_threshold": 0.1, "containment_jaccard_floor": 0.0,
        }
        rows = list(_passage_pair_rows(index, 0, config))
        assert len(rows) == 1
        assert "drug_dosage_mismatch" in rows[0]["guard_reasons"]
        assert rows[0]["review_status"] == "medical_guard_review"
        assert rows[0]["auto_merge_allowed"] is False

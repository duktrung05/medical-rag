from __future__ import annotations

import hashlib

import pytest

from scripts.workflows.vibiomir.build_phase3_passage_content_groups import (
    ROOT,
    _count_parts,
    _open_index,
    _passage_hashes,
    _safe_output,
    _select_parts,
    _verify_full_gate,
)
from src.medrag.data.phase1.selection import sha256_file
from src.medrag.data.phase2.schema import PASSAGE_SCHEMA
from src.medrag.data.phase3.passage import (
    PASSAGE_GROUPING_POLICY_VERSION,
    passage_content_group_id,
    passage_group_row,
    verify_group_content,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _passage(passage_id: str, doc_id: int, text: str, start: int = 10) -> dict:
    return {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "official_url": f"https://example.test/{doc_id}",
        "passage_index": 0,
        "passage_text": text,
        "char_start": start,
        "char_end": start + len(text),
        "source_text_sha256": _sha(f"source-{doc_id}"),
        "text_representation_sha256": _sha(f"representation-{doc_id}"),
        "chunker_version": "chunker-v2",
    }


def _row(passage: dict, *, raw_size: int = 1, normalized_size: int = 1,
         document_count: int = 1) -> dict:
    raw_hash, normalized_hash, _ = _passage_hashes(passage["passage_text"])
    return passage_group_row(
        passage,
        content_sha256=raw_hash,
        normalized_content_hash=normalized_hash,
        raw_group_size=raw_size,
        content_group_size=normalized_size,
        distinct_document_count=document_count,
        normalization_version="normalization-v1",
        phase2_release_id=_sha("phase2"),
        grouping_scope="test",
    )


def test_every_source_occurrence_remains_a_separate_output_row():
    passages = [_passage("p1", 10, "Same text"), _passage("p2", 20, "Same text")]
    rows = [_row(value, raw_size=2, normalized_size=2, document_count=2)
            for value in passages]
    assert [row["passage_id"] for row in rows] == ["p1", "p2"]
    assert [row["doc_id"] for row in rows] == [10, 20]
    assert len({row["passage_content_group_id"] for row in rows}) == 1


def test_normalized_exact_passages_share_content_identity_but_keep_raw_identity():
    left = _passage("p1", 10, "Diabetes   is chronic.")
    right = _passage("p2", 20, "Diabetes is chronic.")
    left_row = _row(left, normalized_size=2, document_count=2)
    right_row = _row(right, normalized_size=2, document_count=2)
    assert left_row["content_sha256"] != right_row["content_sha256"]
    assert left_row["raw_content_group_id"] != right_row["raw_content_group_id"]
    assert left_row["normalized_content_hash"] == right_row["normalized_content_hash"]
    assert left_row["passage_content_group_id"] == right_row["passage_content_group_id"]


def test_near_duplicate_text_is_not_grouped_as_content_identity():
    left = _row(_passage("p1", 10, "Dose is 5 mg daily."))
    right = _row(_passage("p2", 20, "Dose is 10 mg daily."))
    assert left["passage_content_group_id"] != right["passage_content_group_id"]


def test_source_identity_and_offsets_are_preserved():
    passage = _passage("p-source", 42, "Preserved content", start=123)
    row = _row(passage)
    assert row["passage_id"] == passage["passage_id"]
    assert row["doc_id"] == passage["doc_id"]
    assert row["official_url"] == passage["official_url"]
    assert row["source_offsets"] == {
        "char_start": passage["char_start"],
        "char_end": passage["char_end"],
    }
    assert row["source_text_sha256"] == passage["source_text_sha256"]
    assert row["chunker_version"] == passage["chunker_version"]


def test_grouping_is_deterministic():
    passage = _passage("p1", 1, "Stable content")
    first = _row(passage)
    second = _row(passage)
    assert first == second
    assert first["passage_content_group_id"] == passage_content_group_id(
        first["normalized_content_hash"]
    )
    assert first["grouping_policy_version"] == PASSAGE_GROUPING_POLICY_VERSION


def test_duplicate_hash_candidates_receive_exact_string_verification():
    digest = bytes.fromhex(_sha("same"))
    baselines = {}
    verify_group_content(
        baselines, digest, passage_id="p1", text="same", label="passage"
    )
    verify_group_content(
        baselines, digest, passage_id="p2", text="same", label="passage"
    )
    with pytest.raises(ValueError, match="collision or corrupt"):
        verify_group_content(
            baselines, digest, passage_id="p3", text="different", label="passage"
        )


def test_full_build_requires_acceptance_and_dry_run_is_bounded():
    entries = [{"part": "part-00000"}, {"part": "part-00001"}]
    assert _select_parts(entries, 1, False) == [entries[0]]
    with pytest.raises(ValueError, match="fewer than all"):
        _select_parts(entries, 2, False)
    with pytest.raises(ValueError, match="pilot-acceptance"):
        _verify_full_gate(None)


def test_output_must_use_a_dedicated_phase3_namespace():
    output = ROOT / "outputs/vibiomir_phase3_task4_test"
    assert _safe_output(output, []) == output.resolve()
    with pytest.raises(ValueError, match="phase3"):
        _safe_output(ROOT / "outputs/not_phase3", [])


def test_checkpoint_resume_does_not_count_a_shard_twice(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    passages_root = tmp_path / "passages"
    passages_root.mkdir()
    source = _passage("p-resume", 7, "Resume-safe passage", start=0)
    full_row = {
        **source,
        "token_count": 3,
        "language": "en",
        "section_id": "section-0000",
        "section_heading": None,
        "strategy_id": "C",
        "tokenizer_name": "tokenizer",
        "tokenizer_revision": "revision",
        "phase1_release_id": _sha("phase1"),
        "corpus_revision": "corpus",
        "processing_run_id": "phase2",
        "text_representation": "normalized_text",
    }
    part_path = passages_root / "part-00000.parquet"
    pq.write_table(pa.Table.from_pylist([full_row], schema=PASSAGE_SCHEMA), part_path)
    entry = {
        "part": "part-00000",
        "rows": 1,
        "sha256": sha256_file(part_path),
    }
    checkpoint = tmp_path / "checkpoint.json"
    connection = _open_index(tmp_path / "groups.sqlite")
    try:
        assert _count_parts(
            connection, passages_root, [entry], checkpoint, scope="test"
        ) == 1
        assert _count_parts(
            connection, passages_root, [entry], checkpoint, scope="test"
        ) == 1
        assert connection.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0] == 1
    finally:
        connection.close()

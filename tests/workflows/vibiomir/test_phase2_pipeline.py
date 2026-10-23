from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from scripts.workflows.vibiomir.build_phase2_passages import (
    EXPECTED_HELD,
    EXPECTED_PASS,
    _atomic_json,
    _checkpoint_valid,
    ensure_output_is_safe,
    strategies_from_frozen_config,
)
from scripts.workflows.vibiomir.evaluate_phase2_retrieval import _evaluate_strategy
from scripts.workflows.vibiomir.review_phase2_pilot import apply_annotations
from scripts.workflows.vibiomir.stage_phase2_semantic_eligibility import materialize
from src.medrag.data.phase2.chunker import _repair_short_section_passages, chunk_document
from src.medrag.data.phase2.provenance import passage_id, sha256_text
from src.medrag.data.phase2.schema import CHUNKER_VERSION, STRATEGIES, ChunkingStrategy
from src.medrag.data.phase2.segmentation import TextSpan
from src.medrag.data.phase2.structure import SectionSpan, find_sections
from src.medrag.data.phase2.tokenizer import BgeM3Tokenizer, default_tokenizer_path
from src.medrag.data.phase2.validation import covered_non_whitespace, validate_document_passages


ROOT = Path(__file__).resolve().parents[3]
CANDIDATE = ROOT / "outputs/vibiomir_phase1_full_20261014_v1"


@pytest.fixture(scope="module")
def tokenizer() -> BgeM3Tokenizer:
    return BgeM3Tokenizer(default_tokenizer_path(ROOT))


def _row(text: str, *, doc_id: int = 42, language: str = "vi", title: str | None = None) -> dict:
    return {
        "doc_id": doc_id,
        "official_url": f"https://example.test/{doc_id}",
        "title": title,
        "normalized_text": text,
        "normalized_text_sha256": sha256_text(text),
        "source_text_sha256": sha256_text(text),
        "detected_language": language,
    }


def _chunk(text: str, tokenizer: BgeM3Tokenizer, strategy=STRATEGIES[0], **kwargs):
    row = _row(text, **kwargs)
    passages = chunk_document(
        row, tokenizer, strategy, phase1_release_id="release", corpus_revision="corpus",
        processing_run_id="test",
    )
    return row, passages


def test_phase1_release_integrity():
    manifest = json.loads((CANDIDATE / "phase1_manifest.json").read_text())
    release = json.loads((CANDIDATE / "release/phase1_release_manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["build_scope"] == "full"
    assert release["status"] == "complete"
    assert len(manifest["part_checkpoints_sha256"]) == 922


def test_pass_allowlist_count_1073531():
    release = json.loads((CANDIDATE / "release/phase1_release_manifest.json").read_text())
    assert release["pass_documents"] == EXPECTED_PASS == 1_073_531
    assert sum(part["rows"] for part in release["pass_allowlist_parts"]) == EXPECTED_PASS


def test_quarantined_and_review_excluded():
    release = json.loads((CANDIDATE / "release/phase1_release_manifest.json").read_text())
    assert release["held_documents"] == EXPECTED_HELD
    assert release["status_counts"] == {
        "PASS": EXPECTED_PASS, "QUARANTINED": 887, "REVIEW_REQUIRED": 1_181,
    }


def test_official_doc_id_url_preserved(tokenizer):
    row, passages = _chunk("Nội dung y khoa nguyên bản.", tokenizer, doc_id=987)
    assert {(value["doc_id"], value["official_url"]) for value in passages} == {
        (row["doc_id"], row["official_url"])
    }


def test_passage_id_deterministic():
    args = dict(chunker_version=CHUNKER_VERSION, strategy_id="A", doc_id=1,
                passage_index=0, source_sha256="a" * 64, char_start=0, char_end=10)
    assert passage_id(**args) == passage_id(**args)
    assert passage_id(**args) != passage_id(**{**args, "strategy_id": "B"})


def test_passage_ids_unique(tokenizer):
    _, passages = _chunk(("Một câu y khoa đầy đủ. " * 100).strip(), tokenizer)
    ids = [row["passage_id"] for row in passages]
    assert len(ids) == len(set(ids))


def test_passage_source_slice_exact(tokenizer):
    row, passages = _chunk("Đoạn một.\n\nĐoạn hai có nội dung.", tokenizer)
    assert all(row["normalized_text"][p["char_start"]:p["char_end"]] == p["passage_text"]
               for p in passages)


def test_text_representation_correct(tokenizer):
    row, passages = _chunk("Exact normalized text.", tokenizer, language="en")
    assert all(p["text_representation"] == "normalized_text" for p in passages)
    assert all(p["text_representation_sha256"] == row["normalized_text_sha256"]
               for p in passages)


def test_token_count_exact(tokenizer):
    _, passages = _chunk("糖尿病患者应定期监测血糖水平。", tokenizer, language="zh")
    assert all(p["token_count"] == tokenizer.count(p["passage_text"]) for p in passages)


def test_max_tokens_enforced(tokenizer):
    strategy = ChunkingStrategy("tiny", 16, 20, 3, 4)
    _, passages = _chunk(("renal impairment treatment " * 100).strip(), tokenizer, strategy)
    assert passages
    assert max(p["token_count"] for p in passages) <= 20


def test_document_content_coverage(tokenizer):
    row, passages = _chunk("Một.\n\nHai.\n\nBa.", tokenizer)
    assert covered_non_whitespace(row["normalized_text"], passages) == (11, 11)


def test_overlap_policy(tokenizer):
    strategy = ChunkingStrategy("overlap", 24, 32, 4, 8)
    _, passages = _chunk("\n\n".join(f"Sentence number {i} has context." for i in range(20)),
                         tokenizer, strategy)
    for left, right in zip(passages, passages[1:]):
        if left["section_id"] != right["section_id"]:
            continue
        overlap_start = right["char_start"]
        overlap_end = min(left["char_end"], right["char_end"])
        if overlap_end > overlap_start:
            assert tokenizer.count(left["passage_text"][overlap_start-left["char_start"]:
                                                        overlap_end-left["char_start"]]) <= 8


def test_heading_boundaries(tokenizer):
    text = "TỔNG QUAN\n\nNội dung đầu.\n\nĐIỀU TRỊ\n\nNội dung sau."
    sections = find_sections(text)
    assert [section.heading for section in sections] == ["TỔNG QUAN", "ĐIỀU TRỊ"]
    _, passages = _chunk(text, tokenizer)
    assert {p["section_heading"] for p in passages} == {"TỔNG QUAN", "ĐIỀU TRỊ"}


def test_sentence_boundaries(tokenizer):
    strategy = ChunkingStrategy("sentence", 12, 16, 3, 0)
    _, passages = _chunk("First clinical sentence. Second clinical sentence. Third one.",
                         tokenizer, strategy, language="en")
    assert all(p["passage_text"].strip() for p in passages)
    assert all(p["passage_text"][-1] in ".!?" or p["token_count"] == 16 for p in passages)


def test_decimal_and_dotted_values_are_not_sentence_boundaries(tokenizer):
    strategy = ChunkingStrategy("decimal", 12, 18, 3, 0)
    text = (
        "The cohort included 5.963 adults with diabetes. "
        "Their mean result was 16.12/mm3 and the total was 3.272.256 records. "
        "No participant was excluded from analysis."
    )
    _, passages = _chunk(text, tokenizer, strategy, language="en")
    starts = [passage["passage_text"].lstrip() for passage in passages]
    assert not any(value.startswith(("963", "12/mm3", "272.256")) for value in starts)
    joined = " ".join(passage["passage_text"] for passage in passages)
    assert all(value in joined for value in ("5.963", "16.12/mm3", "3.272.256"))


def test_question_and_labeled_answer_stay_together(tokenizer):
    strategy = ChunkingStrategy("qa", 20, 36, 3, 0)
    text = (
        "Background details occupy the opening sentence here. "
        "Which diagnosis best explains these findings? "
        "Answer: acute appendicitis is the confirmed diagnosis. "
        "Follow-up treatment was uncomplicated and successful."
    )
    _, passages = _chunk(text, tokenizer, strategy, language="en")
    assert any(
        "Which diagnosis" in row["passage_text"] and "Answer: acute appendicitis" in row["passage_text"]
        for row in passages
    )


def test_short_orphan_attaches_forward_when_backward_exceeds_limit():
    class Words:
        @staticmethod
        def count(value):
            return len(value.split())

    text = "one two three four five six seven eight nine ten tiny tail next one two three four five six seven eight"
    spans = []
    cursor = 0
    values = (
        "one two three four five six seven eight nine ten",
        "tiny tail",
        "next one two three four five six seven eight",
    )
    for index, value in enumerate(values):
        start = text.index(value, cursor)
        spans.append((
            SectionSpan(f"section-{index:04d}", start, start + len(value), None),
            TextSpan(start, start + len(value), Words.count(value)),
        ))
        cursor = start + len(value)
    strategy = ChunkingStrategy("forward", 10, 11, 3, 0)
    repaired = _repair_short_section_passages(text, spans, Words(), strategy)
    assert len(repaired) == 2
    assert text[repaired[1][1].start:repaired[1][1].end] == (
        "tiny tail next one two three four five six seven eight"
    )


def test_very_long_sentence(tokenizer):
    strategy = ChunkingStrategy("long", 20, 24, 4, 2)
    row, passages = _chunk("word " * 500, tokenizer, strategy, language="en")
    assert len(passages) > 2
    assert not validate_document_passages(row, passages, tokenizer, strategy)


def test_bpe_non_monotonic_boundary(tokenizer):
    strategy = ChunkingStrategy("zh-boundary", 192, 256, 32, 32)
    text = (
        "男性 健身 休闲 保健 疾病 爱好 品味 女性 美容 整形 减肥 职场 情感 育儿 "
        "健康 男科 妇科 疾病 外科 体检 食疗 心理 生活 家居 旅游 养生 急救 解梦 "
    ) * 30
    row, passages = _chunk(text, tokenizer, strategy, language="zh")
    assert max(p["token_count"] for p in passages) <= 256
    assert not validate_document_passages(row, passages, tokenizer, strategy)


def test_very_short_document(tokenizer):
    row, passages = _chunk("Không.", tokenizer)
    assert len(passages) == 1
    assert covered_non_whitespace(row["normalized_text"], passages) == (6, 6)


@pytest.mark.parametrize("text,language", [
    ("Không sử dụng thuốc cho trẻ dưới 12 tuổi.", "vi"),
    ("糖尿病患者应定期监测血糖水平。", "zh"),
    ("Không dùng thuốc. 糖尿病患者应监测血糖。", "mixed"),
])
def test_multilingual_unicode_and_negation(tokenizer, text, language):
    row, passages = _chunk(text, tokenizer, language=language)
    reconstructed = "".join(p["passage_text"] for p in passages)
    assert text in reconstructed
    assert not validate_document_passages(row, passages, tokenizer, STRATEGIES[0])


def test_dosage_and_medical_measurements(tokenizer):
    values = ["5–10 mg/day", "HbA1c ≥ 6.5%", "eGFR < 30 mL/min/1.73m²"]
    text = ". ".join(values) + "."
    _, passages = _chunk(text, tokenizer)
    joined = " ".join(row["passage_text"] for row in passages)
    assert all(value in joined for value in values)


def test_no_cross_document_merge(tokenizer):
    _, first = _chunk("Document one only.", tokenizer, doc_id=1, language="en")
    _, second = _chunk("Document two only.", tokenizer, doc_id=2, language="en")
    assert {p["doc_id"] for p in first} == {1}
    assert {p["doc_id"] for p in second} == {2}
    assert set(p["passage_id"] for p in first).isdisjoint(p["passage_id"] for p in second)


def test_checkpoint_resume(tmp_path):
    passage = tmp_path / "passages.parquet"
    passage.write_bytes(b"passages")
    snapshot_hash = "b" * 64
    strategy = STRATEGIES[0].__dict__
    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(json.dumps({
        "status": "complete", "input_snapshot_sha256": snapshot_hash,
        "strategy": strategy, "passage_sha256": hashlib.sha256(b"passages").hexdigest(),
    }))
    assert _checkpoint_valid(checkpoint, snapshot_hash, strategy, passage)
    assert not _checkpoint_valid(checkpoint, "c" * 64, strategy, passage)


def test_write_once_output(tmp_path):
    path = tmp_path / "value.json"
    _atomic_json(path, {"value": 1})
    with pytest.raises(FileExistsError):
        _atomic_json(path, {"value": 2})


def test_phase0_phase1_immutable(tmp_path):
    candidate = tmp_path / "phase1"
    candidate.mkdir()
    with pytest.raises(ValueError, match="immutable input"):
        ensure_output_is_safe(candidate / "phase2", candidate)
    assert ensure_output_is_safe(tmp_path / "independent_phase2", candidate) == (
        tmp_path / "independent_phase2"
    ).resolve()


def test_manual_review_annotations_require_every_passage(tmp_path):
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({
        "case-001": {
            "A": ["complete", "good", "none", "useful", "Inspected."],
        }
    }))
    row = {
        "case_id": "case-001", "strategy_id": "A",
        "semantic_completeness": "UNREVIEWED", "boundary_quality": "UNREVIEWED",
        "context_loss": "UNREVIEWED", "redundant_overlap": "UNREVIEWED",
        "review_notes": "", "reviewed_by": "", "reviewed_at": "",
    }
    reviewed = apply_annotations([row], annotations)
    assert reviewed[0]["semantic_completeness"] == "complete"
    assert reviewed[0]["reviewed_by"] == "Codex manual inspection"
    with pytest.raises(ValueError, match="Missing manual annotation"):
        apply_annotations([row, {**row, "strategy_id": "B"}], annotations)


def test_manual_review_base_annotations_can_be_overridden(tmp_path):
    base = tmp_path / "base.json"
    override = tmp_path / "override.json"
    base.write_text(json.dumps({
        "case-001": {"A": ["complete", "good", "none", "none", "Base."]},
    }))
    override.write_text(json.dumps({
        "case-001": {"A": ["mostly_complete", "acceptable", "minor", "useful", "V2."]},
    }))
    row = {
        "case_id": "case-001", "strategy_id": "A",
        "semantic_completeness": "UNREVIEWED", "boundary_quality": "UNREVIEWED",
        "context_loss": "UNREVIEWED", "redundant_overlap": "UNREVIEWED",
        "review_notes": "", "reviewed_by": "", "reviewed_at": "",
    }
    reviewed = apply_annotations([row], override, base)
    assert reviewed[0]["semantic_completeness"] == "mostly_complete"
    assert reviewed[0]["review_notes"] == "V2."


def test_semantic_staging_is_separate_and_validated(tmp_path):
    config = tmp_path / "config.json"
    pilot = tmp_path / "pilot.parquet"
    output = tmp_path / "staging"
    config.write_text(json.dumps({
        "format": "vibiomir-phase2-semantic-staging-v1",
        "policy": "exclude_from_staging_retrieval_only",
        "phase1_mutation_allowed": False,
        "documents": [{"doc_id": 7, "review_case": "case-001", "reason": "navigation"}],
    }))
    pq.write_table(__import__("pyarrow").Table.from_pylist([{"doc_id": 7}]), pilot)
    result = materialize(config, pilot, output)
    assert result["phase1_mutated"] is False
    assert pq.read_table(output / "semantic_eligibility.parquet").num_rows == 1


def test_retrieval_evaluation_requires_full_evidence_containment():
    passages = [
        {"doc_id": 1, "passage_id": "split", "passage_text": "target answer first",
         "char_start": 0, "char_end": 19, "passage_index": 0, "token_count": 3},
        {"doc_id": 1, "passage_id": "full", "passage_text": "target answer first and second",
         "char_start": 0, "char_end": 30, "passage_index": 1, "token_count": 5},
        {"doc_id": 2, "passage_id": "noise", "passage_text": "unrelated text",
         "char_start": 0, "char_end": 14, "passage_index": 0, "token_count": 2},
    ]
    qrels = [{
        "query_id": 1, "query": "target answer", "doc_id": 1,
        "evidence_start": 0, "evidence_end": 30,
    }]
    metrics, rankings, _ = _evaluate_strategy(passages, qrels, "T")
    assert metrics["queries_with_containing_passage"] == 1
    assert {row["passage_id"] for row in rankings if row["is_relevant"]} == {"full"}


def test_frozen_config_must_match_runtime_strategy(tmp_path):
    path = tmp_path / "frozen.json"
    path.write_text(json.dumps({
        "status": "frozen_for_full_build",
        "chunker_version": CHUNKER_VERSION,
        "selected_strategy": STRATEGIES[2].__dict__,
    }))
    assert strategies_from_frozen_config(path) == (STRATEGIES[2],)
    config = json.loads(path.read_text())
    config["selected_strategy"]["max_tokens"] = 999
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="differs"):
        strategies_from_frozen_config(path)


def test_adjacent_expansion_can_recover_boundary_spanning_evidence():
    passages = [
        {"doc_id": 1, "passage_id": "left", "passage_text": "target first half",
         "char_start": 0, "char_end": 17, "passage_index": 0, "token_count": 3},
        {"doc_id": 1, "passage_id": "right", "passage_text": "second half",
         "char_start": 18, "char_end": 29, "passage_index": 1, "token_count": 2},
        {"doc_id": 2, "passage_id": "noise", "passage_text": "unrelated",
         "char_start": 0, "char_end": 9, "passage_index": 0, "token_count": 1},
    ]
    qrels = [{
        "query_id": 1, "query": "target first", "doc_id": 1,
        "evidence_start": 7, "evidence_end": 29,
    }]
    base, _, _ = _evaluate_strategy(passages, qrels, "B")
    expanded, rankings, _ = _evaluate_strategy(passages, qrels, "B_adjacent", adjacent=True)
    assert base["queries_with_containing_passage"] == 0
    assert expanded["queries_with_containing_passage"] == 1
    assert expanded["mean_added_context_chars_at_10"] > 0
    assert any(row["is_relevant"] for row in rankings)

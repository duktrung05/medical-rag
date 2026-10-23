from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.workflows.vibiomir.build_phase1_corpus import (
    build_corpus,
    ensure_output_is_safe,
    run_preflight,
)
from scripts.workflows.vibiomir.evaluate_phase1_quality import evaluate
from scripts.workflows.vibiomir.finalize_phase1_release import finalize
from src.medrag.data.phase1.language import detect_document_language
from src.medrag.data.phase1.normalization import normalize_medical_text, text_sha256
from src.medrag.data.phase1.quality_policy import QualityDecision, QualityPolicy
from src.medrag.data.phase1.selection import (
    Phase1IntegrityError,
    Phase1Paths,
    ProductionExpectations,
    load_selection,
    sha256_file,
    validate_locked_inputs,
)
from src.medrag.data.phase1.validation import assess_text_quality


PHASE0_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("official_url", pa.string()),
    ("final_status", pa.string()), ("content_group", pa.string()),
    ("crawled_success", pa.bool_()), ("usable", pa.bool_()),
    ("extraction_failed", pa.bool_()), ("crawl_failed", pa.bool_()),
    ("deferred", pa.bool_()), ("review_required", pa.bool_()),
    ("phase0_status", pa.string()), ("reason", pa.string()),
    ("raw_path", pa.string()), ("raw_sha256", pa.string()),
    ("text_sha256", pa.string()), ("raw_hash_verified", pa.bool_()),
    ("source", pa.string()),
])
STATUS_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("source_kind", pa.string()), ("ledger_state", pa.string()),
    ("final_status", pa.string()), ("usable", pa.bool_()),
    ("content_group", pa.string()), ("reason", pa.string()),
])
HTML_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("hostname", pa.string()),
    ("status", pa.string()), ("title", pa.string()), ("text", pa.string()),
    ("encoding", pa.string()), ("encoding_source", pa.string()),
    ("extraction_method", pa.string()), ("raw_path", pa.string()),
    ("gzip_sha256", pa.string()), ("raw_sha256", pa.string()),
    ("raw_bytes", pa.int64()), ("body_bytes", pa.int64()),
    ("text_sha256", pa.string()), ("error", pa.string()),
])
PUBMED_SCHEMA = pa.schema([
    ("id", pa.int64()), ("official_url", pa.string()), ("source_kind", pa.string()),
    ("title", pa.string()), ("text", pa.string()), ("language_metadata", pa.string()),
    ("pmid", pa.string()), ("raw_file", pa.string()), ("raw_sha256", pa.string()),
    ("text_sha256", pa.string()), ("extractor_version", pa.string()),
])


def _write_parquet(path: Path, rows: list[dict], schema: pa.Schema) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path, compression="zstd")


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def make_fixture(root: Path, *, bad_group: bool = False, bad_text_hash: bool = False):
    paths = Phase1Paths.from_repo(root)
    official = [(1, "https://example.test/one"), (2, "https://pubmed.test/2"),
                (3, "https://example.test/review"), (4, "https://example.test/failed")]
    query_rows = [{"id": 1, "query": "điều trị"}]
    _write_parquet(paths.official_corpus, [{"id": i, "url": u} for i, u in official],
                   pa.schema([("id", pa.int64()), ("url", pa.string())]))
    _write_parquet(paths.official_query, query_rows,
                   pa.schema([("id", pa.int64()), ("query", pa.string())]))

    html_text = "Không dùng Aspirin 81 mg khi bác sĩ chưa chỉ định.\n\nTheo dõi huyết áp mỗi ngày."
    pubmed_text = "Metformin 500 mg was not associated with severe adverse events in this cohort."
    review_text = "A valid article body that remains held for manual review despite source extraction."
    hashes = {1: text_sha256(html_text), 2: text_sha256(pubmed_text),
              3: text_sha256(review_text)}
    phase0_rows = []
    for ident, url in official:
        status = {1: "USABLE", 2: "USABLE", 3: "REVIEW_REQUIRED", 4: "CRAWL_FAILED"}[ident]
        group = {1: "legacy_html", 2: "pubmed", 3: "legacy_html", 4: None}[ident]
        usable = status == "USABLE"
        phase0_rows.append({
            "doc_id": ident, "official_url": url,
            "final_status": "usable" if ident != 4 else "fetch_http_error",
            "content_group": group, "crawled_success": ident != 4, "usable": usable,
            "extraction_failed": False, "crawl_failed": ident == 4, "deferred": False,
            "review_required": status == "REVIEW_REQUIRED", "phase0_status": status,
            "reason": status.lower(), "raw_path": f"/readonly/{ident}.gz" if ident != 4 else None,
            "raw_sha256": f"raw-{ident}" if ident != 4 else None,
            "text_sha256": hashes.get(ident), "raw_hash_verified": ident != 4,
            "source": "canonical_release",
        })
    _write_parquet(paths.phase0_state, phase0_rows, PHASE0_SCHEMA)

    html_path = root / "outputs/vibiomir_step2_20261013/content/legacy/documents.parquet"
    pubmed_path = root / "outputs/vibiomir_step2_20261013/content/pubmed/documents.parquet"
    _write_parquet(html_path, [
        {"id": 1, "official_url": official[0][1], "hostname": "example.test",
         "status": "usable", "title": "Điều trị an toàn", "text": html_text,
         "encoding": "utf-8", "encoding_source": "html_meta", "extraction_method": "fixture",
         "raw_path": "/readonly/1.gz", "gzip_sha256": "raw-1", "raw_sha256": "body-1",
         "raw_bytes": 100, "body_bytes": 200,
         "text_sha256": "bad" if bad_text_hash else hashes[1], "error": None},
        {"id": 3, "official_url": official[2][1], "hostname": "example.test",
         "status": "usable", "title": "Held article", "text": review_text,
         "encoding": "utf-8", "encoding_source": "html_meta", "extraction_method": "fixture",
         "raw_path": "/readonly/3.gz", "gzip_sha256": "raw-3", "raw_sha256": "body-3",
         "raw_bytes": 100, "body_bytes": 200, "text_sha256": hashes[3], "error": None},
    ], HTML_SCHEMA)
    _write_parquet(pubmed_path, [
        {"id": 2, "official_url": official[1][1], "source_kind": "pubmed",
         "title": "Metformin cohort", "text": pubmed_text, "language_metadata": "eng",
         "pmid": "2", "raw_file": "/readonly/pubmed.xml.gz", "raw_sha256": "raw-2",
         "text_sha256": hashes[2], "extractor_version": "fixture"},
    ], PUBMED_SCHEMA)

    status_rows = []
    for ident, url in official:
        group = {1: "pubmed" if bad_group else "legacy_html", 2: "pubmed",
                 3: "legacy_html", 4: None}[ident]
        usable = ident in {1, 2, 3}
        status_rows.append({
            "id": ident, "official_url": url, "hostname": "example.test",
            "source_kind": "pubmed" if ident == 2 else "legacy_web",
            "ledger_state": "fixture", "final_status": "usable" if usable else "fetch_http_error",
            "usable": usable, "content_group": group,
            "reason": "usable" if usable else "fetch_http_error",
        })
    _write_parquet(paths.release_status, status_rows, STATUS_SCHEMA)

    corpus_hash = sha256_file(paths.official_corpus)
    query_hash = sha256_file(paths.official_query)
    state_hash = sha256_file(paths.phase0_state)
    release_status_hash = sha256_file(paths.release_status)
    status_counts = {"CRAWL_FAILED": 1, "REVIEW_REQUIRED": 1, "USABLE": 2}
    _json(paths.source_lock, {
        "revision": "fixture-revision",
        "files": {
            "links_corpus_vi_en_zh.parquet": {"sha256": corpus_hash},
            "query.parquet": {"sha256": query_hash},
        },
    })
    _json(paths.audit_manifest, {
        "revision": "fixture-revision", "official_ids": 4, "queries": 1,
        "official_corpus_sha256": corpus_hash, "official_query_sha256": query_hash,
        "status_counts": status_counts, "staged_recovery_merged": False,
        "staged_recovery_id_list": [99, 100],
        "artifacts": {"document_state.parquet": {"rows": 4, "sha256": state_hash}},
    })
    content_files = []
    for group, kind, path, rows in [
        ("legacy_html", "html", html_path, 2), ("pubmed", "pubmed", pubmed_path, 1),
    ]:
        content_files.append({
            "group": group, "kind": kind,
            "documents": str(path.relative_to(root)), "documents_rows": rows,
            "documents_sha256": sha256_file(path),
        })
    _json(paths.release_manifest, {
        "complete": True, "corpus_revision": "fixture-revision", "official_ids": 4,
        "usable_documents": 3,
        "document_status": {"path": str(paths.release_status.relative_to(root)),
                            "sha256": release_status_hash, "rows": 4},
        "content_files": content_files,
    })
    expectations = ProductionExpectations(
        revision="fixture-revision", official_corpus_sha256=corpus_hash,
        official_query_sha256=query_hash, phase0_state_sha256=state_hash,
        release_status_sha256=release_status_hash, official_rows=4, query_rows=1,
        usable_rows=2, review_rows=1, release_usable_rows=3, content_files=2,
        status_counts=status_counts, staging_ids=frozenset({99, 100}),
    )
    return paths, expectations, {"html": html_text, "pubmed": pubmed_text}


def test_medical_normalization_preserves_meaning_and_paragraphs():
    source = "  Không\tdùng Aspirin 81 mg.  \r\n\r\n  SpO₂ >= 95%; NaCl 0.9%.  "
    result = normalize_medical_text(source)
    assert result.normalized_text == "Không dùng Aspirin 81 mg.\n\nSpO₂ >= 95%; NaCl 0.9%."
    assert "Không" in result.normalized_text
    assert "Aspirin 81 mg" in result.normalized_text
    assert "SpO₂" in result.normalized_text
    assert result.source_text == source
    assert result.source_text_sha256 == text_sha256(source)


@pytest.mark.parametrize(("text", "metadata", "expected"), [
    ("Không dùng Aspirin 81 mg khi chưa có chỉ định của bác sĩ.", None, "vi"),
    ("The patient was treated with metformin and was monitored for adverse events.", "chi", "en"),
    ("糖尿病患者应在医生指导下使用二甲双胍治疗。", None, "zh"),
])
def test_language_detection_uses_content_before_metadata(text, metadata, expected):
    result = detect_document_language(text, metadata)
    assert result.language == expected
    assert result.version == "script-stopwords-metadata-v2"
    if metadata == "chi":
        assert result.metadata_languages == ("zh",)
        assert result.metadata_agrees is False


def test_quality_validation_flags_without_dropping_text():
    text = "<script>challenge()</script> Verify you are human. " + "a" * 90
    result = assess_text_quality(
        text, title="Just a moment", source_kind="html", encoding=None,
        repeated_text_count=20,
    )
    assert result.review_required is True
    assert {"html_contamination", "challenge_text", "encoding_metadata_missing",
            "repeated_template_content"}.issubset(result.flags)
    assert text.endswith("a" * 90)


def test_language_detection_handles_short_vietnamese_and_mixed_content():
    vietnamese = detect_document_language(
        "Lịch thi đấu hôm nay và các bước theo dõi bệnh nhân tại nhà."
    )
    assert vietnamese.language == "vi"
    mixed = detect_document_language(
        "糖尿病患者应接受规范治疗。" * 3
        + " This clinical reference describes treatment and monitoring for the patient." * 3
    )
    assert mixed.language == "mixed"


def test_quality_validation_detects_navigation_and_mojibake():
    text = (
        "Trang chủ Đăng nhập Đăng ký Menu Chính sách bảo mật Điều khoản sử dụng. "
        "Liều dÃ¹ng 500 mg được giữ nguyên."
    )
    result = assess_text_quality(text, title="Bài viết", source_kind="html", encoding="utf-8")
    assert "navigation_heavy" in result.flags
    assert "mojibake_suspected" in result.flags


def test_quality_validation_does_not_flag_vietnamese_initial_as_mojibake():
    result = assess_text_quality(
        "Bệnh nhân cùng bạn là chị H.Â đến nhậu tại quán Toxic.",
        title="Báo cáo", source_kind="html", encoding="utf-8",
    )
    assert "mojibake_suspected" not in result.flags


def test_quality_policy_quarantines_blocking_signals_and_honors_audited_override():
    policy = QualityPolicy(
        version="test", default_status="REVIEW_REQUIRED",
        quarantine_flags=frozenset({"html_contamination"}),
        overrides={9: QualityDecision("PASS", "reviewed")},
    )
    flagged = assess_text_quality("<p>content</p>", title="x", source_kind="html", encoding="utf-8")
    assert policy.decide(1, flagged).status == "QUARANTINED"
    assert policy.decide(9, flagged).status == "PASS"


def test_quality_validation_does_not_treat_genetic_arrows_or_home_care_as_markup():
    text = (
        "The A-->G transition and 8q22-->qter duplication were evaluated. "
        "The patient requested home care, and the home nursing team arranged home support. "
        "Follow-up at home continued without an access menu."
    )
    result = assess_text_quality(text, title="Clinical report", source_kind="pubmed")
    assert "html_contamination" not in result.flags
    assert "navigation_heavy" not in result.flags

    admissions = ("Đăng ký xét tuyển " * 8) + "theo điểm thi và kết quả học tập."
    admissions_result = assess_text_quality(
        admissions, title="Thông báo tuyển sinh", source_kind="html", encoding="utf-8"
    )
    assert "navigation_heavy" not in admissions_result.flags


def test_fixture_selection_and_canonical_resolution(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path)
    context = run_preflight(paths, expectations, batch_size=2)
    assert set(context.selection.documents) == {1, 2}
    assert context.report.usable_rows == 2
    assert context.report.review_rows == 1
    assert context.report.canonical_rows_verified == 2
    assert context.report.staging_usable_intersection == 0


def test_official_identity_and_checksum_fail_closed(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path)
    fingerprints, _audit, _release = validate_locked_inputs(paths, expectations)
    assert fingerprints["official_corpus_sha256"] == expectations.official_corpus_sha256
    with paths.official_corpus.open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(Phase1IntegrityError, match="checksum mismatch"):
        validate_locked_inputs(paths, expectations)


def test_staging_id_cannot_enter_allowlist(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path)
    staged = replace(expectations, staging_ids=frozenset({1, 99, 100}))
    audit = json.loads(paths.audit_manifest.read_text(encoding="utf-8"))
    audit["staged_recovery_id_list"] = [1, 99, 100]
    _json(paths.audit_manifest, audit)
    fingerprints, _audit, _release = validate_locked_inputs(paths, staged)
    with pytest.raises(Phase1IntegrityError, match="Staging recovery IDs"):
        load_selection(paths, staged, fingerprints)


def test_content_group_mismatch_fails_before_output(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path, bad_group=True)
    with pytest.raises(Phase1IntegrityError, match="content_group mismatch"):
        run_preflight(paths, expectations, batch_size=2)
    assert not (tmp_path / "outputs/phase1-fixture").exists()


def test_text_checksum_mismatch_fails_before_output(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path, bad_text_hash=True)
    with pytest.raises(Phase1IntegrityError, match="Document text checksum mismatch"):
        run_preflight(paths, expectations, batch_size=2)


def test_build_does_not_overwrite_inputs_and_resume_is_idempotent(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path)
    before = {path: path.read_bytes() for path in (
        paths.source_lock, paths.audit_manifest, paths.phase0_state,
        paths.release_manifest, paths.release_status,
    )}
    context = run_preflight(paths, expectations, batch_size=2)
    output = tmp_path / "outputs/phase1-fixture-v1"
    first = build_corpus(context, output, batch_size=2)
    second = build_corpus(context, output, batch_size=2, resume=True)
    assert first == second
    assert first["documents"] == 2
    assert first["unique_document_ids"] == 2
    for path, content in before.items():
        assert path.read_bytes() == content
    with pytest.raises(Phase1IntegrityError, match="protected input root"):
        ensure_output_is_safe(paths.phase0_state.parent, paths)


def test_full_release_finalizer_writes_exact_pass_and_held_ledgers(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path)
    context = run_preflight(paths, expectations, batch_size=2)
    candidate = tmp_path / "outputs/phase1-full-v1"
    built = build_corpus(context, candidate, batch_size=2)
    policy_path = tmp_path / "quality-policy.json"
    _json(policy_path, {
        "version": "test-policy-v1",
        "default_status": "REVIEW_REQUIRED",
        "quarantine_flags": ["html_contamination"],
        "overrides": [],
    })
    release = finalize(candidate, policy_path)
    assert release["candidate_documents"] == built["documents"] == 2
    assert release["pass_documents"] + release["held_documents"] == 2
    assert sum(item["rows"] for item in release["decision_parts"]) == 2
    assert (candidate / "release/phase1_release_manifest.json").is_file()


def test_bounded_pilot_still_uses_full_preflight(tmp_path: Path):
    paths, expectations, _ = make_fixture(tmp_path)
    context = run_preflight(paths, expectations, batch_size=2)
    output = tmp_path / "outputs/phase1-pilot-v1"
    manifest = build_corpus(context, output, batch_size=2, pilot_documents=1)
    assert context.report.canonical_rows_verified == 2
    assert manifest["status"] == "pilot_complete"
    assert manifest["documents"] == 1
    assert manifest["build_scope"] == "pilot"
    assert manifest["pilot_sampling"] == "deterministic-source-language-stratified-v1"
    assert sum(manifest["pilot_selected_by_stratum"].values()) == 1
    rows = []
    for part in sorted((output / "clean_documents").glob("*.parquet")):
        rows.extend(pq.read_table(part).to_pylist())
    row = rows[0]
    assert row["detected_language"] in {"vi", "en"}
    assert row["language_detection_version"] == "script-stopwords-metadata-v2"
    assert row["quality_validation_version"] == "phase1-quality-rules-v3"

    quality_output = tmp_path / "outputs/quality-v1"
    summary = evaluate(output, quality_output, pilot_sample=1, quarantine_sample=0,
                       root=tmp_path)
    assert summary["documents_reviewed"] == 1
    assert summary["flagged_pilot_documents_reviewed"] == 1
    assert summary["normalization_invariant_failures"] == 0
    assert pq.read_table(quality_output / "review_sample.parquet").num_rows == 1
    assert pq.read_table(quality_output / "flagged_review.parquet").num_rows == 1

    refined_output = tmp_path / "outputs/phase1-pilot-v2"
    refined = build_corpus(
        context, refined_output, batch_size=2, pilot_documents=1,
        reuse_pilot_selection=output,
    )
    assert refined["pilot_ids_sha256"] == manifest["pilot_ids_sha256"]
    assert refined["pilot_selection_source"] == "outputs/phase1-pilot-v1"


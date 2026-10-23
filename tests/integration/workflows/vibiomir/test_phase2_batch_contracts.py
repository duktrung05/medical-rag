from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from src.medrag.data.phase2_batch import (
    Phase2Batch,
    ReleaseIntegrityError,
    load_phase2_batch,
    normalize_url,
    write_batch,
)


def _passage(doc_id: int, passage_id: str, text: str, index: int = 0) -> dict:
    return {
        "passage_id": passage_id,
        "doc_id": doc_id,
        "official_url": f"https://example.test/{doc_id}",
        "passage_index": index,
        "passage_text": text,
        "char_start": 0,
        "char_end": len(text),
        "source_text_sha256": "raw-hash",
        "text_representation": "normalized_text",
        "text_representation_sha256": "normalized-hash",
        "token_count": 1,
        "language": "vi",
        "section_id": "body",
        "section_heading": None,
        "strategy_id": "C",
        "chunker_version": "test-chunker",
        "tokenizer_name": "test-tokenizer",
        "tokenizer_revision": "test-revision",
        "phase1_release_id": "test-release",
        "corpus_revision": "test-corpus",
        "processing_run_id": "test-run",
    }


def test_normalize_url_ignores_fragment_but_preserves_query():
    assert normalize_url("HTTPS://Example.TEST/a?x=1#part") == "https://example.test/a?x=1"


def test_normalize_url_rejects_non_http_url():
    with pytest.raises(ReleaseIntegrityError):
        normalize_url("file:///etc/passwd")


def test_loader_rejects_duplicate_explicit_ids_before_release_io():
    config = Path("configs/vibiomir/phase2_batch_migration_v1.json")
    with pytest.raises(ValueError, match="Duplicate"):
        load_phase2_batch(config, doc_ids=[73, 73], query_ids=[4], max_documents=1000)


def test_write_batch_preserves_legacy_identity_and_full_sidecar(tmp_path):
    passage = _passage(73, "passage-73-0", "unchanged evidence text")
    ledger = [{"doc_id": 91, "review_case": "fixture", "reason": "excluded", "phase2_passages": 2}]
    batch = Phase2Batch(
        documents=[{"doc_id": 73, "official_url": "https://example.test/73", "title": "Title",
                    "normalized_text_sha256": "normalized-hash", "source_text_sha256": "raw-hash",
                    "detected_language": "vi"}],
        passages=[passage],
        queries=[{"id": 4, "query": "official query"}],
        manifest={"format": "fixture", "preflight": {"status": "pass"},
                  "semantic_exclusion_ledger": ledger},
    )
    output = tmp_path / "batch"
    write_batch(batch, output)
    chunks = pq.read_table(output / "chunks.parquet").to_pylist()
    provenance = pq.read_table(output / "passage_provenance.parquet").to_pylist()
    exclusions = pq.read_table(output / "exclusion_ledger.parquet").to_pylist()
    assert chunks == [{"doc_id": 73, "chunk_index": 0, "title": "Title",
                       "text": "unchanged evidence text", "passage_id": "passage-73-0"}]
    assert provenance[0]["char_start"] == 0 and provenance[0]["char_end"] == len("unchanged evidence text")
    assert provenance[0]["source_text_sha256"] == "raw-hash"
    assert provenance[0]["text_representation_sha256"] == "normalized-hash"
    assert exclusions == ledger
    saved_manifest = json.loads((output / "input_manifest.json").read_text())
    assert saved_manifest["chunks_rows"] == 1
    assert json.loads((output / "preflight_report.json").read_text())["status"] == "pass"


def test_write_batch_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    batch = Phase2Batch([], [], [], {"semantic_exclusion_ledger": [], "preflight": {"status": "pass"}})
    with pytest.raises(FileExistsError):
        write_batch(batch, output)

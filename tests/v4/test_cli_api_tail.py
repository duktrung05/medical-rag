"""Exercise the public V4 CLI and HTTP routes with evidence at the source tail."""
from __future__ import annotations

import json
import subprocess
import sys
from hashlib import sha256

from fastapi.testclient import TestClient

from src.medrag.api import create_app
from src.medrag.core.contracts import Source
from src.medrag.data.corpus_registry import CorpusVersion
from src.medrag.data.prepared import write_prepared


def test_cli_api_and_submission_keep_tail_evidence(tmp_path, monkeypatch):
    text = " ".join([f"ordinary{i}" for i in range(360)] + ["sentineltail", "treatment", "evidence"])
    source = Source("fixture-v1", 73, "https://example.org/73", "generic title", text,
                    "fixture", sha256(text.encode()).hexdigest())
    version = CorpusVersion("fixture-v1", "revision-1", tmp_path / "corpus.parquet", "fixture-hash", 1)
    prepared = tmp_path / "v4" / "prepared.jsonl"
    write_prepared([source], prepared, version)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"versions": {"fixture-v1": {
        "revision": version.revision, "path": str(version.path), "sha256": version.sha256,
        "rows": version.rows}}}), encoding="utf-8")
    config = tmp_path / "v4.json"
    config.write_text(json.dumps({"corpus_version": version.name, "registry": str(registry),
                                  "selection": {"max_docs": 10, "max_chunks": 10, "chunks_per_doc": 1}}),
                      encoding="utf-8")
    monkeypatch.setenv("MEDRAG_PREPARED", str(prepared))
    with TestClient(create_app(str(config))) as client:
        assert client.get("/health").status_code == 200
        response = client.post("/search", json={"id": 9, "query": "sentineltail"})
        assert response.status_code == 200
        prediction = response.json()
    assert prediction["relevant_docs"] == [73]
    assert len(prediction["relevant_chunks"]) == 1
    assert "sentineltail" in prediction["relevant_chunks"][0]["chunk_text"]
    process = subprocess.run(
        [sys.executable, "-m", "src.cli", "search", "--prepared", str(prepared),
         "--config", str(config), "--query", "sentineltail", "--limit", "10"],
        check=True, capture_output=True, text=True,
    )
    cli_results = json.loads(process.stdout)
    assert cli_results[0]["official_id"] == 73
    assert cli_results[0]["evidence"][0]["text"] == prediction["relevant_chunks"][0]["chunk_text"]


def test_cli_api_independent_documents_survive_chunk_budget(tmp_path, monkeypatch):
    sources = [Source("fixture-v1", i, f"https://example.org/{i}", "title",
                      f"medical treatment evidence {i}", "fixture", "raw") for i in (1, 2, 3)]
    version = CorpusVersion("fixture-v1", "revision-1", tmp_path / "corpus.parquet", "fixture-hash", 3)
    prepared = tmp_path / "v4" / "prepared.jsonl"
    write_prepared(sources, prepared, version)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({"versions": {"fixture-v1": {
        "revision": version.revision, "path": str(version.path), "sha256": version.sha256,
        "rows": version.rows}}}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_version": version.name, "registry": str(registry),
                                  "selection": {"max_docs": 10, "max_chunks": 1,
                                                "document_selection": "independent"}}))
    monkeypatch.setenv("MEDRAG_PREPARED", str(prepared))
    with TestClient(create_app(str(config))) as client:
        response = client.post("/search", json={"id": 9, "query": "medical treatment"})
        assert response.status_code == 200
        prediction = response.json()
    assert set(prediction) == {"id", "relevant_docs", "relevant_chunks"}
    assert len(prediction["relevant_docs"]) == 3
    assert len(prediction["relevant_chunks"]) == 1
    process = subprocess.run(
        [sys.executable, "-m", "src.cli", "search", "--prepared", str(prepared),
         "--config", str(config), "--query", "medical treatment", "--limit", "10"],
        check=True, capture_output=True, text=True, timeout=20,
    )
    selected = json.loads(process.stdout)
    assert [doc["official_id"] for doc in selected] == prediction["relevant_docs"]
    assert sum(len(doc["evidence"]) for doc in selected) == 1
    assert all(doc.get("document_evidence") for doc in selected)

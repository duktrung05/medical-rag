"""Dense cache integrity and real exact-search path with a local fixture encoder."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config import DenseRetrievalConfig, PipelineConfig
from src.data.loader import save_jsonl_records
from src.data.schema import ChunkRecord
from src.retrieval.dense import DenseRetriever


COMMIT = "0123456789abcdef0123456789abcdef01234567"


class TinyTokenizer:
    init_kwargs = {"_commit_hash": COMMIT}

    def __call__(self, texts, **kwargs):
        ids = [0 if "heart" in text.casefold() else 1 for text in texts]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long).reshape(-1, 1),
            "attention_mask": torch.ones((len(ids), 1), dtype=torch.long),
        }


class TinyModel:
    config = SimpleNamespace(hidden_size=2, _commit_hash=COMMIT)

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        return self

    def __call__(self, input_ids, attention_mask):
        vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
        hidden = vectors[input_ids]
        return SimpleNamespace(last_hidden_state=hidden)


def _fixture_chunks():
    return [
        ChunkRecord(chunk_id="c-z", doc_id="d-heart", chunk_index=0, language="en", text="heart disease"),
        ChunkRecord(chunk_id="c-a", doc_id="d-other", chunk_index=0, language="vi", text="lung health"),
    ]


def _fake_encoder(*args, device="auto", **kwargs):
    return TinyTokenizer(), TinyModel(), torch.device("cpu"), COMMIT, COMMIT


def test_dense_retriever_builds_cache_and_searches_exactly(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.dense.load_encoder", _fake_encoder)
    config = DenseRetrievalConfig(
        enabled=True,
        model_name="fixture-model",
        index_path=tmp_path / "dense-index",
        batch_size=1,
        normalize_embeddings=True,
    )
    manifest = DenseRetriever.build_index(_fixture_chunks(), config)
    retriever = DenseRetriever(
        model_name=config.model_name,
        index_path=config.index_path,
        revision=config.revision,
        tokenizer_name=config.tokenizer_name,
        tokenizer_revision=config.tokenizer_revision,
        batch_size=config.batch_size,
        max_length=config.max_length,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        normalize_embeddings=config.normalize_embeddings,
    )

    assert manifest["model_revision"] == COMMIT
    assert manifest["tokenizer_revision"] == COMMIT
    assert retriever.chunk_ids == ["c-a", "c-z"]
    assert retriever.search("heart", top_k=2) == [("c-z", 1.0), ("c-a", 0.0)]
    assert retriever.search_batch(["heart", "lung"], top_k=1) == [
        [("c-z", 1.0)],
        [("c-a", 1.0)],
    ]


def test_dense_backend_from_pipeline_config_runs_search(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.dense.load_encoder", _fake_encoder)
    chunks = _fixture_chunks()
    corpus_path = tmp_path / "chunks.jsonl"
    save_jsonl_records(corpus_path, chunks)
    dense_config = DenseRetrievalConfig(
        enabled=True,
        model_name="fixture-model",
        index_path=tmp_path / "dense-index",
    )
    DenseRetriever.build_index(chunks, dense_config)
    config = PipelineConfig(
        experiment_name="dense-fixture",
        backend="dense",
        corpus=corpus_path,
        retrieval={"dense": dense_config, "sparse": {"enabled": False}},
    )

    from src.service import build_pipeline

    pipeline = build_pipeline(config)
    result = pipeline.run_query("q1", "heart")

    assert isinstance(pipeline.retriever, DenseRetriever)
    assert result.relevant_chunks == ["c-z"]
    assert result.relevant_docs == ["d-heart"]


def test_dense_index_rejects_corpus_mismatch_before_loading_model(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.dense.load_encoder", _fake_encoder)
    config = DenseRetrievalConfig(
        enabled=True,
        model_name="fixture-model",
        index_path=tmp_path / "dense-index",
    )
    DenseRetriever.build_index(_fixture_chunks(), config)
    with pytest.raises(ValueError, match="does not match the configured corpus"):
        DenseRetriever(
            model_name=config.model_name,
            index_path=config.index_path,
            expected_corpus_hash="wrong-corpus-hash",
        )


def test_dense_index_rejects_non_finite_embeddings(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.dense.load_encoder", _fake_encoder)
    config = DenseRetrievalConfig(
        enabled=True,
        model_name="fixture-model",
        index_path=tmp_path / "dense-index",
    )
    DenseRetriever.build_index(_fixture_chunks(), config)
    embeddings_path = config.index_path / "passage_embeddings.npy"
    vectors = np.load(embeddings_path)
    vectors[0, 0] = np.nan
    np.save(embeddings_path, vectors)
    with pytest.raises(ValueError, match="NaN or infinity"):
        DenseRetriever(model_name=config.model_name, index_path=config.index_path)


def test_dense_query_dimension_and_model_revision_mismatches_are_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr("src.retrieval.dense.load_encoder", _fake_encoder)
    config = DenseRetrievalConfig(
        enabled=True,
        model_name="fixture-model",
        index_path=tmp_path / "dense-index",
    )
    DenseRetriever.build_index(_fixture_chunks(), config)
    retriever = DenseRetriever(model_name=config.model_name, index_path=config.index_path)
    monkeypatch.setattr(
        "src.retrieval.dense.encode_texts",
        lambda *args, **kwargs: np.ones((1, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="dimension does not match"):
        retriever.search("heart")

    monkeypatch.setattr(
        "src.retrieval.dense.load_encoder",
        lambda *args, **kwargs: (TinyTokenizer(), TinyModel(), torch.device("cpu"), "f" * 40, "f" * 40),
    )
    with pytest.raises(ValueError, match="model_revision"):
        DenseRetriever(model_name=config.model_name, index_path=config.index_path)


def test_cuda_request_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        from src.retrieval.dense_encoding import resolve_device

        assert resolve_device("cuda").type == "cpu"

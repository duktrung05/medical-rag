"""Strict schema checks for runtime and retrieval pipeline YAML configs."""

import pytest
from pydantic import ValidationError

from src.config import (
    PipelineConfig,
    RuntimeConfig,
    load_config,
    load_pipeline_config,
)
from src.retrieval.factory import _require_dependency, _require_index


def test_demo_config_remains_runtime_config():
    config = load_config("configs/demo.yaml")
    assert isinstance(config, RuntimeConfig)
    assert config.corpus.is_absolute()


@pytest.mark.parametrize(
    "path",
    [
        "configs/baseline_bm25.yaml",
        "configs/baseline_dense.yaml",
        "configs/baseline_hybrid.yaml",
        "configs/baseline_hybrid_rerank.yaml",
    ],
)
def test_baseline_configs_load_as_pipeline_config(path):
    config = load_pipeline_config(path)
    assert isinstance(config, PipelineConfig)
    assert config.corpus.is_absolute()


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        RuntimeConfig(corpus="chunks.jsonl", unexpected=True)

    with pytest.raises(ValidationError, match="extra_forbidden"):
        PipelineConfig.model_validate(
            {
                "experiment_name": "typo",
                "backend": "sparse",
                "corpus": "chunks.jsonl",
                "retrieval": {
                    "sparse": {"enabled": True, "index_path": "index", "topk": 10}
                },
            }
        )


def test_invalid_values_and_stage_combinations_are_rejected():
    base = {
        "experiment_name": "invalid",
        "backend": "hybrid",
        "corpus": "chunks.jsonl",
        "retrieval": {
            "sparse": {"enabled": True, "index_path": "sparse"},
            "dense": {"enabled": True, "model_name": "model", "index_path": "dense"},
        },
        "fusion": {"enabled": False},
    }
    with pytest.raises(ValidationError, match="fusion.enabled"):
        PipelineConfig.model_validate(base)

    base["fusion"] = {"enabled": True, "rrf_k": 0}
    with pytest.raises(ValidationError, match="greater_than_equal"):
        PipelineConfig.model_validate(base)


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": True},
        {"enabled": True, "index_path": "sparse"},
    ],
)
def test_enabled_components_require_their_config(config):
    sparse = {"enabled": True, "index_path": "sparse"}
    dense = {"enabled": True, "model_name": "dense-model", "index_path": "dense"}
    reranker = {"enabled": True, "model_name": "cross-encoder"}
    if config == {"enabled": True}:
        sparse.pop("index_path")
    else:
        dense.pop("model_name")
    with pytest.raises(ValidationError):
        PipelineConfig.model_validate(
            {
                "experiment_name": "missing-required-setting",
                "backend": "hybrid",
                "corpus": "chunks.jsonl",
                "retrieval": {"sparse": sparse, "dense": dense},
                "fusion": {"enabled": True},
                "reranker": reranker,
            }
        )


def test_enabled_reranker_requires_model():
    with pytest.raises(ValidationError, match="reranker.model_name"):
        PipelineConfig.model_validate(
            {
                "experiment_name": "missing-reranker-model",
                "backend": "sparse",
                "corpus": "chunks.jsonl",
                "retrieval": {"sparse": {"enabled": True, "index_path": "sparse"}},
                "reranker": {"enabled": True},
            }
        )


def test_missing_index_and_dependency_have_actionable_errors(tmp_path, monkeypatch):
    with pytest.raises(FileNotFoundError, match="Build the index first"):
        _require_index(tmp_path / "absent-index", "dense retrieval")

    monkeypatch.setattr("src.retrieval.factory.find_spec", lambda _: None)
    with pytest.raises(RuntimeError, match="retrieval.*missing"):
        _require_dependency("faiss", "dense retrieval")


def test_pipeline_config_resolves_relative_paths(tmp_path):
    config_file = tmp_path / "pipeline.yaml"
    config_file.write_text(
        """experiment_name: test
backend: sparse
corpus: corpus/chunks.jsonl
retrieval:
  sparse:
    enabled: true
    index_path: indexes/sparse
""",
        encoding="utf-8",
    )
    config = load_pipeline_config(config_file)
    assert config.corpus == (tmp_path / "corpus/chunks.jsonl").resolve()
    assert config.retrieval.sparse.index_path == (tmp_path / "indexes/sparse").resolve()

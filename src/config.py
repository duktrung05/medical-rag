"""Validated runtime settings. Paths are relative to the configuration file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictConfig(BaseModel):
    """Base for configuration models that reject misspelled/unknown keys."""

    model_config = ConfigDict(extra="forbid")


class SparseRetrievalConfig(StrictConfig):
    enabled: bool = False
    engine: Literal["bm25", "tantivy", "rank_bm25"] = "bm25"
    tokenizer: Literal["word", "char_ngram"] = "word"
    char_ngram_size: int = Field(default=3, ge=1, le=8)
    k1: float = Field(default=1.5, gt=0)
    b: float = Field(default=0.75, ge=0, le=1)
    index_path: Path | None = None
    top_k: int = Field(default=200, ge=1)

    @model_validator(mode="after")
    def require_index_when_enabled(self):
        if self.enabled and self.index_path is None:
            raise ValueError("retrieval.sparse.index_path is required when enabled")
        return self


class DenseRetrievalConfig(StrictConfig):
    enabled: bool = False
    model_name: str | None = None
    revision: str = "main"
    tokenizer_name: str | None = None
    tokenizer_revision: str | None = None
    index_path: Path | None = None
    index_type: Literal["exact", "faiss"] = "exact"
    top_k: int = Field(default=300, ge=1)
    batch_size: int = Field(default=32, ge=1)
    max_length: int = Field(default=512, ge=8)
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    normalize_embeddings: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"

    @model_validator(mode="after")
    def require_model_and_index_when_enabled(self):
        if self.enabled and not self.model_name:
            raise ValueError("retrieval.dense.model_name is required when enabled")
        if self.enabled and self.index_path is None:
            raise ValueError("retrieval.dense.index_path is required when enabled")
        return self


class RetrievalConfig(StrictConfig):
    sparse: SparseRetrievalConfig = Field(default_factory=SparseRetrievalConfig)
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)

    @model_validator(mode="after")
    def require_enabled_retriever(self):
        if not self.sparse.enabled and not self.dense.enabled:
            raise ValueError("At least one retrieval method must be enabled")
        return self


class FusionConfig(StrictConfig):
    enabled: bool = False
    method: Literal["rrf"] = "rrf"
    rrf_k: int = Field(default=60, ge=1)
    top_k: int = Field(default=150, ge=1)


class RerankerConfig(StrictConfig):
    enabled: bool = False
    model_name: str | None = None
    top_k: int = Field(default=100, ge=1)
    batch_size: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def require_model_when_enabled(self):
        if self.enabled and not self.model_name:
            raise ValueError("reranker.model_name is required when enabled")
        return self


class DocumentScoringConfig(StrictConfig):
    aggregation: Literal["max", "mean", "hybrid_mean"] = "max"
    top_n_mean: int = Field(default=3, ge=1)


class ScoringConfig(StrictConfig):
    document: DocumentScoringConfig = Field(default_factory=DocumentScoringConfig)


class SelectionStageConfig(StrictConfig):
    threshold: float = Field(default=0.5, ge=0)
    relative_delta: float = Field(default=0.3, ge=0)
    min_k: int = Field(default=1, ge=0)
    max_k: int = Field(default=20, ge=1)
    @model_validator(mode="after")
    def validate_bounds(self):
        if self.min_k > self.max_k:
            raise ValueError("min_k must be less than or equal to max_k")
        return self


class SelectionConfig(StrictConfig):
    chunk: SelectionStageConfig = Field(default_factory=SelectionStageConfig)
    document: SelectionStageConfig = Field(default_factory=SelectionStageConfig)


class PipelineConfig(StrictConfig):
    """Strict schema for retrieval experiments (the baseline YAML files)."""

    experiment_name: str
    backend: Literal["sparse", "dense", "hybrid"]
    corpus: Path
    retrieval: RetrievalConfig
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)
    scoring: ScoringConfig = Field(default_factory=ScoringConfig)
    selection: SelectionConfig = Field(default_factory=SelectionConfig)

    @model_validator(mode="after")
    def validate_fusion_dependency(self):
        sparse_enabled = self.retrieval.sparse.enabled
        dense_enabled = self.retrieval.dense.enabled
        retriever_count = int(sparse_enabled) + int(dense_enabled)
        expected = {
            "sparse": (True, False),
            "dense": (False, True),
            "hybrid": (True, True),
        }[self.backend]
        if (sparse_enabled, dense_enabled) != expected:
            raise ValueError(
                f"backend '{self.backend}' requires retrieval.sparse.enabled="
                f"{expected[0]} and retrieval.dense.enabled={expected[1]}"
            )
        if retriever_count > 1 and not self.fusion.enabled:
            raise ValueError("fusion.enabled must be true when multiple retrievers are enabled")
        if self.fusion.enabled and retriever_count < 2:
            raise ValueError("fusion.enabled requires at least two enabled retrievers")
        return self


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load and strictly validate a retrieval experiment YAML file."""
    path = Path(path).resolve()
    config = PipelineConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not config.corpus.is_absolute():
        config.corpus = (path.parent / config.corpus).resolve()
    for retriever in (config.retrieval.sparse, config.retrieval.dense):
        if retriever.index_path is not None and not retriever.index_path.is_absolute():
            retriever.index_path = (path.parent / retriever.index_path).resolve()
    return config


class RuntimeConfig(StrictConfig):

    experiment_name: str = "demo"
    backend: Literal["demo"] = "demo"
    corpus: Path
    max_chunks: int = Field(default=10, ge=1, le=100)


def load_config(path: str | Path) -> RuntimeConfig | PipelineConfig:
    path = Path(path).resolve()
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if isinstance(raw_config, dict) and "retrieval" in raw_config:
        return load_pipeline_config(path)
    config = RuntimeConfig.model_validate(raw_config)
    if not config.corpus.is_absolute():
        config.corpus = (path.parent / config.corpus).resolve()
    return config

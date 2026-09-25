"""Factories for sparse, dense, and hybrid retrievers."""

from importlib.util import find_spec

from typing import Literal

from src.config import DenseRetrievalConfig, FusionConfig, RetrievalConfig, SparseRetrievalConfig
from src.retrieval.bm25 import BM25Retriever, BaseRetriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


def _require_dependency(module_name: str, feature: str) -> None:
    if find_spec(module_name) is None:
        package_name = {
            "rank_bm25": "rank-bm25",
            "sentence_transformers": "sentence-transformers",
            "faiss": "faiss-cpu",
        }.get(module_name, module_name)
        raise RuntimeError(
            f"Cannot enable {feature}: dependency '{package_name}' is missing. "
            "Install retrieval dependencies with `pip install -e '.[retrieval]'`."
        )


def _require_index(index_path, feature: str) -> None:
    if index_path is None or not index_path.exists():
        raise FileNotFoundError(
            f"Cannot enable {feature}: index path is missing or does not exist: {index_path}. "
            "Build the index first or correct the configured index_path."
        )


def create_sparse_retriever(
    config: SparseRetrievalConfig,
    *,
    expected_corpus_hash: str | None = None,
) -> BaseRetriever:
    if not config.enabled:
        raise ValueError("Cannot create sparse retriever: retrieval.sparse.enabled is false")
    _require_index(config.index_path, "sparse retrieval")
    if config.engine == "tantivy":
        raise NotImplementedError("The configured Tantivy retriever is not implemented yet")
    return BM25Retriever(
        config.index_path,
        expected_corpus_hash=expected_corpus_hash,
        tokenizer=config.tokenizer,
        char_ngram_size=config.char_ngram_size,
        k1=config.k1,
        b=config.b,
    )


def create_dense_retriever(
    config: DenseRetrievalConfig,
    *,
    expected_corpus_hash: str | None = None,
) -> BaseRetriever:
    if not config.enabled:
        raise ValueError("Cannot create dense retriever: retrieval.dense.enabled is false")
    _require_index(config.index_path, "dense retrieval")
    _require_dependency("torch", "dense retrieval")
    _require_dependency("transformers", "dense retrieval")
    if config.index_type == "faiss":
        _require_dependency("faiss", "FAISS dense retrieval")
    return DenseRetriever(
        model_name=config.model_name or "",
        index_path=config.index_path,
        revision=config.revision,
        tokenizer_name=config.tokenizer_name,
        tokenizer_revision=config.tokenizer_revision,
        batch_size=config.batch_size,
        max_length=config.max_length,
        query_prefix=config.query_prefix,
        passage_prefix=config.passage_prefix,
        normalize_embeddings=config.normalize_embeddings,
        expected_corpus_hash=expected_corpus_hash,
        index_type=config.index_type,
        device=config.device,
    )


def create_hybrid_retriever(
    sparse_retriever: BaseRetriever,
    dense_retriever: BaseRetriever,
    config: FusionConfig,
    *,
    sparse_top_k: int | None = None,
    dense_top_k: int | None = None,
) -> BaseRetriever:
    if not config.enabled:
        raise ValueError("Cannot create hybrid retriever: fusion.enabled is false")
    if config.method != "rrf":
        raise ValueError(f"Unsupported fusion method: {config.method}")
    return HybridRetriever(sparse_retriever, dense_retriever, rrf_k=config.rrf_k,
                           sparse_top_k=sparse_top_k, dense_top_k=dense_top_k)


def create_retriever(
    backend: Literal["sparse", "dense", "hybrid"],
    config: RetrievalConfig,
    fusion: FusionConfig,
    *,
    expected_corpus_hash: str | None = None,
) -> tuple[BaseRetriever, int]:
    """Build the configured retriever and return its candidate top_k."""
    if backend == "hybrid":
        if not config.sparse.enabled or not config.dense.enabled:
            raise ValueError("Hybrid backend requires sparse and dense retrieval to be enabled")
        return (
            create_hybrid_retriever(
                create_sparse_retriever(config.sparse, expected_corpus_hash=expected_corpus_hash),
                create_dense_retriever(config.dense, expected_corpus_hash=expected_corpus_hash),
                fusion,
                sparse_top_k=config.sparse.top_k,
                dense_top_k=config.dense.top_k,
            ),
            fusion.top_k,
        )
    if backend == "sparse":
        if not config.sparse.enabled:
            raise ValueError("Sparse backend requires retrieval.sparse.enabled=true")
        return (
            create_sparse_retriever(config.sparse, expected_corpus_hash=expected_corpus_hash),
            config.sparse.top_k,
        )
    if backend == "dense":
        if not config.dense.enabled:
            raise ValueError("Dense backend requires retrieval.dense.enabled=true")
        return (
            create_dense_retriever(config.dense, expected_corpus_hash=expected_corpus_hash),
            config.dense.top_k,
        )
    raise ValueError(f"Unsupported retrieval backend: {backend}")

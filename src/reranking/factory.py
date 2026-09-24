"""Factories for configured rerankers."""

from importlib.util import find_spec

from src.config import RerankerConfig
from src.reranking.base import BaseReranker
from src.reranking.bge_reranker import BGEReranker


def create_reranker(config: RerankerConfig) -> BaseReranker | None:
    if not config.enabled:
        return None
    if find_spec("sentence_transformers") is None:
        raise RuntimeError(
            "Cannot enable reranking: dependency 'sentence-transformers' is missing. "
            "Install retrieval dependencies with `pip install -e '.[retrieval]'`."
        )
    return BGEReranker(model_name=config.model_name or "", batch_size=config.batch_size)

"""BGE Reranker v2 M3 cross-encoder implementation."""

from typing import List, Tuple

from src.reranking.base import BaseReranker


class BGEReranker(BaseReranker):
    """Batched cross-encoder; the model is loaded once at construction."""

    def __init__(self, model_name="BAAI/bge-reranker-v2-m3", batch_size=16,
                 max_length=512, device="auto", model=None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.model = model

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(self.model_name, max_length=self.max_length,
                                      device=None if self.device == "auto" else self.device)
        return self.model

    def rerank(self, query: str, candidates: List[Tuple], top_k: int = 100) -> List[Tuple[str, float]]:
        if not candidates or top_k <= 0:
            return []
        passages = []
        for item in candidates:
            if len(item) == 2:
                chunk_id, text = item
                title, context = None, None
            elif len(item) == 4:
                chunk_id, text, title, context = item
            else:
                raise ValueError("Expected (chunk_id, text) or (chunk_id, text, title, context)")
            parts = ([f"Title: {title}"] if title else [])
            parts += ([f"Context: {context}"] if context else [])
            parts.append(text)
            passages.append((chunk_id, "\n".join(parts)))
        scores = self._load_model().predict(
            [(query, text) for _, text in passages], batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        if len(scores) != len(passages):
            raise ValueError("Reranker score count does not match candidate count")
        ranked = sorted(((cid, float(score)) for (cid, _), score in zip(passages, scores, strict=True)),
                        key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

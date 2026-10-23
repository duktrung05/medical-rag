from __future__ import annotations

from typing import Protocol


class PassageReranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


class BGEReranker:
    """Optional pinned BGE cross-encoder; load only when configured."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3", *, revision: str,
                 max_length: int = 512, device: str | None = None):
        try:
            import torch
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise RuntimeError("Install sentence-transformers and torch to enable the BGE reranker") from exc
        from src.medrag.core.model_revision import require_cached_revision

        snapshot = require_cached_revision(model_name, revision)
        self.model = CrossEncoder(model_name, revision=revision, max_length=max_length, device=device,
                                  model_kwargs={"torch_dtype": torch.float16},
                                  activation_fn=torch.nn.Identity())
        self.revision = revision
        self.max_length = max_length
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)

    def pair_token_lengths(self, query: str, passages: list[str]) -> list[int]:
        return [len(self.tokenizer(query, passage, add_special_tokens=True,
                                   truncation=False)["input_ids"]) for passage in passages]

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        import numpy as np

        values = np.asarray(self.model.predict([[query, passage] for passage in passages],
                                               show_progress_bar=False)).reshape(-1)
        return [float(value) for value in values]


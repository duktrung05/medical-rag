"""Shared transformer encoding path used by dense indexing and smoke runs."""

from __future__ import annotations

import warnings
import re

import numpy as np


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    import torch

    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    return (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)


def resolve_device(requested: str = "auto") -> torch.device:
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        warnings.warn("CUDA was requested but is unavailable; falling back to CPU", RuntimeWarning)
        return torch.device("cpu")
    return torch.device(requested)


def load_encoder(
    model_name: str,
    *,
    revision: str,
    tokenizer_name: str | None = None,
    tokenizer_revision: str | None = None,
    device: str = "auto",
):
    """Load model/tokenizer and report the resolved immutable revisions."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    actual_device = resolve_device(device)
    tokenizer_ref = tokenizer_name or model_name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_ref,
        revision=tokenizer_revision or revision,
    )
    model = AutoModel.from_pretrained(model_name, revision=revision)
    try:
        model = model.to(actual_device)
    except (RuntimeError, AssertionError) as error:
        if actual_device.type != "cuda":
            raise
        warnings.warn(f"Could not initialize model on CUDA ({error}); falling back to CPU", RuntimeWarning)
        actual_device = torch.device("cpu")
        model = model.to(actual_device)
    model.eval()
    model_revision = _immutable_revision(
        getattr(getattr(model, "config", None), "_commit_hash", None), revision, "model"
    )
    requested_tokenizer_revision = tokenizer_revision or revision
    tokenizer_revision_resolved = _immutable_revision(
        getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        requested_tokenizer_revision,
        "tokenizer",
    )
    return tokenizer, model, actual_device, model_revision, tokenizer_revision_resolved


def _immutable_revision(resolved: str | None, requested: str, component: str) -> str:
    if resolved and re.fullmatch(r"[0-9a-fA-F]{40}", str(resolved)):
        return str(resolved).lower()
    if re.fullmatch(r"[0-9a-fA-F]{40}", requested):
        return requested.lower()
    raise RuntimeError(
        f"Could not pin the resolved {component} revision to an immutable commit SHA. "
        f"Set its revision to a 40-character commit SHA (requested: {requested!r})."
    )


def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    *,
    batch_size: int,
    max_length: int,
    normalize: bool,
    device: torch.device,
) -> np.ndarray:
    import torch

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    batches: list[np.ndarray] = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        tokens = tokenizer(
            texts[start : start + batch_size],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        tokens = {key: value.to(device) for key, value in tokens.items()}
        with torch.inference_mode():
            pooled = mean_pool(model(**tokens).last_hidden_state, tokens["attention_mask"])
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        batches.append(pooled.cpu().numpy().astype(np.float32))
    if not batches:
        return np.empty((0, 0), dtype=np.float32)
    embeddings = np.concatenate(batches, axis=0)
    if not np.isfinite(embeddings).all():
        raise ValueError("Encoded embeddings contain NaN or infinity")
    return embeddings

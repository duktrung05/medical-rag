"""Exact transformer dense retrieval with a cached passage-vector index."""

from __future__ import annotations

import json
import hashlib
import warnings
from pathlib import Path
from typing import Sequence

import numpy as np

from src.config import DenseRetrievalConfig
from src.data.schema import ChunkRecord
from src.indexing.sparse_index import corpus_sha256
from src.retrieval.bm25 import BaseRetriever
from src.retrieval.dense_encoding import encode_texts, load_encoder
from src.retrieval.exact_dense import RankedChunk, exact_search


class DenseRetriever(BaseRetriever):
    """Load a passage cache, encode queries, then rank by exact inner product."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        index_path: str | Path = "artifacts/dense_index",
        *,
        revision: str = "main",
        tokenizer_name: str | None = None,
        tokenizer_revision: str | None = None,
        batch_size: int = 32,
        max_length: int = 512,
        query_prefix: str = "query: ",
        passage_prefix: str = "passage: ",
        normalize_embeddings: bool = True,
        expected_corpus_hash: str | None = None,
        index_type: str = "exact",
        device: str = "auto",
    ):
        self.model_name = model_name
        self.index_path = Path(index_path)
        self.revision = revision
        self.tokenizer_name = tokenizer_name or model_name
        self.tokenizer_revision = tokenizer_revision or revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.normalize_embeddings = normalize_embeddings
        self.index_type = index_type
        if index_type not in {"exact", "faiss"}:
            raise ValueError(f"Unsupported dense index_type: {index_type}")

        self.manifest, self.chunk_ids, self.passage_embeddings = self._load_cache(
            expected_corpus_hash
        )
        self.tokenizer, self.model, self.device, model_revision, tokenizer_revision_resolved = load_encoder(
            self.model_name,
            revision=self.revision,
            tokenizer_name=self.tokenizer_name,
            tokenizer_revision=self.tokenizer_revision,
            device=device,
        )
        self.model_revision = model_revision
        self.tokenizer_revision_resolved = tokenizer_revision_resolved
        self._validate_model_manifest()
        model_dimension = getattr(self.model.config, "hidden_size", None)
        if model_dimension is not None and model_dimension != self.passage_embeddings.shape[1]:
            raise ValueError(
                "Dense model/index dimension mismatch "
                f"(model={model_dimension}, index={self.passage_embeddings.shape[1]})"
            )
        self._faiss_index = self._build_faiss_adapter() if index_type == "faiss" else None

    @classmethod
    def build_index(
        cls,
        chunks: Sequence[ChunkRecord],
        config: DenseRetrievalConfig,
        *,
        device: str | None = None,
    ) -> dict:
        if not chunks:
            raise ValueError("Cannot build dense index from an empty corpus")
        ordered_chunks = sorted(chunks, key=lambda item: item.chunk_id)
        chunk_ids = [chunk.chunk_id for chunk in ordered_chunks]
        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Cannot build dense index: duplicate chunk_id values")
        tokenizer, model, actual_device, model_revision, tokenizer_revision_resolved = load_encoder(
            config.model_name or "",
            revision=config.revision,
            tokenizer_name=config.tokenizer_name,
            tokenizer_revision=config.tokenizer_revision,
            device=device or config.device,
        )
        try:
            passage_embeddings = encode_texts(
                [config.passage_prefix + chunk.text for chunk in ordered_chunks],
                tokenizer,
                model,
                batch_size=config.batch_size,
                max_length=config.max_length,
                normalize=config.normalize_embeddings,
                device=actual_device,
            )
        except RuntimeError as error:
            if actual_device.type != "cuda":
                raise
            import torch

            warnings.warn(f"Dense index encoding failed on CUDA ({error}); retrying on CPU", RuntimeWarning)
            torch.cuda.empty_cache()
            actual_device = torch.device("cpu")
            model = model.to(actual_device)
            passage_embeddings = encode_texts(
                [config.passage_prefix + chunk.text for chunk in ordered_chunks],
                tokenizer,
                model,
                batch_size=config.batch_size,
                max_length=config.max_length,
                normalize=config.normalize_embeddings,
                device=actual_device,
            )
        if passage_embeddings.shape[0] != len(chunk_ids):
            raise ValueError("Passage embedding count does not match ordered chunk IDs")
        if passage_embeddings.shape[1] < 1 or not np.isfinite(passage_embeddings).all():
            raise ValueError("Passage embeddings have invalid dimensions or contain NaN/infinity")

        index_dir = Path(config.index_path)
        index_dir.mkdir(parents=True, exist_ok=True)
        _write_numpy(index_dir / "passage_embeddings.npy", passage_embeddings)
        (index_dir / "chunk_ids.json").write_text(
            json.dumps(chunk_ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        manifest = {
            "format_version": 1,
            "model_name": config.model_name,
            "model_revision": model_revision,
            "tokenizer_name": config.tokenizer_name or config.model_name,
            "tokenizer_revision": tokenizer_revision_resolved,
            "requested_revision": config.revision,
            "requested_tokenizer_revision": config.tokenizer_revision or config.revision,
            "corpus_sha256": corpus_sha256(list(ordered_chunks)),
            "ordered_chunk_ids": chunk_ids,
            "ordered_chunk_ids_sha256": hashlib.sha256(
                json.dumps(chunk_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
            "num_passages": len(chunk_ids),
            "embedding_dimension": int(passage_embeddings.shape[1]),
            "batch_size": config.batch_size,
            "max_length": config.max_length,
            "query_prefix": config.query_prefix,
            "passage_prefix": config.passage_prefix,
            "normalize_embeddings": config.normalize_embeddings,
            "pooling": "attention_mask_mean",
            "index_type": config.index_type,
            "device": str(actual_device),
        }
        manifest_tmp = index_dir / "manifest.json.tmp"
        manifest_tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        manifest_tmp.replace(index_dir / "manifest.json")
        return manifest

    def _load_cache(self, expected_corpus_hash: str | None):
        manifest_path = self.index_path / "manifest.json"
        ids_path = self.index_path / "chunk_ids.json"
        vectors_path = self.index_path / "passage_embeddings.npy"
        missing = [path.name for path in (manifest_path, ids_path, vectors_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Dense index at {self.index_path} is missing {', '.join(missing)}; "
                "build it with scripts/build_dense_index.py"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        chunk_ids = json.loads(ids_path.read_text(encoding="utf-8"))
        vectors = np.load(vectors_path, allow_pickle=False)
        if manifest.get("format_version") != 1:
            raise ValueError(f"Unsupported dense index format version: {manifest.get('format_version')}")
        if chunk_ids != manifest.get("ordered_chunk_ids"):
            raise ValueError("Dense index chunk_ids.json does not match ordered_chunk_ids in manifest")
        ids_hash = hashlib.sha256(
            json.dumps(chunk_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if manifest.get("ordered_chunk_ids_sha256") != ids_hash:
            raise ValueError("Dense ordered chunk ID hash does not match manifest")
        if not chunk_ids or len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("Dense index must contain non-empty, unique ordered chunk IDs")
        if vectors.ndim != 2:
            raise ValueError("Dense passage embeddings must be a 2-D matrix")
        if vectors.shape[0] != len(chunk_ids) or vectors.shape[0] != manifest.get("num_passages"):
            raise ValueError("Dense embedding count does not match cached chunk IDs/manifest")
        if vectors.shape[1] != manifest.get("embedding_dimension"):
            raise ValueError("Dense embedding dimension does not match manifest")
        if not np.isfinite(vectors).all():
            raise ValueError("Dense passage embeddings contain NaN or infinity")
        if expected_corpus_hash and manifest.get("corpus_sha256") != expected_corpus_hash:
            raise ValueError(
                "Dense index does not match the configured corpus "
                f"(index={manifest.get('corpus_sha256')}, corpus={expected_corpus_hash}); rebuild the index"
            )
        return manifest, chunk_ids, np.asarray(vectors, dtype=np.float32)

    def _validate_model_manifest(self) -> None:
        expected = {
            "model_name": self.model_name,
            "model_revision": self.model_revision,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision_resolved,
            "max_length": self.max_length,
            "query_prefix": self.query_prefix,
            "passage_prefix": self.passage_prefix,
            "normalize_embeddings": self.normalize_embeddings,
            "pooling": "attention_mask_mean",
        }
        mismatches = [
            f"{key}: index={self.manifest.get(key)!r}, runtime={value!r}"
            for key, value in expected.items()
            if self.manifest.get(key) != value
        ]
        if mismatches:
            raise ValueError("Dense index/model config mismatch: " + "; ".join(mismatches))

    def _encode(self, texts: list[str]) -> np.ndarray:
        try:
            vectors = encode_texts(
                texts,
                self.tokenizer,
                self.model,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=self.normalize_embeddings,
                device=self.device,
            )
        except RuntimeError as error:
            if self.device.type != "cuda":
                raise
            warnings.warn(f"Dense encoding failed on CUDA ({error}); retrying on CPU", RuntimeWarning)
            import torch

            torch.cuda.empty_cache()
            self.device = torch.device("cpu")
            self.model = self.model.to(self.device)
            vectors = encode_texts(
                texts,
                self.tokenizer,
                self.model,
                batch_size=self.batch_size,
                max_length=self.max_length,
                normalize=self.normalize_embeddings,
                device=self.device,
            )
        if vectors.ndim != 2 or vectors.shape[1] != self.passage_embeddings.shape[1]:
            raise ValueError(
                "Dense query embedding dimension does not match passage index "
                f"({vectors.shape[-1] if vectors.ndim else 0} != {self.passage_embeddings.shape[1]})"
            )
        if not np.isfinite(vectors).all():
            raise ValueError("Dense query embeddings contain NaN or infinity")
        return vectors

    def _build_faiss_adapter(self):
        try:
            import faiss
        except ImportError as error:
            raise RuntimeError("FAISS index requested but faiss-cpu is not installed") from error
        index = faiss.IndexFlatIP(self.passage_embeddings.shape[1])
        index.add(self.passage_embeddings)
        return index

    def _search_vectors(self, query_vectors: np.ndarray, top_k: int) -> list[list[tuple[str, float]]]:
        if top_k <= 0:
            return [[] for _ in range(query_vectors.shape[0])]
        if self._faiss_index is not None:
            scores, indexes = self._faiss_index.search(query_vectors, min(top_k, len(self.chunk_ids)))
            results = []
            for row_scores, row_indexes in zip(scores, indexes, strict=True):
                ranked = [
                    (self.chunk_ids[int(index)], float(score))
                    for score, index in zip(row_scores, row_indexes, strict=True)
                    if index >= 0
                ]
                results.append(sorted(ranked, key=lambda item: (-item[1], item[0])))
            return results
        rankings = exact_search(
            query_vectors,
            self.passage_embeddings,
            self.chunk_ids,
            top_k=top_k,
        )
        return [[(item.chunk_id, item.score) for item in row] for row in rankings]

    def search(self, query: str, top_k: int = 300) -> list[tuple[str, float]]:
        if top_k <= 0:
            return []
        queries = self._encode([self.query_prefix + query])
        return self._search_vectors(queries, top_k)[0]

    def search_batch(self, queries: Sequence[str], top_k: int = 300) -> list[list[tuple[str, float]]]:
        if not queries:
            return []
        if top_k <= 0:
            return [[] for _ in queries]
        vectors = self._encode([self.query_prefix + query for query in queries])
        return self._search_vectors(vectors, top_k)


def _write_numpy(path: Path, vectors: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.save(output, vectors.astype(np.float32, copy=False), allow_pickle=False)
    temporary.replace(path)

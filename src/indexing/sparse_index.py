"""Persistent, deterministic BM25 index and sparse indexing interface."""

from abc import ABC, abstractmethod
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata
from typing import List, Tuple

from src.data.schema import ChunkRecord


def tokenize(text: str, mode: str = "word", char_ngram_size: int = 3) -> list[str]:
    """Tokenize Unicode text as words or fixed-size character n-grams."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    if mode == "word":
        return re.findall(r"\w+", normalized, flags=re.UNICODE)
    if mode == "char_ngram":
        compact = "".join(char for char in normalized if not char.isspace())
        if len(compact) < char_ngram_size:
            return [compact] if compact else []
        return [compact[i : i + char_ngram_size] for i in range(len(compact) - char_ngram_size + 1)]
    raise ValueError(f"Unsupported tokenizer: {mode}")


def _document_payload(item: ChunkRecord | dict) -> dict[str, str]:
    if isinstance(item, ChunkRecord):
        return {
            "chunk_id": item.chunk_id,
            "doc_id": item.doc_id,
            "language": item.language,
            "text": item.text,
        }
    return {
        "chunk_id": str(item["chunk_id"]),
        "doc_id": str(item["doc_id"]),
        "language": str(item["language"]),
        "text": str(item["text"]),
    }


def corpus_sha256(chunks: List[ChunkRecord] | list[dict]) -> str:
    """Hash stable chunk identity, parent, language, and full text content."""
    rows = sorted((_document_payload(item) for item in chunks), key=lambda row: row["chunk_id"])
    canonical = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class BaseSparseIndex(ABC):
    """Abstract interface for sparse lexical index."""

    @abstractmethod
    def build(self, chunks: List[ChunkRecord], output_path: Path) -> None:
        """Builds index from chunks and persists to output path."""
        pass

    @abstractmethod
    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        """Searches index and returns list of (chunk_id, score)."""
        pass


class SparseBM25Index(BaseSparseIndex):
    """Small-corpus BM25 index with a JSON persistence format."""

    FORMAT_VERSION = 1

    def __init__(
        self,
        *,
        tokenizer: str = "word",
        char_ngram_size: int = 3,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.tokenizer = tokenizer
        self.char_ngram_size = char_ngram_size
        self.k1 = k1
        self.b = b
        self.documents: list[dict[str, str]] = []
        self._term_freqs: list[Counter[str]] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, list[tuple[int, int]]] = {}
        self._avg_doc_len = 0.0
        self.corpus_hash = ""

    def build(self, chunks: List[ChunkRecord], output_path: Path) -> None:
        if not chunks:
            raise ValueError("Cannot build a sparse index from an empty corpus")
        documents = [_document_payload(item) for item in chunks]
        ids = [item["chunk_id"] for item in documents]
        if len(ids) != len(set(ids)):
            raise ValueError("Cannot build sparse index: duplicate chunk_id values")
        self.documents = sorted(documents, key=lambda item: item["chunk_id"])
        self.corpus_hash = corpus_sha256(self.documents)
        self._build_postings()

        target = self._index_file(Path(output_path))
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": self.FORMAT_VERSION,
            "corpus_sha256": self.corpus_hash,
            "tokenizer": self.tokenizer,
            "char_ngram_size": self.char_ngram_size,
            "k1": self.k1,
            "b": self.b,
            "documents": self.documents,
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(
        cls,
        index_path: Path,
        *,
        expected_corpus_hash: str | None = None,
        tokenizer: str | None = None,
        char_ngram_size: int | None = None,
        k1: float | None = None,
        b: float | None = None,
    ) -> "SparseBM25Index":
        target = cls._index_file(Path(index_path))
        if not target.is_file():
            raise FileNotFoundError(f"Sparse BM25 index file not found: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if payload.get("format_version") != cls.FORMAT_VERSION:
            raise ValueError(f"Unsupported sparse index format version: {payload.get('format_version')}")
        documents = payload.get("documents")
        if not isinstance(documents, list) or not documents:
            raise ValueError(f"Sparse BM25 index has no documents: {target}")
        for item in documents:
            for field in ("chunk_id", "doc_id", "language", "text"):
                if not isinstance(item.get(field), str):
                    raise ValueError(f"Sparse BM25 index document is missing string field '{field}'")
        stored_hash = payload.get("corpus_sha256")
        actual_hash = corpus_sha256(documents)
        if stored_hash != actual_hash:
            raise ValueError("Sparse BM25 index corpus hash does not match its stored documents")
        if expected_corpus_hash is not None and stored_hash != expected_corpus_hash:
            raise ValueError(
                "Sparse BM25 index does not match the configured corpus "
                f"(index={stored_hash}, corpus={expected_corpus_hash}); rebuild the index"
            )
        if tokenizer is not None and payload.get("tokenizer") != tokenizer:
            raise ValueError(
                f"Sparse index tokenizer mismatch: index uses {payload.get('tokenizer')!r}, "
                f"config requests {tokenizer!r}; rebuild the index"
            )
        if char_ngram_size is not None and payload.get("char_ngram_size") != char_ngram_size:
            raise ValueError("Sparse index char_ngram_size does not match config; rebuild the index")
        if k1 is not None and payload.get("k1") != k1:
            raise ValueError("Sparse index k1 does not match config; rebuild the index")
        if b is not None and payload.get("b") != b:
            raise ValueError("Sparse index b does not match config; rebuild the index")
        index = cls(
            tokenizer=payload["tokenizer"],
            char_ngram_size=payload["char_ngram_size"],
            k1=payload["k1"],
            b=payload["b"],
        )
        index.documents = sorted((_document_payload(item) for item in documents), key=lambda item: item["chunk_id"])
        index.corpus_hash = stored_hash
        index._build_postings()
        return index

    def _build_postings(self) -> None:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        term_freqs: list[Counter[str]] = []
        lengths: list[int] = []
        for doc_index, document in enumerate(self.documents):
            frequencies = Counter(tokenize(document["text"], self.tokenizer, self.char_ngram_size))
            term_freqs.append(frequencies)
            lengths.append(sum(frequencies.values()))
            for term, count in frequencies.items():
                postings[term].append((doc_index, count))
        self._term_freqs = term_freqs
        self._doc_lengths = lengths
        self._postings = dict(postings)
        self._avg_doc_len = sum(lengths) / len(lengths) if lengths else 0.0

    @staticmethod
    def _index_file(path: Path) -> Path:
        return path / "index.json" if path.suffix == "" or path.is_dir() else path

    def search(self, query: str, top_k: int = 200) -> List[Tuple[str, float]]:
        if top_k <= 0 or not self.documents:
            return []
        query_terms = Counter(tokenize(query, self.tokenizer, self.char_ngram_size))
        if not query_terms:
            return []
        scores: dict[int, float] = defaultdict(float)
        doc_count = len(self.documents)
        for term, query_frequency in query_terms.items():
            postings = self._postings.get(term, ())
            if not postings:
                continue
            doc_frequency = len(postings)
            idf = math.log1p((doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
            for doc_index, term_frequency in postings:
                doc_len = self._doc_lengths[doc_index]
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_len / max(self._avg_doc_len, 1e-12)
                )
                scores[doc_index] += query_frequency * idf * (
                    term_frequency * (self.k1 + 1) / denominator
                )
        ranked = [
            (self.documents[index]["chunk_id"], score)
            for index, score in scores.items()
            if score > 0
        ]
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:top_k]

"""Data loaders for JSONL, Parquet, and document-to-chunk index maps."""

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Type, TypeVar, Union
import pandas as pd
from pydantic import BaseModel

from src.data.schema import ChunkRecord, GroundTruthRecord, PredictionRecord, QueryRecord

T = TypeVar("T", bound=BaseModel)


def load_jsonl_records(file_path: Union[str, Path], model_cls: Type[T]) -> List[T]:
    """Reads a JSONL file and parses each line into a validated Pydantic model."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    records: List[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line_str = line.strip()
            if not line_str:
                continue
            try:
                data = json.loads(line_str)
                record = model_cls.model_validate(data)
                records.append(record)
            except Exception as e:
                raise ValueError(f"Error parsing line {line_idx} in {path}: {e}") from e

    return records


def save_jsonl_records(
    file_path: Union[str, Path],
    records: Iterable[Union[BaseModel, dict]],
    ensure_ascii: bool = False,
) -> None:
    """Writes a list of Pydantic models or dictionaries to a JSONL file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        for item in records:
            if isinstance(item, BaseModel):
                data = item.model_dump(mode="json")
            elif isinstance(item, dict):
                data = item
            else:
                raise TypeError(f"Unsupported record type: {type(item)}")
            f.write(json.dumps(data, ensure_ascii=ensure_ascii) + "\n")


class DocumentChunkMap:
    """Maintains bi-directional index between documents and their constituent chunks."""

    def __init__(
        self,
        chunk_to_doc: Optional[Dict[str, str]] = None,
        doc_to_chunks: Optional[Dict[str, List[str]]] = None,
    ):
        self.chunk_to_doc: Dict[str, str] = chunk_to_doc or {}
        self.doc_to_chunks: Dict[str, List[str]] = doc_to_chunks or {}

    @classmethod
    def from_chunks(cls, chunks: Iterable[ChunkRecord]) -> "DocumentChunkMap":
        """Builds maps from an iterable of ChunkRecord."""
        chunk_to_doc: Dict[str, str] = {}
        doc_to_chunks: Dict[str, List[str]] = {}

        # Sort or group chunks preserving order
        for chunk in chunks:
            chunk_to_doc[chunk.chunk_id] = chunk.doc_id
            if chunk.doc_id not in doc_to_chunks:
                doc_to_chunks[chunk.doc_id] = []
            doc_to_chunks[chunk.doc_id].append(chunk.chunk_id)

        return cls(chunk_to_doc=chunk_to_doc, doc_to_chunks=doc_to_chunks)

    def get_parent_doc(self, chunk_id: str) -> Optional[str]:
        """Returns the parent document ID for a chunk ID."""
        return self.chunk_to_doc.get(chunk_id)

    def get_child_chunks(self, doc_id: str) -> List[str]:
        """Returns all chunk IDs belonging to a document ID."""
        return self.doc_to_chunks.get(doc_id, [])

    @property
    def all_doc_ids(self) -> Set[str]:
        return set(self.doc_to_chunks.keys())

    @property
    def all_chunk_ids(self) -> Set[str]:
        return set(self.chunk_to_doc.keys())

    def save_parquet(self, output_path: Union[str, Path]) -> None:
        """Saves the chunk-to-doc mapping table to Parquet."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(
            [{"chunk_id": cid, "doc_id": did} for cid, did in self.chunk_to_doc.items()]
        )
        df.to_parquet(path, index=False)

    @classmethod
    def load_parquet(cls, parquet_path: Union[str, Path]) -> "DocumentChunkMap":
        """Loads the mapping from a Parquet file."""
        df = pd.read_parquet(parquet_path)
        chunk_to_doc = dict(zip(df["chunk_id"], df["doc_id"]))
        doc_to_chunks: Dict[str, List[str]] = {}
        for cid, did in chunk_to_doc.items():
            if did not in doc_to_chunks:
                doc_to_chunks[did] = []
            doc_to_chunks[did].append(cid)
        return cls(chunk_to_doc=chunk_to_doc, doc_to_chunks=doc_to_chunks)


class DataLoader:
    """High level data loader managing queries, chunks, and cached Parquet tables."""

    @staticmethod
    def load_queries(file_path: Union[str, Path]) -> List[QueryRecord]:
        return load_jsonl_records(file_path, QueryRecord)

    @staticmethod
    def load_chunks(file_path: Union[str, Path]) -> List[ChunkRecord]:
        return load_jsonl_records(file_path, ChunkRecord)

    @staticmethod
    def load_ground_truth(file_path: Union[str, Path]) -> List[GroundTruthRecord]:
        return load_jsonl_records(file_path, GroundTruthRecord)

    @staticmethod
    def load_predictions(file_path: Union[str, Path]) -> List[PredictionRecord]:
        return load_jsonl_records(file_path, PredictionRecord)

"""Pydantic data schemas for queries, chunks, predictions, and ground truth."""

from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator


class QueryRecord(BaseModel):
    """Represents a search query issued in Vietnamese."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Unique identifier of the query (e.g. Q001)")
    query: str = Field(..., min_length=1, description="Raw query string text in Vietnamese")

    @field_validator("id", "query")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field cannot be empty or whitespace only.")
        return s


class ChunkRecord(BaseModel):
    """Represents a text chunk belonging to a parent document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(..., min_length=1, description="Unique identifier of chunk (e.g. C00012)")
    doc_id: str = Field(..., min_length=1, description="Parent document identifier (e.g. DOC_005)")
    chunk_index: int = Field(..., ge=0, description="0-indexed position within parent document")
    language: str = Field(..., min_length=2, max_length=10, description="Language code: vi, en, zh")
    text: str = Field(..., min_length=1, description="Text content of the chunk")

    @field_validator("chunk_id", "doc_id", "language", "text")
    @classmethod
    def strip_text(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Field cannot be empty or whitespace only.")
        return s


class PredictionRecord(BaseModel):
    """Represents the retrieval output for a single query."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., min_length=1, description="Query identifier matching QueryRecord.id")
    relevant_docs: List[str] = Field(default_factory=list, description="Ordered list of predicted document IDs")
    relevant_chunks: List[str] = Field(default_factory=list, description="Ordered list of predicted chunk IDs")

    @field_validator("id")
    @classmethod
    def strip_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("ID cannot be empty.")
        return s

    @field_validator("relevant_docs", "relevant_chunks")
    @classmethod
    def validate_ids(cls, ids: List[str]) -> List[str]:
        cleaned: List[str] = []
        seen = set()
        for item in ids:
            s = item.strip()
            if not s:
                raise ValueError("Empty ID detected in list.")
            if s in seen:
                raise ValueError(f"Duplicate ID detected in list: {s}")
            seen.add(s)
            cleaned.append(s)
        return cleaned


class GroundTruthRecord(BaseModel):
    """Ground truth relevance annotation for a single query."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(..., min_length=1, description="Query identifier")
    relevant_docs: List[str] = Field(..., min_length=1, description="Set of truly relevant document IDs")
    relevant_chunks: List[str] = Field(..., min_length=1, description="Set of truly relevant chunk IDs")

    @field_validator("id")
    @classmethod
    def strip_id(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("ID cannot be empty.")
        return s

    @field_validator("relevant_docs", "relevant_chunks")
    @classmethod
    def validate_unique(cls, items: List[str]) -> List[str]:
        cleaned = [x.strip() for x in items if x.strip()]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Duplicate entries in ground truth list.")
        return cleaned

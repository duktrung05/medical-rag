"""Data schemas, loaders, and preprocessing utilities."""

from src.data.schema import (
    ChunkRecord,
    GroundTruthRecord,
    PredictionRecord,
    QueryRecord,
)
from src.data.loader import (
    DataLoader,
    DocumentChunkMap,
    load_jsonl_records,
    save_jsonl_records,
)
from src.data.preprocess import normalize_text

__all__ = [
    "QueryRecord",
    "ChunkRecord",
    "PredictionRecord",
    "GroundTruthRecord",
    "DataLoader",
    "DocumentChunkMap",
    "load_jsonl_records",
    "save_jsonl_records",
    "normalize_text",
]

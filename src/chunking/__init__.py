"""Chunk extracted documents while preserving source offsets."""

from .paragraph_chunker import ChunkingConfig, ChunkSpan, chunk_text, detect_language

__all__ = ["ChunkSpan", "ChunkingConfig", "chunk_text", "detect_language"]

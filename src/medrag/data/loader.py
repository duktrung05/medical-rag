from __future__ import annotations

import gzip
from hashlib import sha256
from pathlib import Path

from src.medrag.core.contracts import Source
from src.medrag.data.corpus_registry import CorpusRegistry


def read_corpus_mapping(path: Path, *, batch_size: int = 65536):
    """Stream the official ID/URL pairs; pyarrow is required for Parquet input."""
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Install pyarrow to read the official corpus Parquet") from exc
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=["id", "url"]):
        columns = batch.to_pydict()
        yield from zip(columns["id"], columns["url"], strict=True)


def load_crawled_source(*, version: str, official_id: int, url: str, mapping: dict[int, str],
                        pages_root: Path, encoding: str | None = None) -> Source:
    from src.medrag.data.extraction import extract_html

    CorpusRegistry.assert_identity(official_id, url, mapping)
    path = pages_root / f"{official_id % 1000:03d}" / f"{official_id}.gz"
    raw = gzip.decompress(path.read_bytes())
    title, text = extract_html(raw, encoding)
    if not text:
        raise ValueError(f"No extracted text for {official_id}")
    return Source(version, official_id, url, title, text, str(path), sha256(raw).hexdigest())


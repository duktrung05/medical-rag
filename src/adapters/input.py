"""Convert validated external query records into retrieval-core inputs."""

from dataclasses import dataclass

from src.data.preprocess import normalize_text
from src.data.schema import QueryRecord


@dataclass(frozen=True)
class NormalizedQuery:
    query_id: str
    text: str


def adapt_query(record: QueryRecord) -> NormalizedQuery:
    """Adapt a public QueryRecord to the normalized retrieval input contract."""
    return NormalizedQuery(query_id=record.id, text=normalize_text(record.query, lowercase=True))

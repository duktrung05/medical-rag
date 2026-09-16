"""Multilingual medical query expansion."""

from typing import List, Optional
from src.query.medical_terms import MedicalDictionary


class QueryExpander:
    """Expands Vietnamese clinical queries with cross-lingual synonyms."""

    def __init__(self, dictionary: Optional[MedicalDictionary] = None):
        self.dict = dictionary or MedicalDictionary()

    def expand(self, query: str) -> str:
        """Appends relevant medical synonyms to original query if matches found."""
        expanded_parts = [query]
        for term in self.dict.terms:
            if term in query.lower():
                synonyms = self.dict.get_synonyms(term)
                if synonyms:
                    expanded_parts.extend(synonyms[:3])
        return " ".join(expanded_parts)

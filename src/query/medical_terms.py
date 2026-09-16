"""Medical terms dictionary and synonym lookup."""

import json
from pathlib import Path
from typing import Dict, List, Optional


class MedicalDictionary:
    """Manages multilingual clinical term equivalents (vi, en, zh)."""

    def __init__(self, terms: Optional[Dict[str, Dict[str, List[str]]]] = None):
        self.terms = terms or {}

    @classmethod
    def load_json(cls, file_path: Path) -> "MedicalDictionary":
        if not file_path.is_file():
            return cls({})
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(terms=data)

    def get_synonyms(self, term: str, target_lang: Optional[str] = None) -> List[str]:
        term_lower = term.lower().strip()
        record = self.terms.get(term_lower, {})
        if target_lang:
            return record.get(target_lang, [])
        all_syns: List[str] = []
        for lang_syns in record.values():
            all_syns.extend(lang_syns)
        return list(dict.fromkeys(all_syns))

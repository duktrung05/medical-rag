from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class IndexManifest:
    corpus_version: str
    corpus_revision: str
    corpus_sha256: str
    source_count: int
    passage_count: int
    analyzer: str
    encoder: str | None

    def save(self, namespace: Path) -> None:
        if "v4" not in namespace.parts:
            raise ValueError("V4 indexes must use a separate v4 namespace")
        namespace.mkdir(parents=True, exist_ok=True)
        target = namespace / "manifest.json"
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


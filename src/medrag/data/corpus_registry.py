from __future__ import annotations

import json
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit


def normalized_url(url: str) -> str:
    parts = urlsplit(url.strip())
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid source URL: {url!r}")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


@dataclass(frozen=True)
class CorpusVersion:
    name: str
    revision: str
    path: Path
    sha256: str
    rows: int


class CorpusRegistry:
    def __init__(self, versions: dict[str, CorpusVersion]):
        self.versions = versions

    @classmethod
    def load(cls, path: str | Path) -> CorpusRegistry:
        base = Path(path).resolve().parent
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({key: CorpusVersion(key, value["revision"], (base / value["path"]).resolve(), value["sha256"], value["rows"])
                    for key, value in values["versions"].items()})

    def get(self, version: str) -> CorpusVersion:
        if version not in self.versions:
            raise KeyError(f"Unregistered corpus version: {version}")
        return self.versions[version]

    @staticmethod
    def verify_file(version: CorpusVersion) -> bool:
        digest = sha256()
        with version.path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest() == version.sha256

    @staticmethod
    def assert_identity(official_id: int, url: str, mapping: dict[int, str]) -> None:
        if official_id not in mapping or normalized_url(mapping[official_id]) != normalized_url(url):
            raise ValueError(f"ID/URL mismatch for {official_id}")


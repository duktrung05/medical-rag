"""Versioned, read-only prepared corpus format for the V4 runtime."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path

from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.contracts import Source
from src.medrag.core.factory import build_pipeline
from src.medrag.core.pipeline import Pipeline
from src.medrag.data.corpus_registry import CorpusRegistry, CorpusVersion
from src.medrag.data.loader import load_crawled_source, read_corpus_mapping


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_prepared(sources: list[Source], output: Path, version: CorpusVersion,
                   missing_raw: int = 0, reuse_corpus_sha256: str | None = None,
                   id_url_mismatches: int = 0) -> dict:
    """Write a small V4 snapshot without touching source data or existing output."""
    if "v4" not in output.parts or "v4" not in output.resolve().parts:
        raise ValueError("Prepared output must be in a v4 namespace")
    if output.exists() or output.with_suffix(output.suffix + ".manifest.json").exists():
        raise FileExistsError(output)
    if any(source.corpus_version != version.name for source in sources):
        raise ValueError("Source corpus version differs from snapshot version")
    if len({source.official_id for source in sources}) != len(sources):
        raise ValueError("Duplicate official IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    with temporary.open("x", encoding="utf-8") as handle:
        for source in sources:
            handle.write(json.dumps(asdict(source), ensure_ascii=False, separators=(",", ":")) + "\n")
    manifest = {"format": "medrag-v4-sources-jsonl-v1", "corpus_version": version.name,
                "corpus_revision": version.revision, "corpus_sha256": version.sha256,
                "sources": len(sources), "missing_raw": missing_raw,
                "reuse_corpus_sha256": reuse_corpus_sha256,
                "id_url_mismatches": id_url_mismatches,
                "prepared_sha256": _sha256_file(temporary)}
    sidecar = output.with_suffix(output.suffix + ".manifest.json")
    with sidecar.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    return manifest


def prepare_from_official(version: CorpusVersion, reuse_version: CorpusVersion,
                          pages_root: Path, output: Path,
                          limit: int = 100) -> dict:
    """Build a bounded official-corpus snapshot; full runs require a separate backend."""
    if not 1 <= limit <= 1000:
        raise ValueError("The in-memory V4 pilot supports 1 to 1000 sources")
    if not CorpusRegistry.verify_file(version) or not CorpusRegistry.verify_file(reuse_version):
        raise ValueError("Pinned corpus or historical reuse corpus checksum mismatch")
    candidates: list[tuple[int, str]] = []
    missing = 0
    for official_id, url in read_corpus_mapping(version.path):
        if len(candidates) >= limit * 10:
            break
        path = pages_root / f"{official_id % 1000:03d}" / f"{official_id}.gz"
        if path.is_file():
            candidates.append((official_id, url))
        else:
            missing += 1
    selected_ids = {official_id for official_id, _ in candidates}
    historical_urls = {official_id: url for official_id, url in read_corpus_mapping(reuse_version.path)
                       if official_id in selected_ids}
    sources: list[Source] = []
    mismatches = 0
    for official_id, url in candidates:
        if len(sources) >= limit:
            break
        try:
            CorpusRegistry.assert_identity(official_id, url, historical_urls)
        except ValueError:
            mismatches += 1
            continue
        try:
            source = load_crawled_source(version=version.name, official_id=official_id,
                                         url=url, mapping=historical_urls, pages_root=pages_root)
            sources.append(replace(source, metadata={"reuse_corpus_version": reuse_version.name,
                                                     "reuse_id_url_verified": True}))
        except (OSError, UnicodeError, ValueError):
            missing += 1
    if not sources:
        raise ValueError("No usable raw HTML with matching official and historical ID/URL was found")
    return write_prepared(sources, output, version, missing, reuse_version.sha256, mismatches)


def load_prepared(path: Path, registry: CorpusRegistry, *, max_sources: int = 1000,
                  config: dict | None = None) -> Pipeline:
    if path.is_dir():
        from src.medrag.data.phase2_prepared import load_phase2_prepared

        return load_phase2_prepared(path, registry, max_sources=max_sources, config=config)
    sidecar = path.with_suffix(path.suffix + ".manifest.json")
    manifest = json.loads(sidecar.read_text(encoding="utf-8"))
    version = registry.get(manifest["corpus_version"])
    if (manifest.get("format") != "medrag-v4-sources-jsonl-v1"
            or manifest.get("corpus_revision") != version.revision
            or manifest.get("corpus_sha256") != version.sha256):
        raise ValueError("Prepared corpus manifest does not match pinned registry")
    if _sha256_file(path) != manifest.get("prepared_sha256"):
        raise ValueError("Prepared corpus checksum mismatch")
    if not 1 <= manifest["sources"] <= max_sources:
        raise ValueError("Snapshot exceeds in-memory V4 pilot capacity")
    sources = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                sources.append(Source(**json.loads(line)))
    if len(sources) != manifest["sources"] or any(s.corpus_version != version.name for s in sources):
        raise ValueError("Prepared corpus source count or version mismatch")
    if version.name.startswith("official-"):
        reuse_names = {s.metadata.get("reuse_corpus_version") for s in sources
                       if s.metadata.get("reuse_id_url_verified") is True}
        if len(reuse_names) != 1 or any(s.metadata.get("reuse_id_url_verified") is not True for s in sources):
            raise ValueError("Official prepared sources lack verified historical ID/URL lineage")
        reuse = registry.get(reuse_names.pop())
        if manifest.get("reuse_corpus_sha256") != reuse.sha256 or not CorpusRegistry.verify_file(reuse):
            raise ValueError("Historical reuse corpus checksum mismatch")
        if not CorpusRegistry.verify_file(version):
            raise ValueError("Pinned official corpus checksum mismatch")
        expected = {source.official_id: source.url for source in sources}
        for corpus_path in (version.path, reuse.path):
            found = set()
            for official_id, url in read_corpus_mapping(corpus_path):
                if official_id in expected:
                    CorpusRegistry.assert_identity(official_id, expected[official_id], {official_id: url})
                    found.add(official_id)
                    if len(found) == len(expected):
                        break
            if len(found) != len(expected):
                raise ValueError(f"Prepared source IDs are absent from {corpus_path}")
    passages = [p for source in sources for p in build_passages(source)]
    return build_pipeline(sources, passages, config or {})


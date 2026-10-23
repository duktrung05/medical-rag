"""V4 snapshot adapter for immutable Phase 2 release data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.medrag.core.contracts import Passage, Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.data.corpus_registry import CorpusRegistry, CorpusVersion
from src.medrag.data.phase2_batch import Phase2Batch, sha256_file
from src.medrag.evidence.verifier import verify


FORMAT = "medrag-v4-phase2-snapshot-v1"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_phase2_prepared(batch: Phase2Batch, output: Path, version: CorpusVersion) -> dict:
    """Write a checksummed V4 snapshot without rechunking or changing identities."""
    if "v4" not in output.parts or "v4" not in output.resolve().parts:
        raise ValueError("V4 Phase 2 snapshots must be written in a v4 namespace")
    if output.exists():
        raise FileExistsError(output)
    if not batch.documents or not batch.passages:
        raise ValueError("A V4 Phase 2 snapshot requires documents and passages")
    staging = output.with_name(f".{output.name}.incomplete-{os.getpid()}")
    if staging.exists():
        raise FileExistsError(staging)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True)
    try:
        sources = []
        by_id = {}
        for row in batch.documents:
            doc_id = int(row["doc_id"])
            text = str(row["normalized_text"])
            normalized_sha = _sha256_text(text)
            if normalized_sha != row["normalized_text_sha256"]:
                raise ValueError(f"Normalized text checksum mismatch for doc_id={doc_id}")
            source = Source(
                corpus_version=version.name,
                official_id=doc_id,
                url=str(row["official_url"]),
                title=str(row.get("title") or ""),
                text=text,
                origin=f"phase2:{batch.manifest['phase2_release_manifest_sha256']}",
                raw_sha256=str(row["source_text_sha256"]),
                metadata={
                    "text_representation": "normalized_text",
                    "normalized_text_sha256": normalized_sha,
                    "source_text_sha256": str(row["source_text_sha256"]),
                    "detected_language": row.get("detected_language"),
                },
            )
            sources.append(source)
            by_id[doc_id] = source

        passages = []
        for row in batch.passages:
            doc_id = int(row["doc_id"])
            source = by_id[doc_id]
            passage = Passage(
                passage_id=str(row["passage_id"]),
                corpus_version=version.name,
                official_id=doc_id,
                url=str(row["official_url"]),
                text=str(row["passage_text"]),
                source_start=int(row["char_start"]),
                source_end=int(row["char_end"]),
                source_sha256=str(row["text_representation_sha256"]),
                language=str(row["language"]),
                provenance=json.dumps({key: row[key] for key in (
                    "text_representation", "text_representation_sha256", "source_text_sha256",
                    "strategy_id", "chunker_version", "tokenizer_name", "tokenizer_revision",
                    "phase1_release_id", "corpus_revision", "processing_run_id",
                )}, ensure_ascii=False, sort_keys=True),
            )
            if not verify(passage, source):
                raise ValueError(f"Phase 2 passage does not match its normalized source: {passage.passage_id}")
            passages.append(passage)

        sources_path = staging / "sources.jsonl"
        sources_path.write_text("".join(
            json.dumps({
                "corpus_version": source.corpus_version,
                "official_id": source.official_id,
                "url": source.url,
                "title": source.title,
                "text": source.text,
                "origin": source.origin,
                "raw_sha256": source.raw_sha256,
                "metadata": source.metadata,
            }, ensure_ascii=False, allow_nan=False) + "\n" for source in sources
        ), encoding="utf-8")
        passage_rows = [{
            "passage_id": p.passage_id, "corpus_version": p.corpus_version,
            "official_id": p.official_id, "url": p.url, "text": p.text,
            "source_start": p.source_start, "source_end": p.source_end,
            "source_sha256": p.source_sha256, "language": p.language,
            "provenance": p.provenance,
        } for p in passages]
        pq.write_table(pa.Table.from_pylist(passage_rows), staging / "passages.parquet", compression="zstd")
        if batch.queries:
            pq.write_table(pa.Table.from_pylist(batch.queries), staging / "queries.parquet", compression="zstd")
        manifest = {
            "format": FORMAT,
            "status": "complete",
            "corpus_version": version.name,
            "corpus_revision": version.revision,
            "corpus_sha256": version.sha256,
            "phase1_release_manifest_sha256": batch.manifest["phase1_release_manifest_sha256"],
            "phase2_release_manifest_sha256": batch.manifest["phase2_release_manifest_sha256"],
            "input_manifest_sha256": hashlib.sha256(json.dumps(
                batch.manifest, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest(),
            "source_count": len(sources),
            "passage_count": len(passages),
            "query_count": len(batch.queries),
            "source_representation": "normalized_text",
            "preflight": batch.manifest["preflight"],
            "files": {
                name: {"sha256": sha256_file(staging / name), "bytes": (staging / name).stat().st_size}
                for name in ("sources.jsonl", "passages.parquet")
            },
        }
        if batch.queries:
            manifest["files"]["queries.parquet"] = {
                "sha256": sha256_file(staging / "queries.parquet"),
                "bytes": (staging / "queries.parquet").stat().st_size,
            }
        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.rename(staging, output)
        return manifest
    except Exception:
        # Leave the isolated staging directory for diagnosis; never publish complete.
        raise


def load_phase2_prepared(path: Path, registry: CorpusRegistry, *, max_sources: int = 1000,
                         config: dict | None = None) -> Pipeline:
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT or manifest.get("status") != "complete":
        raise ValueError("Unsupported or incomplete V4 Phase 2 snapshot")
    version = registry.get(str(manifest["corpus_version"]))
    if (manifest.get("corpus_revision") != version.revision
            or manifest.get("corpus_sha256") != version.sha256
            or manifest.get("source_representation") != "normalized_text"):
        raise ValueError("V4 Phase 2 snapshot does not match the pinned corpus registry")
    if not 1 <= int(manifest["source_count"]) <= max_sources:
        raise ValueError(f"Snapshot exceeds in-memory V4 capacity ({max_sources} sources)")
    for name, expected in manifest.get("files", {}).items():
        file_path = path / name
        if not file_path.is_file() or sha256_file(file_path) != expected.get("sha256"):
            raise ValueError(f"V4 Phase 2 snapshot file is missing or altered: {name}")
    sources = []
    with (path / "sources.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = Source(**row)
            if _sha256_text(source.text) != source.metadata.get("normalized_text_sha256"):
                raise ValueError(f"V4 normalized source hash mismatch: doc_id={source.official_id}")
            sources.append(source)
    if len(sources) != int(manifest["source_count"]):
        raise ValueError("V4 Phase 2 source count differs from manifest")
    source_by_id = {s.official_id: s for s in sources}
    if len(source_by_id) != len(sources):
        raise ValueError("V4 Phase 2 source IDs are duplicated")
    table = pq.read_table(path / "passages.parquet")
    passages = [Passage(**row) for row in table.to_pylist()]
    if len(passages) != int(manifest["passage_count"]):
        raise ValueError("V4 Phase 2 passage count differs from manifest")
    if any(p.official_id not in source_by_id or not verify(p, source_by_id[p.official_id]) for p in passages):
        raise ValueError("V4 Phase 2 passages fail source identity/offset verification")
    from src.medrag.core.factory import build_pipeline

    return build_pipeline(sources, passages, config or {})

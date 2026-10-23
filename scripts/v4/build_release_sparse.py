"""Build V4 full-body/document and all-passage BM25 from pinned Phase 1/2 shards."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pyarrow.parquet as pq

from src.medrag.core.contracts import Passage, Source
from src.medrag.core.paths import project_root
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2.provenance import passage_id as make_passage_id
from src.medrag.data.phase2_batch import normalize_url, sha256_file
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.disk_bm25 import DiskBM25


PROVENANCE_FIELDS = (
    "text_representation", "text_representation_sha256", "source_text_sha256",
    "strategy_id", "chunker_version", "tokenizer_name", "tokenizer_revision",
    "phase1_release_id", "corpus_revision", "processing_run_id",
)
P1_FIELDS = ["doc_id", "official_url", "title", "normalized_text",
             "normalized_text_sha256", "source_text_sha256", "detected_language"]
P2_FIELDS = ["passage_id", "doc_id", "official_url", "passage_index", "passage_text",
             "char_start", "char_end", "source_text_sha256", "text_representation",
             "text_representation_sha256", "language", *PROVENANCE_FIELDS[3:]]


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _lock(root: Path, *, max_parts: int | None) -> dict:
    p1_root = root / "outputs/vibiomir_phase1_full_20261014_v1"
    p2_root = root / "outputs/vibiomir_phase2_full_v2"
    p1_path = p1_root / "release/phase1_release_manifest.json"
    p2_path = p2_root / "manifests/phase2_release_manifest.json"
    p1, p2 = _json(p1_path), _json(p2_path)
    p1_sha, p2_sha = sha256_file(p1_path), sha256_file(p2_path)
    if (p1.get("status") != "complete" or p2.get("status") != "complete_validated"
            or p2.get("phase1_release_manifest_sha256") != p1_sha):
        raise ValueError("Phase 1/2 full release linkage is invalid")
    if not (len(p1["pass_allowlist_parts"]) == len(p2["passage_parts"]) == 922):
        raise ValueError("Unexpected Phase 1/2 release part coverage")
    exclusions_path = root / "configs/vibiomir/phase2_semantic_staging_20261015.json"
    exclusions = {int(row["doc_id"]) for row in _json(exclusions_path)["documents"]}
    if len(exclusions) != 6:
        raise ValueError("Unexpected semantic exclusion count")
    migration = _json(root / "configs/vibiomir/phase2_batch_migration_v1.json")
    official_path = root / migration["official_corpus"]
    queries_path = root / migration["official_queries"]
    if (sha256_file(official_path) != migration["official_corpus_sha256"]
            or sha256_file(queries_path) != migration["official_queries_sha256"]):
        raise ValueError("Official corpus/query checksums differ from pinned snapshot")
    registry = CorpusRegistry.load(root / "configs/v4_corpus_registry.json")
    # The corpus revision is fixed by the immutable Phase 2 input snapshot.
    p2_input = _json(p2_root / "manifests/input_snapshot.json")
    version = registry.get("official-" + p2_input["corpus_revision"])
    if version.sha256 != migration["official_corpus_sha256"]:
        raise ValueError("V4 registry and official corpus differ")
    paths = {"phase1": p1_root, "phase2": p2_root, "official": official_path,
             "queries": queries_path}
    total = len(p2["passage_parts"])
    if max_parts is not None and not 1 <= max_parts <= total:
        raise ValueError("max_parts must be 1..922")
    parts = total if max_parts is None else max_parts
    core = {"format": "medrag-v4-release-sparse-lock-v1", "corpus_version": version.name,
            "phase1_release_sha256": p1_sha, "phase2_release_sha256": p2_sha,
            "official_corpus_sha256": migration["official_corpus_sha256"],
            "official_queries_sha256": migration["official_queries_sha256"],
            "semantic_exclusions_sha256": sha256_file(exclusions_path),
            "selected_parts": parts, "release_parts": total,
            "implementation_sha256": {name: sha256_file(root / name) for name in (
                "scripts/v4/build_release_sparse.py", "src/medrag/data/disk_store.py",
                "src/medrag/retrieval/disk_bm25.py", "src/medrag/retrieval/core.py",
                "src/medrag/evidence/verifier.py", "src/medrag/data/phase2/provenance.py")}}
    fingerprint = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()
    return {"core": core, "fingerprint": fingerprint, "p1": p1, "p2": p2,
            "p1_sha": p1_sha, "p2_sha": p2_sha, "exclusions": exclusions,
            "paths": paths, "parts": parts}


def _official_urls(path: Path) -> dict[int, str]:
    result = {}
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(columns=["id", "url"], batch_size=100_000):
        for official_id, url in zip(batch.column(0).to_pylist(), batch.column(1).to_pylist(), strict=True):
            key = int(official_id)
            if key in result:
                raise ValueError(f"Duplicate official corpus ID: {key}")
            result[key] = normalize_url(str(url))
    if len(result) != 1_220_598:
        raise ValueError("Official corpus row count differs from locked release")
    return result


def _read_part(lock: dict, root: Path, index: int, official_urls: dict[int, str]) -> tuple[list[Source], list[Passage], set[int], int]:
    p1_root, p2_root = lock["paths"]["phase1"], lock["paths"]["phase2"]
    name = f"part-{index:05d}"
    p1_pass = lock["p1"]["pass_allowlist_parts"][index]
    p2_pass = lock["p2"]["passage_parts"][index]
    if p1_pass["part"] != name or p2_pass["part"] != name:
        raise ValueError(f"Non-aligned release parts: {name}")
    clean_cp_path = p1_root / "checkpoints" / f"{name}.json"
    # Despite its key name, Phase 1 manifest records clean-part output hashes here.
    candidate_manifest = lock.get("candidate_manifest")
    if candidate_manifest is None:
        candidate_manifest = _json(p1_root / "phase1_manifest.json")
        lock["candidate_manifest"] = candidate_manifest
    clean_cp = _json(clean_cp_path)
    expected_clean_sha = candidate_manifest["part_checkpoints_sha256"][name]
    if clean_cp.get("output_sha256") != expected_clean_sha:
        raise ValueError(f"Phase 1 clean shard/checkpoint link mismatch: {name}")
    paths = {
        "clean": p1_root / "clean_documents" / f"{name}.parquet",
        "pass": p1_root / "release/pass_document_ids" / f"{name}.parquet",
        "passage": p2_root / "passages/strategy_C" / f"{name}.parquet",
    }
    expected = {"clean": expected_clean_sha, "pass": p1_pass["sha256"],
                "passage": p2_pass["sha256"]}
    for key, path in paths.items():
        if sha256_file(path) != expected[key]:
            raise ValueError(f"Release shard checksum mismatch: {path}")
    pass_ids = set(map(int, pq.read_table(paths["pass"], columns=["doc_id"]).column(0).to_pylist()))
    if len(pass_ids) != int(p1_pass["rows"]):
        raise ValueError(f"Duplicate or missing PASS IDs: {name}")
    clean = pq.read_table(paths["clean"], columns=P1_FIELDS).to_pylist()
    by_id = {int(row["doc_id"]): row for row in clean if int(row["doc_id"]) in pass_ids}
    if len(by_id) != len(pass_ids):
        raise ValueError(f"PASS/clean source mismatch: {name}")
    eligible_ids = pass_ids - lock["exclusions"]
    sources = []
    for ident in sorted(eligible_ids):
        row = by_id[ident]
        text = row["normalized_text"]
        if (not isinstance(text, str) or hashlib.sha256(text.encode()).hexdigest()
                != row["normalized_text_sha256"]):
            raise ValueError(f"Normalized source hash mismatch: {ident}")
        if normalize_url(str(row["official_url"])) != official_urls.get(ident):
            raise ValueError(f"Official source ID/URL mismatch: {ident}")
        sources.append(Source(lock["core"]["corpus_version"], ident, row["official_url"],
                              row.get("title") or "", text, f"phase2:{lock['p2_sha']}",
                              row["source_text_sha256"],
                              {"text_representation": "normalized_text",
                               "normalized_text_sha256": row["normalized_text_sha256"],
                               "source_text_sha256": row["source_text_sha256"],
                               "detected_language": row.get("detected_language")}))
    source_by_id = {source.official_id: source for source in sources}
    passage_rows = pq.read_table(paths["passage"], columns=P2_FIELDS).to_pylist()
    if len(passage_rows) != int(p2_pass["rows"]):
        raise ValueError(f"Phase 2 passage row count mismatch: {name}")
    if {int(row["doc_id"]) for row in passage_rows} != pass_ids:
        raise ValueError(f"Phase 1 PASS and Phase 2 parent sets differ: {name}")
    passages = []
    excluded_rows = 0
    for row in sorted(passage_rows, key=lambda row: (int(row["doc_id"]), int(row["passage_index"]))):
        ident = int(row["doc_id"])
        if ident in lock["exclusions"]:
            excluded_rows += 1
            continue
        source = source_by_id[ident]
        start, end = int(row["char_start"]), int(row["char_end"])
        if (row["text_representation"] != "normalized_text"
                or row["text_representation_sha256"] != source.text_sha256
                or row["source_text_sha256"] != source.raw_sha256):
            raise ValueError(f"Phase 2 representation/hash mismatch: {row['passage_id']}")
        if row["official_url"] != source.url:
            raise ValueError(f"Phase 2 passage/source URL mismatch: {row['passage_id']}")
        expected_id = make_passage_id(
            chunker_version=row["chunker_version"], strategy_id=row["strategy_id"],
            doc_id=ident, passage_index=int(row["passage_index"]),
            source_sha256=row["text_representation_sha256"], char_start=start, char_end=end)
        if row["passage_id"] != expected_id:
            raise ValueError(f"Phase 2 passage ID mismatch: {row['passage_id']}")
        lineage = {"strategy_id": lock["p2"]["strategy"]["strategy_id"],
                   "chunker_version": lock["p2"]["chunker_version"],
                   "tokenizer_name": lock["p2"]["tokenizer_name"],
                   "tokenizer_revision": lock["p2"]["tokenizer_revision"],
                   "phase1_release_id": lock["p1_sha"],
                   "corpus_revision": lock["core"]["corpus_version"].removeprefix("official-")}
        if any(row[key] != expected for key, expected in lineage.items()):
            raise ValueError(f"Phase 2 passage lineage mismatch: {row['passage_id']}")
        passage = Passage(row["passage_id"], source.corpus_version, ident, row["official_url"],
                          row["passage_text"], start, end, row["text_representation_sha256"],
                          row["language"], json.dumps({key: row[key] for key in PROVENANCE_FIELDS},
                                                     ensure_ascii=False, sort_keys=True))
        if not verify(passage, source):
            raise ValueError(f"Phase 2 passage grounding failure: {row['passage_id']}")
        passages.append(passage)
    if {passage.official_id for passage in passages} != eligible_ids:
        raise ValueError(f"Eligible passage parent coverage mismatch: {name}")
    return sources, passages, eligible_ids, excluded_rows


def build(root: Path, output: Path, *, max_parts: int | None = None,
          stop_after_parts: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(output)
    if "v4" not in output.parts:
        raise ValueError("Full release index output must be under V4 namespace")
    lock = _lock(root, max_parts=max_parts)
    fingerprint = lock["fingerprint"]
    staging = output.with_name(output.name + ".building")
    staging.parent.mkdir(parents=True, exist_ok=True)
    if staging.exists():
        checkpoint = _json(staging / "checkpoint.json")
        if checkpoint.get("fingerprint") != fingerprint or checkpoint.get("parts") != lock["parts"]:
            raise ValueError("Existing build checkpoint fingerprint/scope mismatch")
        if _json(staging / "lock.json") != lock["core"]:
            raise ValueError("Existing build implementation/input lock mismatch")
        store = DiskCorpusStore(staging / "corpus.sqlite", lock["core"]["corpus_version"], fingerprint)
        docs = DiskBM25(staging / "documents.sqlite", "document_bm25", fingerprint)
        chunks = DiskBM25(staging / "passages.sqlite", "passage_bm25", fingerprint)
    else:
        staging.mkdir()
        checkpoint = {"format": "medrag-v4-release-sparse-checkpoint-v1", "status": "building",
                      "fingerprint": fingerprint, "parts": lock["parts"], "next_part": 0,
                      "sources": 0, "passages": 0, "excluded_passages": 0}
        _atomic_json(staging / "lock.json", lock["core"])
        store = DiskCorpusStore(staging / "corpus.sqlite", lock["core"]["corpus_version"],
                                fingerprint, create=True)
        docs = DiskBM25(staging / "documents.sqlite", "document_bm25", fingerprint, create=True)
        chunks = DiskBM25(staging / "passages.sqlite", "passage_bm25", fingerprint, create=True)
        _atomic_json(staging / "checkpoint.json", checkpoint)
    official_urls = _official_urls(lock["paths"]["official"])
    for index in range(int(checkpoint["next_part"]), lock["parts"]):
        sources, passages, eligible_ids, excluded_rows = _read_part(lock, root, index, official_urls)
        # A process may stop after any of the three independent SQLite commits.
        # Roll back only the pending partition to the last durable checkpoint.
        meta = store._check(require_complete=False)
        recorded_counts = (int(meta["sources"]), int(meta["passages"]))
        actual_counts = (store.rollback_part(eligible_ids)
                         if recorded_counts != (checkpoint["sources"], checkpoint["passages"])
                         else recorded_counts)
        if actual_counts != (checkpoint["sources"], checkpoint["passages"]):
            raise ValueError(f"Store checkpoint count mismatch before part {index}")
        docs.truncate_to_rows(checkpoint["sources"])
        chunks.truncate_to_rows(checkpoint["passages"])
        store.append_sources(sources)
        docs.append_batch((s.official_id, None, f"{s.title} {s.text}") for s in sources)
        store.append_passages(passages)
        chunks.append_batch((p.official_id, p.passage_id, p.text) for p in passages)
        checkpoint = {**checkpoint, "next_part": index + 1,
                      "sources": checkpoint["sources"] + len(sources),
                      "passages": checkpoint["passages"] + len(passages),
                      "excluded_passages": checkpoint["excluded_passages"] + excluded_rows}
        _atomic_json(staging / "checkpoint.json", checkpoint)
        print(f"part {index + 1}/{lock['parts']}: {checkpoint['sources']} sources, "
              f"{checkpoint['passages']} passages", flush=True)
        if stop_after_parts is not None and index + 1 >= stop_after_parts:
            return {"status": "building", "next_part": index + 1,
                    "fingerprint": fingerprint, "checkpoint": str(staging / "checkpoint.json")}
    if lock["parts"] == 922 and (checkpoint["sources"] != 1_073_525
                                 or checkpoint["passages"] != 2_951_224
                                 or checkpoint["excluded_passages"] != 49):
        raise ValueError("Full release eligible counts/exclusions differ from lock")
    if store._check(require_complete=False)["status"] == "building":
        store.complete()
    else:
        if store.counts() != (checkpoint["sources"], checkpoint["passages"]):
            raise ValueError("Completed store count differs from checkpoint")
    if docs._check(require_complete=False)["status"] == "building":
        docs.complete()
    if chunks._check(require_complete=False)["status"] == "building":
        chunks.complete()
    if (int(docs._check(require_complete=True)["rows"]) != checkpoint["sources"]
            or int(chunks._check(require_complete=True)["rows"]) != checkpoint["passages"]):
        raise ValueError("Completed BM25 row counts differ from checkpoint")
    files = {name: sha256_file(staging / name) for name in
             ("corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json")}
    manifest = {"format": "medrag-v4-disk-index-v1", "status": "complete",
                "scope": "full_release" if lock["parts"] == 922 else "release_parts_pilot",
                "corpus_version": lock["core"]["corpus_version"], "fingerprint": fingerprint,
                "source_count": checkpoint["sources"], "passage_count": checkpoint["passages"],
                "excluded_passages": checkpoint["excluded_passages"],
                "parts": lock["parts"], "files": files, "dense": "not_built"}
    _atomic_json(staging / "manifest.json", manifest)
    os.rename(staging, output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-parts", type=int)
    parser.add_argument("--stop-after-parts", type=int,
                        help="Test checkpoint resume: leave the index incomplete after this part")
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.output.resolve(),
                           max_parts=args.max_parts,
                           stop_after_parts=args.stop_after_parts), indent=2))

"""Verify the published V4 sparse release without opening a GPU context."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time
from pathlib import Path


EXPECTED = {"parts": 922, "source_count": 1_073_525,
            "passage_count": 2_951_224, "excluded_passages": 49}
MODULUS = 1 << 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def multiset_digest(db: sqlite3.Connection, query: str) -> dict:
    count = total = xor = 0
    for row in db.execute(query):
        payload = json.dumps(row, separators=(",", ":"), ensure_ascii=False).encode()
        value = int.from_bytes(hashlib.sha256(payload).digest(), "big")
        count += 1
        total = (total + value) % MODULUS
        xor ^= value
    return {"count": count, "sum_sha256": f"{total:064x}", "xor_sha256": f"{xor:064x}"}


def inspect_db(path: Path, expected_meta: dict[str, str], count_queries: dict[str, str]) -> tuple[dict, dict]:
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as db:
        meta = dict(db.execute("SELECT key,value FROM meta"))
        for key, value in expected_meta.items():
            if meta.get(key) != value:
                raise ValueError(f"{path.name}: metadata {key}={meta.get(key)!r}, expected {value!r}")
        counts = {key: int(db.execute(query).fetchone()[0]) for key, query in count_queries.items()}
        quick_check = db.execute("PRAGMA quick_check(1)").fetchone()[0]
        if quick_check != "ok":
            raise ValueError(f"{path.name}: SQLite quick_check: {quick_check}")
    return meta, counts


def verify(root: Path, index: Path, reports: Path) -> dict:
    manifest_path = index / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key, value in {"status": "complete", "scope": "full_release", **EXPECTED}.items():
        if manifest.get(key) != value:
            raise ValueError(f"Manifest {key}={manifest.get(key)!r}, expected {value!r}")
    fingerprint = manifest["fingerprint"]
    if not isinstance(fingerprint, str) or len(fingerprint) != 64:
        raise ValueError("Manifest fingerprint is invalid")
    required_files = {"corpus.sqlite", "documents.sqlite", "passages.sqlite", "lock.json"}
    if set(manifest.get("files", {})) != required_files:
        raise ValueError("Manifest file list differs from sparse contract")

    release_audit_path = root / "outputs/v4/task4/release_coverage_audit_20261016.json"
    release_audit = json.loads(release_audit_path.read_text(encoding="utf-8"))
    if (release_audit.get("status") != "pass"
            or release_audit.get("eligible_source_count") != EXPECTED["source_count"]
            or release_audit.get("eligible_passage_count") != EXPECTED["passage_count"]
            or release_audit.get("excluded_passage_count") != EXPECTED["excluded_passages"]):
        raise ValueError("Independent release coverage audit is missing or differs")

    hashes = {}
    for name in sorted(required_files):
        actual = sha256_file(index / name)
        if actual != manifest["files"][name]:
            raise ValueError(f"Published file checksum mismatch: {name}")
        hashes[name] = actual
        print(f"hash PASS {name}", flush=True)

    corpus_path = index / "corpus.sqlite"
    document_path = index / "documents.sqlite"
    passage_path = index / "passages.sqlite"
    corpus_meta, corpus_counts = inspect_db(corpus_path,
        {"format": "medrag-v4-disk-store-v1", "status": "complete",
         "fingerprint": fingerprint, "corpus_version": manifest["corpus_version"],
         "sources": str(EXPECTED["source_count"]),
         "passages": str(EXPECTED["passage_count"])},
        {"sources": "SELECT count(*) FROM sources",
         "passages": "SELECT count(*) FROM passages",
         "passage_parents": "SELECT count(DISTINCT official_id) FROM passages",
         "orphan_passages": "SELECT count(*) FROM passages p LEFT JOIN sources s "
                            "ON p.official_id=s.official_id WHERE s.official_id IS NULL"})
    if corpus_counts != {"sources": EXPECTED["source_count"],
                          "passages": EXPECTED["passage_count"],
                          "passage_parents": EXPECTED["source_count"], "orphan_passages": 0}:
        raise ValueError(f"Corpus row/parent coverage mismatch: {corpus_counts}")
    print("corpus SQLite PASS", flush=True)

    bm25 = {}
    for name, path, channel, expected_rows in (
        ("documents", document_path, "document_bm25", EXPECTED["source_count"]),
        ("passages", passage_path, "passage_bm25", EXPECTED["passage_count"])):
        meta, counts = inspect_db(path,
            {"format": "v4-bm25-disk-v1", "status": "complete", "channel": channel,
             "fingerprint": fingerprint, "rows": str(expected_rows)},
            {"rows": "SELECT count(*) FROM docs",
             "total_length": "SELECT coalesce(sum(length),0) FROM docs",
             "min_row": "SELECT min(row_id) FROM docs",
             "max_row": "SELECT max(row_id) FROM docs",
             "postings": "SELECT count(*) FROM postings",
             "vocabulary": "SELECT count(*) FROM df"})
        if (counts["rows"] != expected_rows or counts["total_length"] != int(meta["total_length"])
                or counts["min_row"] != 0 or counts["max_row"] != expected_rows - 1
                or counts["postings"] <= 0 or counts["vocabulary"] <= 0):
            raise ValueError(f"{name} BM25 statistics mismatch: {counts}")
        bm25[name] = {"meta": meta, "counts": counts}
        print(f"{name} BM25 SQLite PASS", flush=True)

    identities = {}
    comparisons = (
        ("sources", corpus_path, "SELECT official_id FROM sources", document_path,
         "SELECT official_id FROM docs"),
        ("passages", corpus_path, "SELECT official_id,passage_id FROM passages", passage_path,
         "SELECT official_id,passage_id FROM docs"),
    )
    for label, left_path, left_sql, right_path, right_sql in comparisons:
        with sqlite3.connect(f"file:{left_path}?mode=ro", uri=True) as left:
            left_digest = multiset_digest(left, left_sql)
        with sqlite3.connect(f"file:{right_path}?mode=ro", uri=True) as right:
            right_digest = multiset_digest(right, right_sql)
        if left_digest != right_digest:
            raise ValueError(f"{label} corpus/BM25 identity multiset mismatch")
        identities[label] = left_digest
        print(f"{label} identity coverage PASS", flush=True)

    report = {"format": "medrag-v4-sparse-integrity-v1", "status": "SPARSE_FULL_PASS",
              "manifest_sha256": sha256_file(manifest_path), "index": str(index),
              "fingerprint": fingerprint, "files_sha256": hashes,
              "corpus_meta": corpus_meta, "corpus_counts": corpus_counts,
              "bm25": bm25, "identity_multisets": identities,
              "release_audit_sha256": sha256_file(release_audit_path)}
    write_json(reports / "sparse_integrity.json", report)
    write_json(reports / "sparse_coverage.json",
               {"status": "SPARSE_FULL_PASS", "index_manifest_sha256": report["manifest_sha256"],
                "release_audit_sha256": report["release_audit_sha256"],
                "corpus_counts": corpus_counts, "identity_multisets": identities})
    return report


def wait_for_publish(manifest_path: Path, worker_pid: int, interval: int) -> None:
    while not manifest_path.is_file():
        try:
            os.kill(worker_pid, 0)
        except ProcessLookupError as exc:
            raise RuntimeError("Sparse worker exited without publishing manifest") from exc
        time.sleep(interval)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--interval", type=int, default=30)
    args = parser.parse_args()
    if args.interval < 1:
        parser.error("--interval must be positive")
    root, index, reports = args.root.resolve(), args.index.resolve(), args.reports.resolve()
    try:
        if args.wait_pid is not None:
            wait_for_publish(index / "manifest.json", args.wait_pid, args.interval)
        result = verify(root, index, reports)
    except Exception as exc:
        write_json(reports / "sparse_integrity.json",
                   {"status": "SPARSE_FULL_FAIL", "index": str(index),
                    "error": f"{type(exc).__name__}: {exc}"})
        print(f"SPARSE_FULL_FAIL: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return 1
    print(json.dumps({"status": result["status"], "index": str(index)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify the Phase-0 backup can restore the pinned official crawl snapshot."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / "outputs/vibiomir_phase0_20261014"
BACKUP = BASE / "backup"
OFFICIAL = BASE / "backup/official_snapshot.tar"
AUDIT = BASE / "audit/document_state.parquet"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def relpath(value: str) -> str:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"Unsafe raw path: {value}")
    return path.relative_to(ROOT).as_posix() if path.is_absolute() else path.as_posix()


def tar_members(path: Path, expected: set[str]) -> tuple[int, set[str]]:
    process = subprocess.Popen(["tar", "-tf", str(path)], stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, text=True)
    assert process.stdout is not None
    count = 0
    for line in process.stdout:
        member = line.rstrip("\n/")
        expected.discard(member)
        count += 1
    stderr = process.stderr.read() if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"Cannot list archive members: {stderr[-1000:]}")
    return count, expected


def extract_hash(archive: Path, member: str) -> str:
    process = subprocess.Popen(["tar", "--occurrence=1", "-xOf", str(archive), member],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert process.stdout is not None
    sha = hashlib.sha256()
    for block in iter(lambda: process.stdout.read(8 * 1024 * 1024), b""):
        sha.update(block)
    stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
    if process.wait() != 0:
        raise RuntimeError(f"Cannot restore {member}: {stderr[-1000:]}")
    return sha.hexdigest()


def run() -> dict:
    progress_path = BACKUP / "backup_progress.json"
    if not progress_path.is_file() or not OFFICIAL.is_file() or not AUDIT.is_file():
        raise FileNotFoundError("Phase-0 audit and both archives must exist")
    progress = json.loads(progress_path.read_text())
    if progress["status"] != "complete" or not all(
            item["complete"] for item in progress["archives"].values()):
        raise RuntimeError("Raw backup archives are not complete")
    expected = set()
    raw_page_example = None
    for batch in pq.ParquetFile(AUDIT).iter_batches(batch_size=50_000, columns=["raw_path"]):
        for value in batch.column(0).to_pylist():
            if not value:
                continue
            relative = relpath(value)
            expected.add(relative)
            if relative.startswith("data/vibiomir/crawl/pages/"):
                if raw_page_example is None or relative < raw_page_example:
                    raw_page_example = relative
    raw_count = len(expected)
    checkpoints = {
        "data/raw/vibiomir/links_corpus.parquet",
        "data/raw/vibiomir/query.parquet",
        "data/vibiomir/crawl/dead_hosts.json",
        "outputs/vibiomir_step1_20261013/manifest.json",
        "outputs/vibiomir_step2_20261013/ledger_manifest.json",
        "outputs/vibiomir_step2_20261013/release/document_status.parquet",
        "outputs/vibiomir_step2_20261013/release/release_manifest.json",
        "outputs/crawl_failure_recovery/RECOVERY_RESULTS.parquet",
    }
    expected.update(checkpoints)
    counts = {}
    archive_records = {}
    for name, item in progress["archives"].items():
        archive = BACKUP / f"{name}.tar"
        if archive.stat().st_size != item["bytes"]:
            raise RuntimeError(f"Backup archive changed: {archive}")
        count, expected = tar_members(archive, expected)
        counts[name] = count
        archive_records[archive.name] = {"bytes": archive.stat().st_size,
                                         "sha256": digest(archive)}
    if expected:
        raise RuntimeError(f"Backup misses {len(expected)} required raw/checkpoint files: {sorted(expected)[:12]}")
    official_members = {
        "outputs/vibiomir_official_verify_20261012/links_corpus_vi_en_zh.parquet",
        "outputs/vibiomir_official_verify_20261012/query.parquet",
        "outputs/vibiomir_official_verify_20261012/verification.json",
    }
    official_count, missing = tar_members(OFFICIAL, official_members)
    if missing:
        raise RuntimeError(f"Official backup misses: {sorted(missing)}")
    restored = {}
    for archive, member in (
        (BACKUP / f"raw_pages_{int(raw_page_example.split('/')[4]) // 50 * 50:03d}_"
                  f"{int(raw_page_example.split('/')[4]) // 50 * 50 + 49:03d}.tar",
         raw_page_example),
        (OFFICIAL, "outputs/vibiomir_official_verify_20261012/query.parquet"),
        (OFFICIAL, "outputs/vibiomir_official_verify_20261012/links_corpus_vi_en_zh.parquet"),
    ):
        if member is None:
            raise RuntimeError("No official raw page was found in the audit")
        got = extract_hash(archive, member)
        source = ROOT / member
        if got != digest(source):
            raise RuntimeError(f"Restored bytes differ from source: {member}")
        restored[member] = got
    report = {
        "status": "complete", "verified_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
        "scope": "All raw paths referenced by the Phase-0 official audit, crawl manifests, and step-1/step-2/recovery state; out-of-scope legacy raw pages excluded",
        "required_raw_paths_present": raw_count, "raw_and_state_members": sum(counts.values()),
        "members_by_archive": counts,
        "official_snapshot_members": official_count,
        "archives": {**archive_records,
                     "official_snapshot.tar": {"bytes": OFFICIAL.stat().st_size,
                                               "sha256": digest(OFFICIAL)}},
        "restored_file_sha256": restored,
    }
    (BASE / "backup/backup_manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({key: result[key] for key in ("status", "required_raw_paths_present",
          "raw_and_state_members")}, indent=2))

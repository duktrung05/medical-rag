"""Read-only progress monitor for a running sparse release builder.

This process never opens a SQLite database or changes the builder/checkpoint.
It only samples procfs and file metadata, then appends a small JSONL report.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def read_worker(pid: int) -> dict | None:
    proc = Path("/proc") / str(pid)
    try:
        command = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        if "build_release_sparse.py" not in command:
            raise RuntimeError(f"PID {pid} is not the sparse builder: {command!r}")
        io = {}
        for line in (proc / "io").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"read_bytes", "write_bytes"}:
                io[key] = int(value.strip())
        stat = (proc / "stat").read_text()
        fields = stat[stat.rfind(")") + 2:].split()
        return {"pid": pid, "state": fields[0], "cpu_ticks": int(fields[11]) + int(fields[12]),
                "read_bytes": io["read_bytes"], "write_bytes": io["write_bytes"]}
    except FileNotFoundError:
        return None


def sample(pid: int, staging: Path, published: Path) -> dict:
    worker = read_worker(pid)
    checkpoint_path = staging / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text()) if checkpoint_path.exists() else None
    journal_path = staging / "documents.sqlite-journal"
    journal = journal_path.stat() if journal_path.exists() else None
    manifest_path = published / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    return {"timestamp": datetime.now(timezone.utc).isoformat(), "worker": worker,
            "checkpoint_next_part": checkpoint.get("next_part") if checkpoint else None,
            "checkpoint_parts": checkpoint.get("parts") if checkpoint else None,
            "journal_bytes": journal.st_size if journal else None,
            "journal_mtime_ns": journal.st_mtime_ns if journal else None,
            "published_status": manifest.get("status") if manifest else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--published", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.pid <= 0 or args.interval_seconds <= 0 or args.stall_seconds <= 0:
        parser.error("PID, interval and stall threshold must be positive")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    previous = None
    last_progress = time.monotonic()
    while True:
        record = sample(args.pid, args.staging, args.published)
        worker = record["worker"]
        progress = (record["checkpoint_next_part"], record["journal_bytes"],
                    record["journal_mtime_ns"], record["published_status"],
                    worker["cpu_ticks"] if worker else None,
                    worker["read_bytes"] if worker else None,
                    worker["write_bytes"] if worker else None)
        now = time.monotonic()
        if progress != previous:
            last_progress = now
        previous = progress
        record["seconds_without_observed_progress"] = round(now - last_progress)
        if worker is None:
            record["status"] = ("published" if record["published_status"] == "complete"
                                else "worker_exited_check_logs")
        elif now - last_progress >= args.stall_seconds:
            record["status"] = "suspect_stall_inspect_only"
        else:
            record["status"] = "running"
        with args.report.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(record, sort_keys=True), flush=True)
        if args.once or worker is None:
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()

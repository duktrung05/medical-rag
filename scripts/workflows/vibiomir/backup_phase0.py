"""Create resumable tar archives for official ViBioMIR raw and crawl state."""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
BACKUP = ROOT / "outputs/vibiomir_phase0_20261014/backup"
SOURCE_LIST = BACKUP / "backup_filelist.nul"


def groups() -> dict[str, list[str]]:
    result = defaultdict(list)
    for item in SOURCE_LIST.read_bytes().split(b"\0"):
        if not item:
            continue
        path = item.decode()
        if path.startswith("data/vibiomir/crawl/pages/"):
            shard = int(path.split("/")[4])
            start = shard // 50 * 50
            key = f"raw_pages_{start:03d}_{start+49:03d}"
        elif path.startswith("data/vibiomir/pubmed/"):
            key = "pubmed_raw"
        else:
            key = "state_and_checkpoints"
        result[key].append(path)
    return dict(sorted(result.items()))


def archive_one(key: str, paths: list[str]) -> int:
    target = BACKUP / f"{key}.tar"
    temporary = BACKUP / f"{key}.tar.partial"
    temporary.unlink(missing_ok=True)
    filelist = BACKUP / f"{key}.nul"
    with filelist.open("wb") as stream:
        for path in paths:
            stream.write(path.encode() + b"\0")
    with temporary.open("wb") as output:
        subprocess.run(["tar", "-C", str(ROOT), "--null", "-T", str(filelist), "-cf", "-"],
                       stdout=output, check=True)
    os.replace(temporary, target)
    return target.stat().st_size


def run() -> dict:
    BACKUP.mkdir(parents=True, exist_ok=True)
    plan = groups()
    report = {"status": "in_progress", "updated_at": None,
              "scope": "All raw paths referenced by the official Phase-0 audit, plus crawl manifests and step-1/step-2/recovery checkpoints; out-of-scope legacy pages excluded",
              "archives": {}}
    progress = BACKUP / "backup_progress.json"
    if progress.exists():
        previous = json.loads(progress.read_text())
        if set(previous["archives"]) != set(plan):
            raise ValueError("Existing backup plan has different archive groups")
        report = previous
    else:
        report["archives"] = {key: {"listed_paths": len(paths), "complete": False}
                              for key, paths in plan.items()}
    pending = {}
    for key, paths in plan.items():
        target = BACKUP / f"{key}.tar"
        if report["archives"][key]["complete"]:
            if not target.is_file() or target.stat().st_size != report["archives"][key]["bytes"]:
                raise RuntimeError(f"Completed archive missing or changed: {target}")
            continue
        pending[key] = paths
    # Three independent archives keep the run resumable while using the disk
    # array more effectively than one small-file tar process.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(archive_one, key, paths): key for key, paths in pending.items()}
        for future in as_completed(futures):
            key = futures[future]
            size = future.result()
            report["archives"][key].update(complete=True, bytes=size)
            report["updated_at"] = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
            progress.write_text(json.dumps(report, indent=2) + "\n")
            print(f"Completed {key}: {size} bytes", flush=True)
    report["status"] = "complete"
    report["updated_at"] = datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat()
    progress.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"], "archives": len(result["archives"])}))

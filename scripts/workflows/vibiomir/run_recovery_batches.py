"""Resume bounded official-URL recovery batches, stopping on poor success."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAN = ROOT / "outputs/vibiomir_step2_20261013/recovery_longchau_batches/plan_manifest.json"


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def run(plan_path: Path, *, concurrency: int, delay_seconds: float,
        min_success_rate: float, max_batches: int | None,
        batch_timeout_seconds: int = 1800) -> dict:
    plan = json.loads(plan_path.read_text())
    if plan["status"] != "prepared" or not plan["corpus_revision"]:
        raise ValueError("Invalid recovery batch plan")
    base = plan_path.parent
    result_path = base / "runner_progress.json"
    completed = 0
    statuses = Counter()
    stopped = None
    for part in plan["batch_parts"]:
        if max_batches is not None and completed >= max_batches:
            stopped = "max_batches"
            break
        folder = base / f"batch_{part['batch']:05d}"
        output = folder / "crawl"
        summary_path = output / "crawl_summary.json"
        cmd = [sys.executable, "-m", "scripts.workflows.vibiomir.crawl_vibiomir_batch",
               "--input", str(folder / "input.parquet"), "--output-dir", str(output),
               "--limit", str(part["rows"]), "--concurrency", str(concurrency),
               "--delay-seconds", str(delay_seconds), "--timeout-seconds", "20",
               "--max-consecutive-blocked", "5"]
        try:
            with (folder / "run.log").open("a", encoding="utf-8") as log:
                code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                      check=False, timeout=batch_timeout_seconds).returncode
        except subprocess.TimeoutExpired:
            stopped = f"batch_{part['batch']:05d}_timeout_after_{batch_timeout_seconds}s"
            break
        if code != 0 or not summary_path.exists():
            stopped = f"batch_{part['batch']:05d}_exit_{code}"
            break
        summary = json.loads(summary_path.read_text())
        if summary["processed_urls"] != part["rows"]:
            stopped = f"batch_{part['batch']:05d}_incomplete"
            break
        statuses.update(summary["status_counts"])
        completed += 1
        print(f"batch {part['batch']:05d}: {summary['success_count']}/{part['rows']} success", flush=True)
        atomic_json(result_path, {"updated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
                                  "completed_batches": completed,
                                  "processed_urls": sum(statuses.values()),
                                  "status_counts": dict(statuses), "stopped_reason": None})
        if summary["success_count"] / part["rows"] < min_success_rate:
            stopped = f"batch_{part['batch']:05d}_low_success"
            break
    status = "complete" if completed == plan["batches"] else "stopped"
    result = {"status": status, "updated_at": datetime.now(ZoneInfo("Asia/Ho_Chi_Minh")).isoformat(),
              "corpus_revision": plan["corpus_revision"], "completed_batches": completed,
              "planned_batches": plan["batches"], "processed_urls": sum(statuses.values()),
              "planned_urls": plan["rows"], "status_counts": dict(statuses),
              "stopped_reason": stopped, "min_success_rate": min_success_rate,
              "concurrency": concurrency, "delay_seconds": delay_seconds,
              "batch_timeout_seconds": batch_timeout_seconds}
    atomic_json(result_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--delay-seconds", type=float, default=1.0)
    parser.add_argument("--min-success-rate", type=float, default=0.8)
    parser.add_argument("--max-batches", type=int)
    parser.add_argument("--batch-timeout-seconds", type=int, default=1800)
    args = parser.parse_args()
    if (not 1 <= args.concurrency <= 4 or args.delay_seconds < 0.5
            or args.batch_timeout_seconds < 300):
        raise ValueError("Recovery runner requires 1..4 workers, at least 0.5s per-domain delay, and at least 300s batch timeout")
    print(json.dumps(run(args.plan.resolve(), concurrency=args.concurrency,
                         delay_seconds=args.delay_seconds,
                         min_success_rate=args.min_success_rate,
                         max_batches=args.max_batches,
                         batch_timeout_seconds=args.batch_timeout_seconds), indent=2))


if __name__ == "__main__":
    main()

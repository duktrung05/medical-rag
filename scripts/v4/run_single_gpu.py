"""Launch a Task 4 script on its one pinned GPU, refusing an occupied card."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


GPU_UUID = "GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7"
ROOT = Path(__file__).resolve().parents[2]
WAIT_EXIT = 75


def validate_config(config: dict) -> None:
    for stage in ("dense", "reranker"):
        settings = config.get(stage, {})
        if settings.get("enabled") is not True or settings.get("device") != "cuda:0":
            raise ValueError(f"{stage} must be enabled on the single visible cuda:0")


def query_state() -> tuple[dict, list[dict]]:
    def query(fields: str, kind: str) -> list[list[str]]:
        result = subprocess.run(
            ["nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True, timeout=15)
        return [[cell.strip() for cell in row] for row in csv.reader(result.stdout.splitlines()) if row]

    rows = query("index,uuid,name,memory.total,memory.used,utilization.gpu", "gpu")
    selected = [row for row in rows if row[1] == GPU_UUID]
    if len(selected) != 1:
        raise RuntimeError("The pinned Task 4 GPU UUID is absent or ambiguous")
    index, uuid, name, total, used, util = selected[0]
    gpu = {"index": int(index), "uuid": uuid, "name": name,
           "memory_total_mib": int(total), "memory_used_mib": int(used),
           "utilization_percent": int(util)}
    processes = [{"pid": int(row[0]), "gpu_uuid": row[1], "process_name": row[2]}
                 for row in query("pid,gpu_uuid,process_name", "compute-apps")
                 if row[1] == GPU_UUID]
    return gpu, processes


def busy_reasons(gpu: dict, processes: list[dict]) -> list[str]:
    reasons = []
    if processes:
        reasons.append("selected GPU has existing compute processes")
    if gpu["memory_used_mib"] > 256:
        reasons.append("selected GPU already uses more than 256 MiB")
    if gpu["utilization_percent"] > 0:
        reasons.append("selected GPU is executing work")
    return reasons


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.incomplete-{os.getpid()}")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/v4_phase2_full_a0_gpu3.yaml")
    parser.add_argument("--report", type=Path, default=ROOT /
                        "outputs/v4/task4/completion_20261019/reports/gpu_preflight.json")
    parser.add_argument("--preflight-only", action="store_true",
                        help="Check GPU occupancy without importing torch or allocating CUDA")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- scripts/v4/<script>.py [arguments]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    report = {"timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "gpu_uuid": GPU_UUID, "pid": os.getpid(),
              "preflight_only": args.preflight_only, "status": "checking"}
    lock_path = ROOT / "outputs/v4/task4/locks" / f"{GPU_UUID}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = lock_path.open("a+")
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            report["status"] = "WAIT_TASK4_GPU_LOCK"
            write_report(args.report, report)
            print(json.dumps(report))
            return WAIT_EXIT
        # Configuration parsing is independent of model/CUDA initialization.
        sys.path.insert(0, str(ROOT))
        from src.medrag.core.config import load_config

        validate_config(load_config(args.config))
        gpu, processes = query_state()
        reasons = busy_reasons(gpu, processes)
        report.update(gpu=gpu, existing_processes=processes, reasons=reasons)
        if reasons:
            report["status"] = "WAIT_GPU_BUSY"
            write_report(args.report, report)
            print(json.dumps(report))
            return WAIT_EXIT
        if args.preflight_only:
            report["status"] = "GPU_IDLE_CUDA_CHECK_PENDING"
            write_report(args.report, report)
            print(json.dumps(report))
            return 0
        if not command:
            raise ValueError("A Task 4 Python script is required")
        script = (ROOT / command[0]).resolve()
        if script.parent != ROOT / "scripts/v4" or not script.is_file() or script.suffix != ".py":
            raise ValueError("Only an existing scripts/v4 Python entrypoint is accepted")
        if script == Path(__file__).resolve():
            raise ValueError("Recursive launcher invocation is not allowed")
        # Refuse a child config that could override the validated device contract.
        for index, argument in enumerate(command[1:], 1):
            if argument == "--config":
                validate_config(load_config(ROOT / command[index + 1]))
            elif argument.startswith("--config="):
                validate_config(load_config(ROOT / argument.split("=", 1)[1]))
            elif argument in ("--device", "--device=cuda:0") or argument.startswith("--device="):
                value = command[index + 1] if argument == "--device" else argument.split("=", 1)[1]
                if value != "cuda:0":
                    raise ValueError("A child device override must be cuda:0")
        os.environ.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=GPU_UUID,
                          HF_HOME=str(ROOT / ".cache/huggingface"), HF_HUB_OFFLINE="1",
                          OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
                          PYTHONPATH=str(ROOT))
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("The worker must see exactly one usable CUDA device")
        props = torch.cuda.get_device_properties(0)
        actual_uuid = str(getattr(props, "uuid", ""))
        normalize = lambda value: value.removeprefix("GPU-").replace("-", "").lower()
        if normalize(actual_uuid) != normalize(GPU_UUID):
            raise RuntimeError(f"CUDA device UUID differs from pinned GPU: {actual_uuid}")
        allowed = set(os.sched_getaffinity(0))
        if not {4, 5, 6, 7}.issubset(allowed):
            raise RuntimeError("Task 4 GPU worker requires its planned CPU affinity 4-7")
        os.sched_setaffinity(0, {4, 5, 6, 7})
        os.nice(10)
        subprocess.run(["ionice", "-c", "3", "-p", str(os.getpid())], check=True, timeout=5)
        # Model imports can take minutes on this shared host. Recheck occupancy
        # after CUDA validation so a job arriving during import is noticed.
        latest_gpu, latest_processes = query_state()
        foreign_processes = [process for process in latest_processes
                             if process["pid"] != os.getpid()]
        latest_reasons = busy_reasons(latest_gpu, foreign_processes)
        if latest_processes and not foreign_processes:
            # The current process may now own a CUDA context; its VRAM is not
            # evidence of a competing job.
            latest_reasons = []
        report.update(gpu_after_cuda_check=latest_gpu,
                      processes_after_cuda_check=latest_processes)
        if latest_reasons:
            report.update(status="WAIT_GPU_BUSY", reasons=latest_reasons)
            write_report(args.report, report)
            print(json.dumps(report), flush=True)
            return WAIT_EXIT
        report.update(status="LAUNCHING", cuda_visible_devices=GPU_UUID,
                      torch_device_count=1, torch_device_uuid=actual_uuid,
                      cpu_affinity=sorted(os.sched_getaffinity(0)), command=command,
                      initial_free_vram_mib=gpu["memory_total_mib"] - gpu["memory_used_mib"])
        write_report(args.report, report)
        print(json.dumps(report), flush=True)
        # exec keeps the PID and the advisory lock throughout the child worker.
        os.set_inheritable(lock.fileno(), True)
        os.chdir(ROOT)
        os.execve(sys.executable, [sys.executable, str(script), *command[1:]], os.environ)
    except Exception as exc:
        report.update(status="PREFLIGHT_FAILED", error=f"{type(exc).__name__}: {exc}")
        write_report(args.report, report)
        print(json.dumps(report), file=sys.stderr)
        return 1
    finally:
        lock.close()


if __name__ == "__main__":
    raise SystemExit(main())

"""Plan, archive, verify and restore obsolete local research artifacts.

Nothing is deleted. Entries are renamed to an external, same-filesystem archive
with a file inventory and a recovery command. Runtime/source/config/test trees,
raw data, models and champion lineage remain at their original paths.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE_PARENT = ROOT.parent / "project-archives"
CONTROL = ROOT / "docs/workspace_cleanup_20261025"

KEEP_OUTPUTS = {
    "v4", "vibiomir_official_verify_20261012", "scored_feedback_recheck_20261009",
    "submissions", "vibiomir_phase0_20261014", "vibiomir_phase1_full_20261014_v1",
    "vibiomir_phase2_full_v2", "vibiomir_step1_20261013", "vibiomir_step2_20261013",
    "vibiomir_step2_runtime_deps_20261013", "pubmed_runtime_deps_20261012",
}
KEEP_V4 = {
    "task4", "a0_fast_a6000_final_20261023", "a0_remaining_passages_20261024",
    "a0_refined_candidates_20261024", "a0_quality_submissions_20261024",
    "a0_quality_diagnosis_20261024", "a0_quality_comparison_20261024",
    "a0_t4_v1_full_20261022", "a0_fast_parent_first_v1_retry_20261023",
    "a0_fast_parent_first_v1_20261023", "p5_gpu_incremental_a6000_20261023",
    "p5_fast_a0_migration_retry_20261023", "p5_fast_a0_migration_execution_20261023",
    "p5_fast_a0_production_decision_20261023", "p5_parent_fallback_v2_production_newplan_20261023",
    "p5_a0_completion_20261023", "p5_2_split_pilot_20261022",
    "p5_2_split_pilot_live_20261022", "p5_3a_colab_import", "p5_3a_colab_canary_20261022",
    "parent_first_evidence_pilot_20261021_v2", "adjacent_context_replay_d200_20261021",
    "p0_positive_path_replay_20261021_v3", "r3_chunk_reranker_20261024",
    "r5_s2_context_candidates_20261024", "r5_s2_same_input_original_20261024",
    "r5_same_input_evaluation_20261024", "r5_window_scoring_evaluation_20261024",
    "r5_context_selection_20261024", "r6_stage_oracle_20261025",
    "r8_evidence_extraction_20261025", "project_review_20261025",
}
KEEP_TASK4 = {
    "full_index_gpu3_20261019", "sparse_full_20261016", "dense_full_gpu3_20261019",
    "full_index_acceptance_20261021", "completion_20261019", "completion_20261020", "locks",
}
KEEP_DOCS = {
    "repository_hygiene", "workspace_cleanup_20261025", "V4_WORKSPACE_GUIDE.md",
    "v4_project_review_20261025_vi.md", "v4_a0_quality_fix_20261024_vi.md",
    "v4_s1_s2_document_selection_20261024_vi.md", "v4_r3_r4_execution_20261024_vi.md",
    "v4_r5_execution_20261024_vi.md", "v4_p5_2_split_runner_20261022_vi.md",
    "v4_task4_full_index_execution_result_20261021_vi.md", "refactor_v4_v5",
}


def dump(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*args: str) -> str:
    return subprocess.check_output(args, cwd=ROOT, text=True)


def members(path: Path):
    yield path
    if path.is_dir() and not path.is_symlink():
        for directory, dirs, files in os.walk(path, followlinks=False):
            base = Path(directory)
            for name in list(dirs):
                child = base / name
                if child.is_symlink():
                    dirs.remove(name)
                    yield child
            for name in files:
                yield base / name


def metadata(path: Path) -> dict:
    info = path.lstat()
    result = {"path": str(path.relative_to(ROOT)), "bytes": info.st_size,
              "inode": info.st_ino, "device": info.st_dev, "mtime_ns": info.st_mtime_ns,
              "kind": "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file"}
    if path.is_symlink():
        result["link_target"] = os.readlink(path)
    elif path.is_file() and info.st_size <= 2_000_000:
        result["sha256"] = sha(path)
    return result


def candidate_paths() -> dict[str, str]:
    selected = {}
    def add(path, reason):
        if path.exists() or path.is_symlink():
            selected[str(path.relative_to(ROOT))] = reason
    for path in (ROOT / "outputs").iterdir():
        if path.name not in KEEP_OUTPUTS:
            add(path, "historical_non_v4_experiment_or_stale_job_output")
    for path in (ROOT / "outputs/v4").iterdir():
        if path.name not in KEEP_V4:
            add(path, "aborted_run" if "aborted" in path.name else
                "superseded_pilot_or_rejected_experiment")
    for path in (ROOT / "outputs/v4/task4").iterdir():
        if path.is_dir() and path.name not in KEEP_TASK4:
            add(path, "superseded_index_pilot_or_recovery_snapshot")
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix in {".md", ".csv", ".json"} \
                and path.name not in {"README.md", "AGENTS.md"}:
            add(path, "historical_research_report_from_repository_root")
    for path in (ROOT / "docs").iterdir():
        if path.name not in KEEP_DOCS:
            add(path, "superseded_plan_or_non_v4_document")
    for path in (ROOT / "docs/refactor_v4_v5").iterdir():
        if path.name != "PROTECTED_DATA_MANIFEST.json":
            add(path, "historical_refactor_plan_replaced_by_current_workspace_guide")
    for path in (ROOT / "outputs/submissions").iterdir():
        if path.name != "vibiomir_threshold_neg2_20261008":
            add(path, "historical_non_champion_submission")
    for path in (ROOT / "outputs/v4/a0_quality_submissions_20261024").iterdir():
        if path.name not in {"s2_docs_adaptive", "rerank_refined"}:
            add(path, "s1_superseded_by_s2")
    return selected


def incoming_symlinks() -> list[dict]:
    result = []
    for base in (ROOT / "outputs", ROOT / "data", ROOT / "configs"):
        for directory, dirs, files in os.walk(base, followlinks=False):
            parent = Path(directory)
            for name in dirs + files:
                path = parent / name
                if path.is_symlink():
                    result.append({"path": str(path.relative_to(ROOT)),
                                   "resolved": str(path.resolve()),
                                   "target": os.readlink(path)})
    return result


def core_dependencies() -> list[dict]:
    """Explicit file references of current champion/audit code and pinned metadata."""
    files = [
        ROOT / "configs/vibiomir/phase2_batch_migration_v1.json",
        ROOT / "configs/vibiomir/phase2_semantic_staging_20261015.json",
        ROOT / "configs/v4_submission_p2_a0.yaml",
        ROOT / "outputs/v4/a0_fast_a6000_final_20261023/lock.json",
        ROOT / "outputs/v4/a0_fast_parent_first_v1_retry_20261023/lineage.json",
        ROOT / "outputs/v4/a0_fast_parent_first_v1_retry_20261023/lock.json",
        ROOT / "outputs/v4/a0_t4_v1_full_20261022/lock.json",
        ROOT / "outputs/v4/p5_gpu_incremental_a6000_20261023/runtime_lock.json",
        ROOT / "outputs/v4/a0_refined_candidates_20261024/lock.json",
    ]
    scripts = ["run_p5_2_split.py", "run_p5_fast_a0.py", "complete_p5_a0.py",
               "run_p5_incremental_gpu.py", "score_a0_remaining_passages.py",
               "refine_a0_submission.py", "diagnose_a0_quality.py",
               "audit_r6_stage_oracle.py", "run_r8_evidence_extraction.py"]
    files.extend(ROOT / "scripts/v4" / name for name in scripts)
    refs = {}
    pattern = re.compile(r"outputs/[a-zA-Z0-9_./-]+")
    for path in files:
        if not path.is_file():
            continue
        text = path.read_text()
        for match in pattern.findall(text):
            normalized = match.rstrip("/.")
            target = ROOT / normalized
            if target.exists():
                refs.setdefault(normalized, set()).add(str(path.relative_to(ROOT)))
    return [{"path": path, "referenced_by": sorted(sources)} for path, sources in sorted(refs.items())]


def is_within(path: Path, parent: Path) -> bool:
    return path == parent or parent in path.parents


def live_paths() -> list[dict]:
    result = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            comm = (entry / "comm").read_text().strip()
            cwd = (entry / "cwd").resolve(strict=True)
            files = []
            for fd in (entry / "fd").iterdir():
                try:
                    path = fd.resolve(strict=True)
                    if is_within(path, ROOT / "outputs"):
                        files.append(str(path))
                except (OSError, RuntimeError):
                    pass
            if files:
                result.append({"pid": int(entry.name), "command": comm,
                               "cwd": str(cwd), "open_output_paths": sorted(set(files))})
        except (OSError, RuntimeError):
            continue
    return result


def plan() -> dict:
    if (CONTROL / "plan.json").exists():
        raise FileExistsError(CONTROL)
    CONTROL.mkdir(parents=True, exist_ok=True)
    candidates = candidate_paths()
    deps = core_dependencies()
    links = incoming_symlinks()
    live = live_paths()
    blocked = []
    for relative in list(candidates):
        path = ROOT / relative
        reasons = []
        for dep in deps:
            if is_within(ROOT / dep["path"], path):
                reasons.append("referenced_by_core:" + dep["path"])
        for link in links:
            link_path = ROOT / link["path"]
            # A surviving symlink must never point into a moved artifact.
            source_will_move = any(is_within(link_path, ROOT / candidate) for candidate in candidates)
            if not source_will_move and is_within(Path(link["resolved"]), path):
                reasons.append("target_of_surviving_symlink:" + link["path"])
        for process in live:
            for opened in process["open_output_paths"]:
                if is_within(Path(opened), path):
                    reasons.append("open_by_process:" + str(process["pid"]))
        if reasons:
            blocked.append({"path": relative, "retained_because": reasons})
            del candidates[relative]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive = ARCHIVE_PARENT / f"v4_cleanup_{stamp}"
    entries = []
    for relative, reason in sorted(candidates.items()):
        inventory = [metadata(path) for path in members(ROOT / relative)]
        entries.append({"path": relative, "reason": reason,
                        "file_count": sum(row["kind"] == "file" for row in inventory),
                        "logical_bytes": sum(row["bytes"] for row in inventory if row["kind"] == "file"),
                        "inventory": inventory})
    protected = []
    for directory in ("src", "scripts", "tests", "configs", "resources", "notebooks"):
        for path in (ROOT / directory).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts:
                protected.append({"path": str(path.relative_to(ROOT)), "sha256": sha(path)})
    protected.append({"path": "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip",
                      "sha256": sha(ROOT / "outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip")})
    baseline_refs = ["outputs/v4/task4/full_index_gpu3_20261019/manifest.json",
                     "outputs/v4/a0_refined_candidates_20261024/run_manifest.json",
                     "outputs/v4/r6_stage_oracle_20261025/positive_registry.parquet"]
    for relative in baseline_refs:
        protected.append({"path": relative, "sha256": sha(ROOT / relative)})
    result = {"format": "v4-workspace-cleanup-v1", "root": str(ROOT), "archive": str(archive),
              "method": "same-filesystem rename; no deletion", "entries": entries,
              "retained_candidates": blocked, "protected_hashes": protected,
              "core_dependencies": deps, "symlink_inventory": links, "live_processes": live,
              "summary": {"move_entries": len(entries), "files": sum(e["file_count"] for e in entries),
                          "logical_bytes": sum(e["logical_bytes"] for e in entries),
                          "retained_candidates": len(blocked)}}
    dump(CONTROL / "plan.json", result)
    dump(CONTROL / "status_before.json", command("git", "status", "--porcelain=v1", "--untracked-files=all"))
    print(json.dumps({"archive": str(archive), **result["summary"], "retained": blocked}, ensure_ascii=False))
    return result


def verify(plan_data: dict) -> dict:
    failures = []
    for row in plan_data["protected_hashes"]:
        path = ROOT / row["path"]
        if not path.is_file() or sha(path) != row["sha256"]:
            failures.append("protected_content_changed:" + row["path"])
    archive = Path(plan_data["archive"])
    for entry in plan_data["entries"]:
        if (ROOT / entry["path"]).exists():
            failures.append("source_still_exists:" + entry["path"])
        for row in entry["inventory"]:
            path = archive / "tree" / row["path"]
            try:
                info = path.lstat()
            except FileNotFoundError:
                failures.append("archived_item_missing:" + row["path"])
                continue
            if row["kind"] == "file" and (info.st_size != row["bytes"] or info.st_ino != row["inode"]):
                failures.append("archived_identity_changed:" + row["path"])
            if row.get("sha256") and sha(path) != row["sha256"]:
                failures.append("archived_content_changed:" + row["path"])
    for dep in plan_data["core_dependencies"]:
        if not (ROOT / dep["path"]).exists():
            failures.append("core_dependency_missing:" + dep["path"])
    for link in plan_data["symlink_inventory"]:
        source = ROOT / link["path"]
        was_moved = any(is_within(source, ROOT / entry["path"]) for entry in plan_data["entries"])
        if not was_moved and source.is_symlink() and not source.exists():
            # Only flag links which pointed at existing targets before cleanup.
            if Path(link["resolved"]).exists() or any(
                    is_within(Path(link["resolved"]), ROOT / entry["path"]) for entry in plan_data["entries"]):
                failures.append("surviving_link_broken:" + link["path"])
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures,
              "protected_files": len(plan_data["protected_hashes"]),
              "archived_entries": len(plan_data["entries"]),
              "archived_files": plan_data["summary"]["files"]}
    dump(CONTROL / "verification.json", result)
    print(json.dumps(result))
    if failures:
        raise RuntimeError("Cleanup verification failed")
    return result


def apply() -> None:
    data = json.loads((CONTROL / "plan.json").read_text())
    archive = Path(data["archive"])
    if not is_within(archive, ARCHIVE_PARENT) or archive.exists():
        raise ValueError("Archive path is not a new child of project-archives")
    archive.mkdir(parents=True)
    if ROOT.stat().st_dev != archive.stat().st_dev:
        raise ValueError("Cleanup requires same-filesystem rename")
    # Preserve the entire dirty code/doc/config state, independently of Git.
    with tarfile.open(archive / "working_tree_code_before.tar.gz", "w:gz", dereference=False) as tar:
        for name in ("src", "scripts", "tests", "configs", "resources", "notebooks", "docs", "archive", "r2ai"):
            tar.add(ROOT / name, arcname=name,
                    filter=lambda info: None if "__pycache__" in info.name.split("/") else info)
        for path in ROOT.iterdir():
            if path.is_file():
                tar.add(path, arcname=path.name)
    (archive / "git_diff_before.patch").write_text(command("git", "diff", "--binary"))
    (archive / "git_staged_diff_before.patch").write_text(command("git", "diff", "--cached", "--binary"))
    dump(archive / "cleanup_manifest.json", data)
    journal = []
    for entry in data["entries"]:
        source = ROOT / entry["path"]
        target = archive / "tree" / entry["path"]
        if not is_within(source, ROOT) or source == ROOT or target.exists():
            raise ValueError("Unsafe or conflicting move")
        for process in live_paths():
            if any(is_within(Path(opened), source) for opened in process["open_output_paths"]):
                raise RuntimeError(f"Artifact became live before move: {source}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
        journal.append({"source": str(source), "destination": str(target), "reason": entry["reason"]})
        dump(archive / "move_journal.json", journal)
    dump(CONTROL / "archive_location.json", {"archive": str(archive),
         "restore_command": "python3 scripts/tooling/cleanup_v4_workspace.py restore",
         "summary": data["summary"]})
    verify(data)


def restore() -> None:
    data = json.loads((CONTROL / "plan.json").read_text())
    archive = Path(data["archive"])
    journal_path = archive / "move_journal.json"
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else []
    for entry in reversed(journal):
        source = Path(entry["destination"])
        target = Path(entry["source"])
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"Restore refuses to overwrite: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        source.rename(target)
    dump(CONTROL / "restored.json", {"restored_entries": len(journal)})
    print(json.dumps({"restored_entries": len(journal)}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "apply", "verify", "restore"))
    args = parser.parse_args()
    if args.command == "plan":
        plan()
    elif args.command == "apply":
        apply()
    elif args.command == "restore":
        restore()
    else:
        verify(json.loads((CONTROL / "plan.json").read_text()))


if __name__ == "__main__":
    main()

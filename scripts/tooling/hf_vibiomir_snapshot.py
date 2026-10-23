"""Upload and restore a private ViBioMIR snapshot without copying the data tree.

Crawl pages are packed into bounded tar shards. Other files are uploaded directly.
Only a successfully uploaded, verified temporary archive is removed. Source files
are never removed. The final manifest is published after every upload succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tarfile
import time
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STATE = ROOT / "outputs/hf_med_rag_20261007"
MANIFEST = "snapshot/manifest.json"


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def signature(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def temporary_source(path: str) -> bool:
    name = PurePosixPath(path).name
    return ".partial." in name or name.endswith((".tmp", ".lock"))


def make_plan(inventory: dict, target_bytes: int) -> dict:
    regular = [entry for entry in inventory["regular_files"]
               if entry["path"].startswith(("vibiomir/", "raw/vibiomir/"))
               and not temporary_source(entry["path"])]
    groups, current, size = [], [], 0
    for shard in sorted(inventory["page_shards"], key=lambda entry: entry["path"]):
        if not shard["path"].startswith("vibiomir/crawl/pages/"):
            raise ValueError(f"Unexpected page shard: {shard['path']}")
        if current and size + shard["tar_size"] > target_bytes:
            groups.append(current)
            current, size = [], 0
        current.append(shard)
        size += shard["tar_size"]
    if current:
        groups.append(current)
    return {"version": 1, "source": inventory["root"], "regular_files": regular,
            "page_groups": groups, "created_at": inventory["created_at"],
            "estimated_bytes": sum(entry["size"] for entry in regular)
            + sum(entry["tar_size"] for group in groups for entry in group)}


def pack_pages(source: Path, group: list[dict], archive: Path) -> dict:
    archive.parent.mkdir(parents=True, exist_ok=True)
    members, source_bytes = 0, 0
    temporary = archive.with_suffix(".tar.tmp")
    try:
        with tarfile.open(temporary, "w", format=tarfile.GNU_FORMAT) as output:
            for shard in group:
                count, size = 0, 0
                folder = source / shard["path"]
                for base, directories, names in os.walk(folder):
                    directories.sort()
                    for name in sorted(names):
                        path = Path(base) / name
                        if path.is_symlink() or not path.is_file():
                            raise ValueError(f"Expected a regular page file: {path}")
                        before = signature(path)
                        info = tarfile.TarInfo("data/" + path.relative_to(source).as_posix())
                        info.size, info.mtime = before[0], before[1] // 1_000_000_000
                        info.mode = 0o644
                        with path.open("rb") as stream:
                            output.addfile(info, stream)
                        if signature(path) != before:
                            raise RuntimeError(f"Page changed while packing: {path}")
                        count += 1
                        size += before[0]
                if (count, size) != (shard["files"], shard["size"]):
                    raise RuntimeError(f"Page inventory changed for {folder}; regenerate the inventory")
                members += count
                source_bytes += size
        temporary.replace(archive)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"size": archive.stat().st_size, "sha256": digest(archive),
            "files": members, "source_bytes": source_bytes}


def require_private(api, repo_id: str):
    info = api.repo_info(repo_id, repo_type="dataset", files_metadata=True)
    if not info.private:
        raise RuntimeError(f"Refusing to upload data: {repo_id} is not private")
    return info


def verify_remote(api, repo_id: str, records: list[dict]) -> None:
    by_path = {entry.path: entry for entry in api.get_paths_info(
        repo_id, [record["path"] for record in records], repo_type="dataset")}
    for record in records:
        remote = by_path.get(record["path"])
        if remote is None or remote.size != record["size"]:
            raise RuntimeError(f"Remote file missing or wrong size: {record['path']}")
        lfs = getattr(remote, "lfs", None)
        remote_hash = getattr(lfs, "sha256", None) if lfs else None
        if remote_hash and remote_hash != record["sha256"]:
            raise RuntimeError(f"Remote checksum mismatch: {record['path']}")


def wait_for_extract(pid_file: Path | None) -> None:
    if not pid_file or not pid_file.exists():
        return
    pid = int(pid_file.read_text().strip())
    process = Path("/proc") / str(pid)
    while process.exists():
        try:
            command = (process / "cmdline").read_bytes()
            state = (process / "stat").read_text().split(") ", 1)[1].split()[0]
        except FileNotFoundError:
            break
        if b"r2ai/extract.py" not in command or state == "Z":
            break
        log(f"Waiting for extract pid={pid}; completed files will upload after it exits")
        time.sleep(30)


def upload(args) -> None:
    from huggingface_hub import CommitOperationAdd, HfApi, get_token

    plan = json.loads(args.plan.read_text())
    source = Path(plan["source"]).resolve()
    state_file = args.state_dir / "state.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {
        "repo_id": args.repo_id, "source": str(source), "plan_sha256": digest(args.plan),
        "archives": {}, "files": {}}
    if (state["repo_id"], state["source"]) != (args.repo_id, str(source)):
        raise ValueError("Resume state belongs to another repository or source")
    if state.get("plan_sha256") != digest(args.plan):
        raise ValueError("Upload plan changed; do not reuse state for a different snapshot")
    while not get_token():
        if not args.wait_for_auth:
            raise RuntimeError("Hugging Face login is required; run hf auth login")
        log("Waiting for Hugging Face login; no data has been uploaded")
        time.sleep(30)
    api = HfApi()
    identity = api.whoami()
    log(f"Authenticated as {identity['name']}; target={args.repo_id}")
    repo_info = require_private(api, args.repo_id)
    # This small write validates permission before any multi-GB transfer.
    instructions = f"""# Restore this private ViBioMIR snapshot

Authenticate with a Hugging Face token allowed to read `{args.repo_id}`.
Install `huggingface_hub` in a Python 3.10+ environment, then run:

```bash
hf download {args.repo_id} snapshot/hf_vibiomir_snapshot.py --repo-type dataset --local-dir .
python snapshot/hf_vibiomir_snapshot.py restore --repo-id {args.repo_id} --destination /path/to/medical-rag
```

The command restores `data/vibiomir/` and `data/raw/vibiomir/` directly under
the destination. It downloads and unpacks one crawl archive at a time, verifies
SHA-256 checksums, removes that temporary archive, and preserves existing files
unless `--overwrite` is explicitly provided. Re-run to resume an interruption.

Wait until `snapshot/upload_status.json` says `complete: true`. The manifest
only appears once all planned files and archives have uploaded successfully.
This is a snapshot of files selected at upload preparation time. Files created
by later pipeline stages are not automatically added. Temporary `.partial`,
`.tmp`, and `.lock` files are excluded. An extract already in progress is allowed
to finish before its selected output is uploaded.
"""
    operations = [
        CommitOperationAdd(path_in_repo="snapshot/upload_status.json",
                           path_or_fileobj=json.dumps({"status": "uploading", "complete": False,
                              "scope": "ViBioMIR", "estimated_bytes": plan["estimated_bytes"]}).encode()),
        CommitOperationAdd(path_in_repo="snapshot/hf_vibiomir_snapshot.py", path_or_fileobj=Path(__file__)),
        CommitOperationAdd(path_in_repo="snapshot/RESTORE.md", path_or_fileobj=instructions.encode()),
    ]
    if not any(entry.rfilename == "README.md" for entry in repo_info.siblings or []):
        card = ("---\npretty_name: Medical RAG ViBioMIR snapshot\n---\n\n"
                "# ViBioMIR data snapshot\n\nPrivate project data for migration and reproducible "
                "retrieval experiments. Includes crawled pages, crawl manifests, source query/URL "
                "tables, completed extracted chunks, and existing indexes/embedding caches. "
                "Original upstream data: [ViBioMIR](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR).\n\n"
                "Crawl pages are stored in tar shards; other files retain their project-relative "
                "paths. See [restore instructions](snapshot/RESTORE.md) and "
                "[upload status](snapshot/upload_status.json).\n")
        operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=card.encode()))
    api.create_commit(repo_id=args.repo_id, repo_type="dataset", operations=operations,
                      commit_message="Begin private ViBioMIR data snapshot", num_threads=2)
    log(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")
    for index, group in enumerate(plan["page_groups"]):
        remote_path = f"archives/crawl-pages-{index:04d}.tar"
        if remote_path in state["archives"]:
            verify_remote(api, args.repo_id, [state["archives"][remote_path]])
            log(f"Resume: verified {remote_path}")
            continue
        archive = args.state_dir / "scratch" / Path(remote_path).name
        log(f"Packing {remote_path} ({index + 1}/{len(plan['page_groups'])})")
        record = {"path": remote_path, **pack_pages(source, group, archive),
                  "source_shards": [shard["path"] for shard in group]}
        require_private(api, args.repo_id)
        api.upload_file(repo_id=args.repo_id, repo_type="dataset", path_or_fileobj=archive,
                        path_in_repo=remote_path, commit_message=f"Upload {remote_path}")
        verify_remote(api, args.repo_id, [record])
        state["archives"][remote_path] = record
        write_json(state_file, state)
        archive.unlink()
        log(f"Uploaded and verified {remote_path}: {record['size'] / 1e9:.3f} GB")

    wait_for_extract(args.extract_pid_file)
    if args.extract_log_file:
        expected = f"wrote {source / args.extract_output}:"
        if expected not in args.extract_log_file.read_text(errors="replace"):
            raise RuntimeError("Extract did not log successful completion; its output will not be uploaded")
    remaining = []
    for entry in plan["regular_files"]:
        path = source / entry["path"]
        before = signature(path)
        remote_path = "data/" + entry["path"]
        previous = state["files"].get(remote_path)
        if previous and previous["source_signature"] == before:
            verify_remote(api, args.repo_id, [previous])
            continue
        if path.suffix == ".parquet":
            import pyarrow.parquet as pq
            pq.ParquetFile(path)  # Reject unfinished/truncated Parquet files.
        record = {"path": remote_path, "size": before[0], "sha256": digest(path),
                  "source_signature": before}
        if signature(path) != before:
            raise RuntimeError(f"File is still changing: {path}")
        remaining.append((path, record))
        if len(remaining) == 30:
            upload_batch(api, args.repo_id, remaining, state, state_file, CommitOperationAdd)
            remaining = []
    if remaining:
        upload_batch(api, args.repo_id, remaining, state, state_file, CommitOperationAdd)
    require_private(api, args.repo_id)
    manifest = {"version": 1, "complete": True, "repo_id": args.repo_id,
                "scope": ["data/vibiomir", "data/raw/vibiomir"],
                "temporary_files_excluded": True,
                "archives": list(state["archives"].values()),
                "files": list(state["files"].values())}
    write_json(args.state_dir / "manifest.json", manifest)
    result = api.upload_file(repo_id=args.repo_id, repo_type="dataset",
                    path_in_repo=MANIFEST, path_or_fileobj=args.state_dir / "manifest.json",
                    commit_message="Publish completed ViBioMIR snapshot manifest")
    completed = api.upload_file(repo_id=args.repo_id, repo_type="dataset",
                    path_in_repo="snapshot/upload_status.json",
                    path_or_fileobj=json.dumps({"status": "complete", "complete": True,
                                               "manifest_revision": result.oid}).encode(),
                    commit_message="Mark ViBioMIR snapshot complete")
    state["complete"] = True
    state["manifest_revision"] = result.oid
    state["snapshot_revision"] = completed.oid
    write_json(state_file, state)
    log(f"COMPLETE https://huggingface.co/datasets/{args.repo_id}; revision={completed.oid}")


def upload_batch(api, repo_id, batch, state, state_file, operation_class) -> None:
    require_private(api, repo_id)
    api.create_commit(repo_id=repo_id, repo_type="dataset",
                      operations=[operation_class(path_in_repo=record["path"], path_or_fileobj=path)
                                  for path, record in batch],
                      commit_message=f"Upload {len(batch)} ViBioMIR source and processed files",
                      num_threads=2)
    verify_remote(api, repo_id, [record for path, record in batch])
    for path, record in batch:
        if signature(path) != record["source_signature"]:
            raise RuntimeError(f"File changed during upload: {path}")
        state["files"][record["path"]] = record
    write_json(state_file, state)
    log(f"Uploaded {len(batch)} files; completed ordinary files={len(state['files'])}")


def safe_member_path(name: str, destination: Path) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name:
        raise ValueError(f"Unsafe archive member: {name}")
    if relative.parts[:4] != ("data", "vibiomir", "crawl", "pages"):
        raise ValueError(f"Unexpected archive member: {name}")
    target = destination.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise ValueError(f"Archive member escapes destination: {name}")
    return target


def unpack_pages(archive: Path, destination: Path, overwrite: bool = False) -> None:
    with tarfile.open(archive, "r:") as source:
        for member in source:
            if not member.isfile():
                raise ValueError(f"Unexpected non-file archive member: {member.name}")
            target = safe_member_path(member.name, destination)
            stream = source.extractfile(member)
            if target.exists() and not overwrite:
                identical = target.is_file() and target.stat().st_size == member.size
                if identical:
                    with stream, target.open("rb") as existing:
                        while block := stream.read(1024 * 1024):
                            if existing.read(len(block)) != block:
                                identical = False
                                break
                else:
                    stream.close()
                if identical:
                    continue
                raise FileExistsError(f"Restore target differs: {target}; use --overwrite explicitly")
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".hf-restore.tmp")
            with stream, temporary.open("wb") as output:
                shutil.copyfileobj(stream, output, length=1024 * 1024)
            temporary.replace(target)


def restore(args) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.repo_info(args.repo_id, repo_type="dataset", revision=args.revision)
    revision = info.sha  # Pin every download to the same commit.
    scratch = args.destination / ".hf-restore"
    status_path = api.hf_hub_download(args.repo_id, "snapshot/upload_status.json", repo_type="dataset",
                                    revision=revision, local_dir=scratch)
    if not json.loads(Path(status_path).read_text()).get("complete"):
        raise ValueError("Snapshot upload has not completed yet")
    manifest_path = api.hf_hub_download(args.repo_id, MANIFEST, repo_type="dataset",
                                      revision=revision, local_dir=scratch)
    manifest = json.loads(Path(manifest_path).read_text())
    if not manifest.get("complete") or manifest["repo_id"] != args.repo_id:
        raise ValueError("Snapshot is incomplete or belongs to another repository")
    for record in manifest["files"]:
        relative = PurePosixPath(record["path"])
        if (relative.is_absolute() or ".." in relative.parts or "\\" in record["path"]
                or not record["path"].startswith(("data/vibiomir/", "data/raw/vibiomir/"))):
            raise ValueError(f"Unexpected manifest path: {record['path']}")
        target = args.destination.joinpath(*relative.parts)
        if not target.resolve().is_relative_to(args.destination.resolve()):
            raise ValueError(f"Manifest path escapes destination: {record['path']}")
        if target.exists() and not args.overwrite:
            if target.is_file() and digest(target) == record["sha256"]:
                log(f"Already restored: {record['path']}")
                continue
            raise FileExistsError(f"Restore target exists: {target}; use --overwrite explicitly")
        downloaded = Path(api.hf_hub_download(args.repo_id, record["path"], repo_type="dataset",
                          revision=revision, local_dir=args.destination))
        if digest(downloaded) != record["sha256"]:
            raise RuntimeError(f"Downloaded checksum mismatch: {record['path']}")
    for record in manifest["archives"]:
        marker = scratch / (Path(record["path"]).name + ".restored")
        if marker.exists() and marker.read_text() == record["sha256"]:
            continue
        downloaded = Path(api.hf_hub_download(args.repo_id, record["path"], repo_type="dataset",
                          revision=revision, local_dir=scratch))
        if digest(downloaded) != record["sha256"]:
            raise RuntimeError(f"Downloaded checksum mismatch: {record['path']}")
        unpack_pages(downloaded, args.destination, args.overwrite)
        marker.write_text(record["sha256"])
        downloaded.unlink()
        log(f"Restored {record['path']} ({record['files']:,} pages)")
    log(f"RESTORE COMPLETE {args.destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan")
    planner.add_argument("--inventory", type=Path, required=True)
    planner.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    planner.add_argument("--archive-gib", type=float, default=1.0)
    uploader = commands.add_parser("upload")
    uploader.add_argument("--repo-id", default="duktrung/med-rag")
    uploader.add_argument("--state-dir", type=Path, default=DEFAULT_STATE)
    uploader.add_argument("--plan", type=Path, default=DEFAULT_STATE / "plan.json")
    uploader.add_argument("--wait-for-auth", action="store_true")
    uploader.add_argument("--extract-pid-file", type=Path)
    uploader.add_argument("--extract-log-file", type=Path)
    uploader.add_argument("--extract-output", default="vibiomir/chunks_crawl_20261007.parquet")
    downloader = commands.add_parser("restore")
    downloader.add_argument("--repo-id", default="duktrung/med-rag")
    downloader.add_argument("--destination", type=Path, required=True)
    downloader.add_argument("--revision", default="main")
    downloader.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.command == "plan":
        if args.archive_gib <= 0:
            parser.error("--archive-gib must be positive")
        plan = make_plan(json.loads(args.inventory.read_text()), int(args.archive_gib * 1024 ** 3))
        write_json(args.state_dir / "plan.json", plan)
        log(f"Plan: {len(plan['page_groups'])} archives, {len(plan['regular_files'])} files; "
            f"estimated={plan['estimated_bytes'] / 1e9:.3f} GB")
    elif args.command == "upload":
        import fcntl
        args.state_dir.mkdir(parents=True, exist_ok=True)
        with (args.state_dir / "upload.lock").open("w") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("An uploader already uses this state directory") from error
            upload(args)
    else:
        restore(args)


if __name__ == "__main__":
    main()

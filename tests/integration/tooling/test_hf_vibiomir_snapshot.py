"""Check private transfer scope, archive fidelity, and interruption recovery."""
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from scripts.tooling.hf_vibiomir_snapshot import (
    make_plan, pack_pages, require_private, unpack_pages, verify_remote,
)


class SnapshotTests(unittest.TestCase):
    def test_plan_keeps_vibiomir_and_excludes_other_datasets_and_partial_files(self):
        inventory = {"root": "/source/data", "created_at": 1,
                     "regular_files": [{"path": path, "size": 10} for path in [
                         "vibiomir/chunks_v4.parquet", "raw/vibiomir/query.parquet",
                         "vibiomir/chunks.partial.parquet", "raw/vimedaqa/train.json"]],
                     "page_shards": [{"path": f"vibiomir/crawl/pages/{i:03d}", "tar_size": 10}
                                     for i in range(3)]}
        plan = make_plan(inventory, 20)
        self.assertEqual([entry["path"] for entry in plan["regular_files"]],
                         ["vibiomir/chunks_v4.parquet", "raw/vibiomir/query.parquet"])
        self.assertEqual([len(group) for group in plan["page_groups"]], [2, 1])

    def test_archive_restores_exact_bytes_and_can_resume_after_partial_restore(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / "source"
            shard = source / "vibiomir/crawl/pages/007"
            shard.mkdir(parents=True)
            payloads = {"7.gz": b"\x1f\x8b\x00binary\x00data", "1007.gz": b"second page"}
            for name, payload in payloads.items():
                (shard / name).write_bytes(payload)
            archive = root / "pages.tar"
            record = pack_pages(source, [{"path": "vibiomir/crawl/pages/007", "files": 2,
                                         "size": sum(map(len, payloads.values()))}], archive)
            self.assertEqual(record["files"], 2)
            destination = root / "restored"
            first = destination / "data/vibiomir/crawl/pages/007/7.gz"
            first.parent.mkdir(parents=True)
            first.write_bytes(payloads["7.gz"])
            unpack_pages(archive, destination)
            for name, payload in payloads.items():
                self.assertEqual((first.parent / name).read_bytes(), payload)
            unpack_pages(archive, destination)
            first.write_bytes(b"different")
            with self.assertRaises(FileExistsError):
                unpack_pages(archive, destination)
            self.assertEqual(first.read_bytes(), b"different")

    def test_changed_page_inventory_does_not_publish_a_partial_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shard = root / "data/vibiomir/crawl/pages/001"
            shard.mkdir(parents=True)
            (shard / "1.gz").write_bytes(b"page")
            archive = root / "pages.tar"
            with self.assertRaises(RuntimeError):
                pack_pages(root / "data", [{"path": "vibiomir/crawl/pages/001",
                                           "files": 2, "size": 8}], archive)
            self.assertFalse(archive.exists())
            self.assertFalse(archive.with_suffix(".tar.tmp").exists())

    def test_public_destination_is_rejected(self):
        api = SimpleNamespace(repo_info=lambda *args, **kwargs: SimpleNamespace(private=False))
        with self.assertRaisesRegex(RuntimeError, "not private"):
            require_private(api, "owner/repo")

    def test_verification_detects_wrong_remote_content_and_missing_files(self):
        records = [{"path": "archive.tar", "size": 4, "sha256": "expected"}]
        for files in [[], [SimpleNamespace(path="archive.tar", size=4,
                                          lfs=SimpleNamespace(sha256="wrong"))]]:
            api = SimpleNamespace(get_paths_info=lambda *args, **kwargs: files)
            with self.assertRaises(RuntimeError):
                verify_remote(api, "owner/repo", records)

    def test_restore_rejects_traversal_and_links(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for name, member_type in [("../../escape", tarfile.REGTYPE),
                                      ("data/vibiomir/crawl/pages/001/1.gz", tarfile.SYMTYPE)]:
                archive = root / "unsafe.tar"
                with tarfile.open(archive, "w") as output:
                    member = tarfile.TarInfo(name)
                    member.type = member_type
                    member.size = 1 if member_type == tarfile.REGTYPE else 0
                    member.linkname = "../../escape" if member_type == tarfile.SYMTYPE else ""
                    output.addfile(member, io.BytesIO(b"x") if member.size else None)
                with self.assertRaises(ValueError):
                    unpack_pages(archive, root / "destination")


if __name__ == "__main__":
    unittest.main()

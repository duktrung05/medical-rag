import builtins
import fcntl
import json
import os
import sys
from types import SimpleNamespace

import pytest

from scripts.v4 import run_single_gpu as launcher


def test_process_blocks_gpu_even_with_low_memory_and_zero_utilization():
    gpu = {"memory_used_mib": 4, "utilization_percent": 0}
    assert launcher.busy_reasons(gpu, [{"pid": 123}])
    assert launcher.busy_reasons(gpu, []) == []


@pytest.mark.parametrize("device", ["cpu", "cuda:2", ["cuda:0", "cuda:1"]])
def test_config_rejects_other_device_or_device_list(device):
    with pytest.raises(ValueError):
        launcher.validate_config({"dense": {"enabled": True, "device": device},
                                  "reranker": {"enabled": True, "device": "cuda:0"}})


def test_busy_preflight_never_imports_torch(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_version": "test-corpus", **{
        stage: {"enabled": True, "device": "cuda:0"}
        for stage in ("dense", "reranker")}}))
    report = tmp_path / "report.json"
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["launcher", "--config", str(config),
                                      "--report", str(report), "--preflight-only"])
    monkeypatch.setattr(launcher, "query_state", lambda: (
        {"memory_used_mib": 3720, "utilization_percent": 0}, [{"pid": 123}]))
    original_import = builtins.__import__

    def checked_import(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("Busy preflight must not initialize torch/CUDA")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", checked_import)
    assert launcher.main() == launcher.WAIT_EXIT
    assert json.loads(report.read_text())["status"] == "WAIT_GPU_BUSY"


def test_active_task4_lock_blocks_another_launcher(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    lock_path = tmp_path / "outputs/v4/task4/locks" / f"{launcher.GPU_UUID}.lock"
    lock_path.parent.mkdir(parents=True)
    report = tmp_path / "report.json"
    monkeypatch.setattr(sys, "argv", ["launcher", "--report", str(report), "--preflight-only"])
    with lock_path.open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert launcher.main() == launcher.WAIT_EXIT
    assert json.loads(report.read_text())["status"] == "WAIT_TASK4_GPU_LOCK"


@pytest.mark.parametrize("device_count,uuid", [(2, launcher.GPU_UUID), (1, "GPU-other")])
def test_cuda_must_report_one_device_and_the_pinned_uuid(tmp_path, monkeypatch, device_count, uuid):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_version": "test", **{
        stage: {"enabled": True, "device": "cuda:0"}
        for stage in ("dense", "reranker")}}))
    script = tmp_path / "scripts/v4/worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise AssertionError('worker must not start')\n")
    report = tmp_path / "report.json"
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["launcher", "--config", str(config),
                                      "--report", str(report), "--", "scripts/v4/worker.py"])
    monkeypatch.setattr(launcher, "query_state", lambda: (
        {"memory_used_mib": 4, "utilization_percent": 0}, []))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: device_count,
                                get_device_properties=lambda _: SimpleNamespace(uuid=uuid))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    assert launcher.main() == 1
    assert json.loads(report.read_text())["status"] == "PREFLIGHT_FAILED"


def test_foreign_job_arriving_during_cuda_import_blocks_worker(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_version": "test", **{
        stage: {"enabled": True, "device": "cuda:0"}
        for stage in ("dense", "reranker")}}))
    script = tmp_path / "scripts/v4/worker.py"
    script.parent.mkdir(parents=True)
    script.write_text("raise AssertionError('worker must not start')\n")
    report = tmp_path / "report.json"
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["launcher", "--config", str(config),
                                      "--report", str(report), "--", "scripts/v4/worker.py"])
    states = iter([({"memory_used_mib": 4, "utilization_percent": 0}, []),
                   ({"memory_used_mib": 2222, "utilization_percent": 0},
                    [{"pid": 12345, "gpu_uuid": launcher.GPU_UUID}])])
    monkeypatch.setattr(launcher, "query_state", lambda: next(states))
    monkeypatch.setattr(os, "environ", dict(os.environ))
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: {4, 5, 6, 7})
    monkeypatch.setattr(os, "sched_setaffinity", lambda *_: None)
    monkeypatch.setattr(os, "nice", lambda _: None)
    monkeypatch.setattr(launcher.subprocess, "run", lambda *_, **__: None)
    monkeypatch.setattr(os, "execve", lambda *_: pytest.fail("Late foreign process must block exec"))
    fake_cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1,
                                get_device_properties=lambda _: SimpleNamespace(uuid=launcher.GPU_UUID))
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=fake_cuda))
    assert launcher.main() == launcher.WAIT_EXIT
    assert json.loads(report.read_text())["status"] == "WAIT_GPU_BUSY"

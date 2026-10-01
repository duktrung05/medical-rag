"""The build supervisor must fail visibly on timeout and preserve incomplete outputs."""

import subprocess

import pytest

from scripts import build_dense_index


def test_build_timeout_uses_process_deadline(monkeypatch):
    calls = []

    def run(command, timeout):
        calls.append((command, timeout))
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(build_dense_index.subprocess, 'run', run)
    monkeypatch.setattr('sys.argv', ['build_dense_index', '--timeout-seconds', '23'])
    with pytest.raises(SystemExit) as error:
        build_dense_index.main()
    assert error.value.code == 124
    assert calls[0][1] == 23
    assert '--worker' in calls[0][0]


def test_build_worker_failure_is_not_reported_as_success(monkeypatch):
    monkeypatch.setattr(build_dense_index.subprocess, 'run', lambda *a, **kw: subprocess.CompletedProcess([], 7))
    monkeypatch.setattr('sys.argv', ['build_dense_index'])
    with pytest.raises(SystemExit) as error:
        build_dense_index.main()
    assert error.value.code == 7

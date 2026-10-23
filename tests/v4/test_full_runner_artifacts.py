"""Guard atomic, replay-safe output publication before full inference is available."""
import json

import pytest

from scripts.v4.run_full_baseline import _write_bytes_checked, run


def test_write_bytes_checked_is_replay_safe(tmp_path):
    target = tmp_path / "selected_results.jsonl"
    _write_bytes_checked(target, b'{"id":1}\n')
    _write_bytes_checked(target, b'{"id":1}\n')
    assert target.read_bytes() == b'{"id":1}\n'
    with pytest.raises(ValueError, match="differs from replay"):
        _write_bytes_checked(target, b'{"id":2}\n')
    assert target.read_bytes() == b'{"id":1}\n'


def test_full_runner_refuses_pilot_before_inference(tmp_path):
    index = tmp_path / "pilot"
    index.mkdir()
    (index / "manifest.json").write_text(json.dumps({"scope": "release_parts_pilot", "parts": 2}))
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"corpus_version": "fixture-v1",
                                  "batch": {"result_limit": 200, "query_count": 1200}}))
    with pytest.raises(ValueError, match="922-part V4 index"):
        run(tmp_path, index, config, tmp_path / "output")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "output.building").exists()

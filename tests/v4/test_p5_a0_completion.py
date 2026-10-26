"""A0 completion must reject corrupted scores and never finalize partial input."""
import pytest

from scripts.v4 import complete_p5_a0 as completion
from scripts.v4 import run_p5_incremental_gpu as gpu
from tests.v4.test_p5_incremental_gpu import handoff, checkpoint


def test_readonly_audit_accepts_mixed_lineage_and_rejects_modified_transport(handoff):
    h = handoff
    for qid in (1, 2):
        pair, mapping, committed = checkpoint(h, qid)
        committed['proof'].update(cache_hits=0, completed_utc='2026-10-23T00:00:00+00:00')
        gpu.commit(h.output, committed)
        paths = [h.resolver.resolve(qid)[0], h.output / 'checkpoints' / f'{qid}.json',
                 h.output / 'scores' / f'{qid}.json']
        before = {p: p.read_bytes() for p in paths}
        result, values, hashes = completion.validate_committed(
            h.output, h.resolver, h.cache, h.lock_sha, 'runtime', qid, f'q{qid}')
        assert result == pair and values == [1.0] * 200
        assert hashes['source_sha256'] == mapping['source_sha256']
        assert {p: p.read_bytes() for p in paths} == before
    path = h.staging / 'pairs/2.json'
    bad = gpu.read(path)
    bad['candidates'][0]['score'] += 1
    path.write_bytes(gpu.canonical_bytes(bad))
    with pytest.raises(ValueError, match='Transport differs'):
        completion.validate_committed(h.output, h.resolver, h.cache, h.lock_sha,
                                      'runtime', 2, 'q2')


@pytest.mark.parametrize('corruption', ['score', 'proof', 'cache_hits'])
def test_readonly_audit_rejects_detached_or_unproven_score(handoff, corruption):
    h = handoff
    _, _, committed = checkpoint(h)
    committed['proof'].update(cache_hits=0, completed_utc='2026-10-23T00:00:00+00:00')
    if corruption == 'cache_hits':
        committed['proof']['cache_hits'] = 1
    gpu.commit(h.output, committed)
    if corruption in ('score', 'proof'):
        folder = 'scores' if corruption == 'score' else 'provenance'
        path = h.output / folder / '1.json'
        value = gpu.read(path)
        if corruption == 'score':
            value['scores'][0]['score'] = 2
        else:
            value['device'] = 'cpu'
        path.write_bytes(gpu.canonical_bytes(value))
    with pytest.raises(ValueError):
        completion.validate_committed(h.output, h.resolver, h.cache, h.lock_sha,
                                      'runtime', 1, 'q1')


def test_exit_gate_requires_matching_successful_supervisor_evidence(tmp_path):
    assert not completion.successful_exit(tmp_path, 'launch.json', 'exit.json')
    launch = {'started_utc': 'start', 'child_pid': 123, 'command': ['worker']}
    gpu.write_once(tmp_path / 'launch.json', launch)
    assert not completion.successful_exit(tmp_path, 'launch.json', 'exit.json')
    ended = dict(launch, exit_code=0, ended_utc='end')
    path = tmp_path / 'exit.json'
    path.write_bytes(gpu.canonical_bytes(ended))
    assert completion.successful_exit(tmp_path, 'launch.json', 'exit.json')
    for change in ({'child_pid': 124}, {'exit_code': 1}, {'ended_utc': None}):
        path.write_bytes(gpu.canonical_bytes(ended | change))
        with pytest.raises(ValueError):
            completion.successful_exit(tmp_path, 'launch.json', 'exit.json')


def test_coverage_rejects_phantom_scores_and_inherited_conflicts(tmp_path):
    source, output = tmp_path / 'cpu', tmp_path / 'gpu'
    gpu.write_once(source / 'lineage.json', {'inherited_query_ids': [1]})
    gpu.write_once(output / 'checkpoints/1.json', {})
    status = completion.progress(source, output)
    assert status['cpu_queries'] == status['gpu_queries'] == 1
    assert not status['coverage_complete']
    gpu.write_once(output / 'checkpoints/2.json', {})
    with pytest.raises(ValueError, match='unpublished'):
        completion.progress(source, output)
    gpu.write_once(source / 'pairs/1.json', {})
    with pytest.raises(ValueError, match='conflict'):
        completion.progress(source, output)


@pytest.mark.parametrize('complete,cpu_exit,gpu_exit', [
    (False, True, True), (True, False, True), (True, True, False), (True, True, True)])
def test_ready_gate_requires_full_coverage_and_both_exits(
        tmp_path, monkeypatch, complete, cpu_exit, gpu_exit):
    source, output, logs = tmp_path / 'cpu', tmp_path / 'gpu', tmp_path / 'logs'
    monkeypatch.setattr(completion, 'progress', lambda *a: {'coverage_complete': complete})
    monkeypatch.setattr(completion, 'successful_exit',
                        lambda folder, *a: cpu_exit if folder == logs else gpu_exit)
    status = completion.completion_ready(source, output, logs)
    assert status['ready_to_finalize'] == (complete and cpu_exit and gpu_exit)


def test_watcher_waits_for_both_exits_before_audit_and_finalize(tmp_path, monkeypatch):
    calls = []
    states = iter([{'ready_to_finalize': False}, {'ready_to_finalize': True}])
    monkeypatch.setattr(completion, 'completion_ready', lambda *a: next(states))
    monkeypatch.setattr(completion.time, 'sleep', lambda _: calls.append('wait'))
    monkeypatch.setattr(completion, 'audit', lambda *a: calls.append('audit'))
    manifest = {'status': 'PASS', 'query_count': 1200, 'pair_count': 240000}
    def finalize(*args):
        calls.append('finalize')
        return manifest
    monkeypatch.setattr(gpu, 'finalize_complete', finalize)
    status = completion.watch(tmp_path / 'cpu', tmp_path / 'gpu', tmp_path / 'reports',
                              tmp_path / 'destination', tmp_path / 'logs')
    assert calls == ['wait', 'audit', 'finalize']
    assert status['status'] == 'LOCAL_EXPORT_PASS'
    assert status['official_f2'] == 'UNMEASURED'
    assert status['b1'] == 'DEFERRED_UNTIL_A0_SCORE_REVIEW'


def test_failed_preexport_audit_prevents_finalization(tmp_path, monkeypatch):
    monkeypatch.setattr(completion, 'completion_ready', lambda *a: {'ready_to_finalize': True})
    def fail(*args):
        raise ValueError('corrupt checkpoint')
    monkeypatch.setattr(completion, 'audit', fail)
    def unexpected(*args):
        pytest.fail('Finalization must not run after a failed audit')
    monkeypatch.setattr(gpu, 'finalize_complete', unexpected)
    with pytest.raises(ValueError, match='corrupt checkpoint'):
        completion.watch(tmp_path / 'cpu', tmp_path / 'gpu', tmp_path / 'reports',
                         tmp_path / 'destination', tmp_path / 'logs')

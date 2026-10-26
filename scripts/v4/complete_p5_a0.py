"""Audit scored A0 outputs now; finalize locally after both producers succeed.

No retrieval, model inference, label access, worker restart, or BTC upload.
The running CPU/GPU implementations and their locks are left unchanged.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path
import time
import traceback

import pyarrow.parquet as pq

from scripts.v4 import run_p5_incremental_gpu as gpu
from scripts.v4.run_full_baseline import _all_official_ids
from scripts.v4.run_p5_2_split import finish_pipeline, validate_score_shard
from src.medrag.adapters.validation import canonical_payload, validate_selected
from src.medrag.core.config import load_config
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.pair_lineage import PairLineageResolver, write_atomic_replace, write_once

EXPECTED = set(range(1, 1201))
CPU_LOGS = gpu.ROOT / 'outputs/v4/p5_fast_a0_migration_retry_20261023/logs'
REPORTS = gpu.ROOT / 'outputs/v4/p5_a0_completion_20261023'
DESTINATION = gpu.ROOT / 'outputs/v4/a0_fast_a6000_final_20261023'
SAMPLE_IDS = (1, 2, 53, 130, 280, 281, 282, 380)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def published_ids(folder):
    ids = []
    for path in folder.glob('*.json'):
        if path.stem.isdigit():
            qid = int(path.stem)
            if path.name != f'{qid}.json' or qid not in EXPECTED:
                raise ValueError(f'Unexpected query filename: {path}')
            ids.append(qid)
    return set(ids)


def progress(source, output):
    inherited = set(read(source / 'lineage.json')['inherited_query_ids'])
    local = published_ids(source / 'pairs')
    if inherited & local:
        raise ValueError('Inherited/local query conflict')
    scored = published_ids(output / 'checkpoints')
    ready = inherited | local
    if scored - ready:
        raise ValueError('Scores exist for unpublished CPU queries')
    return {'observed_utc': gpu.utc(), 'cpu_queries': len(ready),
            'gpu_queries': len(scored), 'gpu_pairs': 200 * len(scored),
            'coverage_complete': ready == scored == EXPECTED}


def successful_exit(logs, launch_name, exit_name):
    """Require positive supervisor evidence, independent of PID visibility."""
    launch_path, exit_path = logs / launch_name, logs / exit_name
    if not launch_path.exists() or not exit_path.exists():
        return False
    launch, ended = read(launch_path), read(exit_path)
    if any(ended.get(k) != launch.get(k) for k in ('started_utc', 'child_pid', 'command')):
        raise ValueError(f'Exit belongs to a different launch: {exit_path}')
    if ended.get('exit_code') != 0 or not ended.get('ended_utc'):
        raise ValueError(f'Worker did not exit successfully: {exit_path}')
    return True


def completion_ready(source, output, cpu_logs):
    status = progress(source, output)
    status['cpu_successful_exit'] = successful_exit(cpu_logs, 'launch.json', 'exit.json')
    status['gpu_successful_exit'] = successful_exit(
        output / 'logs', 'worker_latest.json', 'worker_latest_exit.json')
    status['ready_to_finalize'] = (status['coverage_complete']
                                 and status['cpu_successful_exit']
                                 and status['gpu_successful_exit'])
    return status


def check_implementations(*locks):
    for lock in locks:
        for name, expected in lock['implementation_sha256'].items():
            if gpu.sha256_file(gpu.ROOT / name) != expected:
                raise ValueError(f'Frozen implementation changed: {name}')


def validate_committed(output, resolver, cache, lock_sha, runtime_sha, qid, query):
    """Verify originals, transport, proof, and derived scores without rewriting any."""
    original, origin, source_sha = resolver.validate(qid, query)
    pair_path = output / 'staging/pairs' / f'{qid}.json'
    pair = read(pair_path)
    expected = copy.deepcopy(original)
    expected['lock_sha256'] = lock_sha
    if pair != expected:
        raise ValueError(f'Transport differs from original: {qid}')
    mapping = read(output / 'staging/lineage' / f'{qid}.json')
    source_path, _ = resolver.resolve(qid)
    expected_mapping = {
        'id': qid, 'origin': origin, 'source_path': str(source_path),
        'source_sha256': source_sha, 'transport_sha256': gpu.sha256_file(pair_path),
        'pair_count': 200,
        'input_hashes_sha256': gpu.digest([p['input_sha256'] for p in pair['pairs']]),
    }
    if mapping != expected_mapping:
        raise ValueError(f'Transport lineage differs: {qid}')
    checkpoint_path = output / 'checkpoints' / f'{qid}.json'
    checkpoint = read(checkpoint_path)
    score = gpu.validate_checkpoint(checkpoint, pair, mapping, cache.fingerprint, runtime_sha)
    proof = checkpoint['proof']
    if proof.get('cache_hits') != 0 or not proof.get('completed_utc'):
        raise ValueError(f'Fresh GPU provenance missing: {qid}')
    if read(output / 'scores' / f'{qid}.json') != score:
        raise ValueError(f'Derived score conflicts with checkpoint: {qid}')
    if read(output / 'provenance' / f'{qid}.json') != proof:
        raise ValueError(f'Derived proof conflicts with checkpoint: {qid}')
    values = validate_score_shard(score, pair, mapping['transport_sha256'], cache.fingerprint)
    return pair, values, {'checkpoint_sha256': gpu.sha256_file(checkpoint_path),
                          'source_sha256': source_sha, 'score_sha256': gpu.digest(score)}


def audit(source, output, reports, *, sample_ids=SAMPLE_IDS):
    resolver = PairLineageResolver(source)
    runtime_path = output / 'runtime_lock.json'
    runtime, runtime_sha = read(runtime_path), gpu.sha256_file(runtime_path)
    lock_path = output / 'staging/lock.json'
    lock, lock_sha = read(lock_path), gpu.sha256_file(lock_path)
    if (lock != gpu.transport_lock(resolver)
            or runtime['transport_lock_sha256'] != lock_sha
            or runtime['score_contract'] != lock['score_contract']):
        raise ValueError('Runtime/transport contract differs')
    check_implementations(runtime, resolver.lock, resolver.source_lock)
    if gpu.sha256_file(gpu.CONFIG) != gpu.FROZEN_CONFIG_SHA256:
        raise ValueError('Frozen A0 config changed')
    config = load_config(gpu.CONFIG)
    index = gpu.ROOT / config['index_namespace']
    if gpu.sha256_file(index / 'manifest.json') != lock['index_manifest_sha256']:
        raise ValueError('Index manifest changed')
    migration = read(gpu.ROOT / 'configs/vibiomir/phase2_batch_migration_v1.json')
    query_path = gpu.ROOT / migration['official_queries']
    if gpu.sha256_file(query_path) != lock['official_queries_sha256']:
        raise ValueError('Official queries changed')
    queries = {int(r['id']): str(r['query']) for r in pq.read_table(
        query_path, columns=['id', 'query']).to_pylist()}
    if set(queries) != EXPECTED or lock['query_ids'] != sorted(EXPECTED):
        raise ValueError('Official query coverage differs')
    cache = gpu.ScoreCache(output / 'staging/key_contract.sqlite', lock['score_contract'])
    snapshot = read(output / 'ledger.json')
    if snapshot['runtime_sha256'] != runtime_sha:
        raise ValueError('Ledger runtime mismatch')
    ids = sorted(int(q) for q in snapshot['queries'])
    if (not set(ids) <= EXPECTED or snapshot['scored_queries'] != len(ids)
            or snapshot['scored_pairs'] != 200 * len(ids)):
        raise ValueError('Ledger coverage mismatch')
    chosen = sorted(set(sample_ids))
    if not chosen or set(chosen) - set(ids):
        raise ValueError('Output smoke requires all requested sample queries scored')
    pipeline = finish_pipeline(config, index)
    store = DiskCorpusStore(index / 'corpus.sqlite', config['corpus_version'],
                            read(index / 'manifest.json')['fingerprint'])
    selected, checked = {}, {}
    for qid in ids:
        pair, values, hashes = validate_committed(
            output, resolver, cache, lock_sha, runtime_sha, qid, queries[qid])
        entry = snapshot['queries'][str(qid)]
        if any(entry[k] != v for k, v in hashes.items()):
            raise ValueError(f'Ledger/checkpoint mismatch: {qid}')
        checked[str(qid)] = hashes
        if qid in chosen:
            triples = [(p['official_id'], p['passage_id'], p['payload']) for p in pair['pairs']]
            result, _ = pipeline.finish_rerank(copy.deepcopy(pair['candidates']), triples,
                                              values, 200, pair['diagnostics'])
            replay, _ = pipeline.finish_rerank(copy.deepcopy(pair['candidates']), triples,
                                              values, 200, pair['diagnostics'])
            if result != replay:
                raise ValueError(f'Selection replay differs: {qid}')
            if len(result) != 200 or sum(len(d['evidence']) for d in result) != 60:
                raise ValueError(f'A0 document/chunk count differs: {qid}')
            selected[qid] = result
    corpus_path = gpu.ROOT / migration['official_corpus']
    if gpu.sha256_file(corpus_path) != lock['official_corpus_sha256']:
        raise ValueError('Official corpus changed')
    predictions = validate_selected(chosen, selected, store, pipeline.selection_policy,
        result_limit=200, reranker_enabled=True, official_ids=_all_official_ids(corpus_path))
    reports.mkdir(parents=True, exist_ok=True)
    write_once(reports / 'smoke_selected.json', selected)
    payload_path = reports / 'smoke_predictions.json'
    from scripts.v4.run_full_baseline import _write_bytes_checked
    _write_bytes_checked(payload_path, canonical_payload(predictions))
    result = {'status': 'PASS', 'observed_utc': gpu.utc(),
        'validated_queries': len(ids), 'validated_pairs': 200 * len(ids),
        'runtime_sha256': runtime_sha, 'smoke_query_ids': chosen,
        'smoke_documents': 200 * len(chosen), 'smoke_chunks': 60 * len(chosen),
        'smoke_payload_sha256': gpu.sha256_file(payload_path),
        'smoke_source_grounding': 'PASS', 'smoke_selection_replay': 'PASS',
        'completion_implementation_sha256': gpu.sha256_file(Path(__file__)),
        'full_coverage': set(ids) == EXPECTED, 'retrieval_calls': 0, 'model_calls': 0,
        'official_f2': 'UNMEASURED', 'b1': 'DEFERRED_UNTIL_A0_SCORE_REVIEW',
        'validated_checkpoint_hashes': checked}
    # Derived report may be refreshed on resume; inputs and smoke outputs stay immutable.
    write_atomic_replace(reports / 'audit.json', result)
    return result


def watch(source, output, reports, destination, cpu_logs, *, poll_seconds=30,
          timeout_hours=36):
    reports.mkdir(parents=True, exist_ok=True)
    with (reports / 'completion.lock').open('a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        started = time.monotonic()
        while True:
            status = completion_ready(source, output, cpu_logs)
            status.update(status='WAITING_FOR_A0', official_f2='UNMEASURED',
                          b1='DEFERRED_UNTIL_A0_SCORE_REVIEW')
            write_atomic_replace(reports / 'status.json', status)
            print(json.dumps(status), flush=True)
            if status['ready_to_finalize']:
                break
            if time.monotonic() - started > timeout_hours * 3600:
                raise TimeoutError('A0 completion timed out; workers were left untouched')
            time.sleep(poll_seconds)
        audit(source, output, reports / 'pre_finalize_audit')
        manifest = gpu.finalize_complete(source, output, destination)
        if (manifest.get('status') != 'PASS' or manifest.get('query_count') != 1200
                or manifest.get('pair_count') != 240000):
            raise ValueError('Final A0 manifest failed coverage gate')
        status.update(status='LOCAL_EXPORT_PASS', completed_utc=gpu.utc(),
                      destination=str(destination), run_manifest=manifest,
                      next_action='Review A0 relevance/official score before considering B1')
        write_atomic_replace(reports / 'status.json', status)
        print(json.dumps(status), flush=True)
        return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['audit', 'watch'])
    parser.add_argument('--source', type=Path, default=gpu.SOURCE)
    parser.add_argument('--output', type=Path, default=gpu.OUTPUT)
    parser.add_argument('--reports', type=Path, default=REPORTS)
    parser.add_argument('--destination', type=Path, default=DESTINATION)
    parser.add_argument('--cpu-logs', type=Path, default=CPU_LOGS)
    args = parser.parse_args()
    # Report writers and the finalizer must not overlap either live namespace.
    protected = [args.source.resolve(), args.output.resolve(),
                 Path(read(args.source / 'lineage.json')['source_namespace']).resolve()]
    for path in (args.reports.resolve(), args.destination.resolve()):
        if any(path == p or p in path.parents or path in p.parents for p in protected):
            parser.error('Completion artifacts must be outside CPU/GPU namespaces')
    try:
        if args.command == 'audit':
            result = audit(args.source, args.output, args.reports)
        else:
            result = watch(args.source, args.output, args.reports, args.destination, args.cpu_logs)
        print(json.dumps({k: v for k, v in result.items()
                          if k != 'validated_checkpoint_hashes'}, indent=2), flush=True)
    except Exception as exc:
        args.reports.mkdir(parents=True, exist_ok=True)
        write_atomic_replace(args.reports / 'failure.json', {
            'status': 'FAILED', 'observed_utc': gpu.utc(),
            'error': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()

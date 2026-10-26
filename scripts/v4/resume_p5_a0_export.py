"""Resume an existing A0 export with an explicit operational ZIP size limit.

The frozen ranking config and CPU/GPU artifacts retain their original hashes.
Only batch.zip_max_bytes in the in-memory export config is overridden.
"""
from __future__ import annotations

import argparse
import copy
import fcntl
import json
from pathlib import Path

import pyarrow.parquet as pq

from scripts.v4 import complete_p5_a0 as completion
from scripts.v4 import run_p5_2_split as split
from scripts.v4 import run_p5_incremental_gpu as gpu
from src.medrag.core.config import load_config
from src.medrag.data.pair_lineage import write_atomic_replace, write_once


def resume(zip_max_bytes: int) -> dict:
    if not 0 < zip_max_bytes <= 100_000_000:
        raise ValueError('ZIP limit must be positive and at most 100,000,000 bytes')
    source, output = gpu.SOURCE, gpu.OUTPUT
    reports, destination = completion.REPORTS, completion.DESTINATION
    with (reports / 'completion.lock').open('a') as guard, \
            (output / 'worker.lock').open('a') as gpu_guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(gpu_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        status = completion.completion_ready(source, output, completion.CPU_LOGS)
        if not status['ready_to_finalize']:
            raise ValueError('Both workers must have completed successfully')
        export = destination / 'export'
        before = {name: gpu.sha256_file(export / name)
                  for name in ('predictions.json', 'submission.zip')}
        print('Auditing all 1200 query checkpoints / 240000 pairs', flush=True)
        audit = completion.audit(source, output, reports / 'export_100mb_audit')
        if not audit['full_coverage'] or audit['status'] != 'PASS':
            raise ValueError('Complete score audit required')
        lock = split.read(destination / 'lock.json')
        if lock != split.read(output / 'staging/lock.json'):
            raise ValueError('Final destination differs from GPU transport lock')
        if completion.published_ids(destination / 'final') != completion.EXPECTED:
            raise ValueError('Resume requires all 1200 existing final query checkpoints')
        config = load_config(gpu.CONFIG)
        effective_config = copy.deepcopy(config)
        effective_config['batch']['zip_max_bytes'] = zip_max_bytes
        override = {
            'format': 'medrag-v4-a0-export-operational-override-v1',
            'parameter': 'batch.zip_max_bytes',
            'frozen_value': config['batch']['zip_max_bytes'],
            'effective_value': zip_max_bytes,
            'authority': 'User confirmed BTC submission ZIP limit is 100 MB on 2026-10-24',
            'frozen_config_sha256': gpu.sha256_file(gpu.CONFIG),
            'export_before_sha256': before,
            'implementation_sha256': gpu.sha256_file(Path(__file__)),
        }
        write_once(reports / 'export_100mb_override.json', override)
        migration = split.read(gpu.ROOT / 'configs/vibiomir/phase2_batch_migration_v1.json')
        queries = sorted(pq.read_table(gpu.ROOT / migration['official_queries'],
                                      columns=['id', 'query']).to_pylist(),
                         key=lambda row: int(row['id']))
        print('Validating existing final selections, source grounding and ZIP at '
              f'{zip_max_bytes} bytes; reusing all final checkpoints', flush=True)
        manifest = split.finalize(destination, effective_config, lock, queries,
                                  gpu.ROOT / config['index_namespace'],
                                  gpu.ROOT / migration['official_corpus'])
        after = {name: gpu.sha256_file(export / name) for name in before}
        if before != after:
            raise ValueError('Existing export bytes changed during validation')
        if (manifest['status'] != 'PASS' or manifest['query_count'] != 1200
                or manifest['pair_count'] != 240000):
            raise ValueError('Final manifest coverage gate failed')
        status.update(status='LOCAL_EXPORT_PASS', completed_utc=gpu.utc(),
                      destination=str(destination), run_manifest=manifest,
                      operational_override=override,
                      export_bytes_unchanged=True, official_f2='UNMEASURED',
                      b1='DEFERRED_UNTIL_A0_SCORE_REVIEW',
                      next_action='Review A0 relevance/official score before considering B1')
        failure_path = reports / 'failure.json'
        if failure_path.exists():
            status['resolved_prior_failure'] = split.read(failure_path)
        write_atomic_replace(reports / 'status.json', status)
        write_atomic_replace(reports / 'export_100mb_result.json', status)
        return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip-max-bytes', type=int, default=100_000_000)
    args = parser.parse_args()
    print(json.dumps(resume(args.zip_max_bytes), ensure_ascii=False, indent=2), flush=True)

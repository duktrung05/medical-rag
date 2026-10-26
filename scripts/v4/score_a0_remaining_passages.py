"""Score only unscored passages already present in the frozen A0 candidate cache."""
from __future__ import annotations

import argparse
import fcntl
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts.v4.run_p5_2_split import ROOT, CONFIG, FROZEN_CONFIG_SHA256, read, encoded
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.verifier import verify

BASE = ROOT / 'outputs/v4/a0_fast_a6000_final_20261023'
ORIGINAL_GPU = ROOT / 'outputs/v4/p5_gpu_incremental_a6000_20261023'
OUTPUT = ROOT / 'outputs/v4/a0_remaining_passages_20261024'
GPU_UUID = 'GPU-3eb95da2-57af-94b3-2ae9-8e5d46d9c079'


def missing_evidence(candidates):
    result, seen = [], set()
    for doc in candidates:
        for evidence in doc['evidence']:
            identity = (doc['official_id'], evidence['passage_id'])
            if identity in seen:
                raise ValueError('Duplicate candidate pair')
            seen.add(identity)
            if 'rerank_score' not in evidence:
                result.append((doc['official_id'], evidence))
    return result


def gpu_preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('Worker must expose only physical GPU 2 by its pinned UUID')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    memory = {r.split(',')[0].strip(): int(r.split(',')[1]) for r in rows.splitlines()}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if GPU_UUID not in memory or memory[GPU_UUID] > 512 or GPU_UUID in apps:
        raise RuntimeError('Requested GPU is occupied; no model was launched')


def run(output, limit):
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'worker.lock').open('a') as guard, (ORIGINAL_GPU / 'worker.lock').open('a') as old_guard:
        fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(old_guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        gpu_preflight()
        baseline_runtime = read(ORIGINAL_GPU / 'runtime_lock.json')
        for name, expected in baseline_runtime['implementation_sha256'].items():
            if sha256_file(ROOT / name) != expected:
                raise ValueError(f'Baseline implementation changed: {name}')
        if sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
            raise ValueError('Frozen baseline config changed')
        versions = {name: importlib.metadata.version(name) for name in baseline_runtime['versions']}
        if versions != baseline_runtime['versions']:
            raise ValueError('Model package versions differ from baseline')
        import torch
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or torch.version.cuda != baseline_runtime['cuda']
                or torch.cuda.get_device_name(0) != baseline_runtime['gpu_name']):
            raise ValueError('GPU/runtime differs from baseline')
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = baseline_runtime['matmul_allow_tf32']
        torch.backends.cudnn.allow_tf32 = baseline_runtime['cudnn_allow_tf32']
        contract = baseline_runtime['score_contract']
        snapshot = require_cached_revision(contract['model'], contract['revision'])
        for name, expected in baseline_runtime['model_snapshot_files_sha256'].items():
            if sha256_file(snapshot / name) != expected:
                raise ValueError(f'Pinned model file differs: {name}')
        runtime = {'format': 'a0-supplemental-passage-scores-v1',
                   'baseline_runtime_sha256': sha256_file(ORIGINAL_GPU / 'runtime_lock.json'),
                   'baseline_manifest_sha256': sha256_file(BASE / 'run_manifest.json'),
                   'gpu_uuid': GPU_UUID, 'versions': versions, 'score_contract': contract,
                   'selection_scope': 'Every unscored passage in frozen fused candidates; no label access',
                   'implementation_sha256': sha256_file(Path(__file__))}
        write_once(output / 'runtime_lock.json', runtime)
        runtime_sha = sha256_file(output / 'runtime_lock.json')
        index = ROOT / 'outputs/v4/task4/full_index_gpu3_20261019'
        manifest = read(index / 'manifest.json')
        if sha256_file(index / 'manifest.json') != contract['index_manifest_sha256']:
            raise ValueError('Index identity changed')
        store = DiskCorpusStore(index / 'corpus.sqlite', manifest['corpus_version'], manifest['fingerprint'])
        model = None
        counts = {'completed_queries': 0, 'new_pairs': 0, 'model_seconds': 0.0}
        for qid in range(1, (limit or 1200) + 1):
            final_path = BASE / 'final' / f'{qid}.json'
            final = read(final_path)
            pair_path = BASE / 'pairs' / f'{qid}.json'
            if (final['pair_shard_sha256'] != sha256_file(pair_path)
                    or final['score_shard_sha256'] != sha256_file(BASE / 'scores' / f'{qid}.json')):
                raise ValueError(f'Original checkpoint differs: {qid}')
            pair = read(pair_path)
            missing = missing_evidence(final['trace']['ranked_candidates'])
            path = output / 'scores' / f'{qid}.json'
            final_sha = sha256_file(final_path)
            if path.exists():
                score = read(path)
                if (score['runtime_sha256'] != runtime_sha or score['baseline_final_sha256'] != final_sha
                        or score['id'] != qid or len(score['scores']) != len(missing)
                        or [(r['official_id'], r['passage_id']) for r in score['scores']]
                        != [(parent, e['passage_id']) for parent, e in missing]
                        or any(not math.isfinite(r['score']) for r in score['scores'])):
                    raise ValueError(f'Supplemental resume checkpoint mismatch: {qid}')
            else:
                sources = store.sources_batch({parent for parent, _ in missing}) if missing else {}
                passages = store.passages_batch([e['passage_id'] for _, e in missing]) if missing else {}
                payloads = []
                for parent, e in missing:
                    source, passage = sources[parent], passages[e['passage_id']]
                    if (passage.official_id != parent or not verify(passage, source)
                            or (e['text'], e['start'], e['end'])
                            != (passage.text, passage.source_start, passage.source_end)):
                        raise ValueError('Unscored input is not grounded in the registered source')
                    payloads.append(f'{source.title}\n{passage.text}' if source.title else passage.text)
                if model is None and missing:
                    from src.medrag.ranking.reranker import BGEReranker
                    model = BGEReranker(contract['model'], revision=contract['revision'],
                                        max_length=contract['max_length'], device='cuda:0')
                    weight = next(model.model.model.parameters())
                    if weight.device.type != 'cuda' or weight.dtype != torch.float16:
                        raise ValueError('Expected CUDA FP16 model')
                torch.cuda.synchronize()
                started = time.monotonic()
                values = model.score(pair['query'], payloads) if missing else []
                torch.cuda.synchronize()
                seconds = time.monotonic() - started
                if len(values) != len(missing) or any(not math.isfinite(v) for v in values):
                    raise ValueError('Supplemental model score count/values invalid')
                import hashlib
                score = {'id': qid, 'runtime_sha256': runtime_sha,
                         'baseline_final_sha256': final_sha, 'query_sha256': pair['query_sha256'],
                         'seconds': seconds, 'scores': [
                             {'official_id': parent, 'passage_id': e['passage_id'],
                              'input_sha256': hashlib.sha256(payload.encode()).hexdigest(), 'score': value}
                             for (parent, e), payload, value in zip(missing, payloads, values, strict=True)]}
                write_once(path, score)
            counts['completed_queries'] += 1
            counts['new_pairs'] += len(score['scores'])
            counts['model_seconds'] += score['seconds']
            status = {**counts, 'status': 'COMPLETE' if counts['completed_queries'] == 1200 else 'RUNNING',
                      'runtime_sha256': runtime_sha, 'baseline_pairs_reused': 200 * counts['completed_queries']}
            write_atomic_replace(output / 'status.json', status)
            if qid % 25 == 0 or qid == limit:
                print(json.dumps(status), flush=True)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 1200:
        parser.error('--limit must be between 1 and 1200')
    run(args.output, args.limit)

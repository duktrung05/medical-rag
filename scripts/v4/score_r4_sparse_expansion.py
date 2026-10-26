"""Stream R4 sparse expansion pairs through the trained R3 reranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts.v4.prepare_r4_sparse_expansion import OUTPUT as EXPANSION
from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.train_r3_chunk_reranker import GPU_UUID, OUTPUT as TRAINING
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file

OUTPUT = ROOT / 'outputs/v4/r4_sparse_scores_20261024'


def preflight():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('R4 scoring must expose the pinned physical GPU 3 UUID')
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if GPU_UUID in apps:
        raise RuntimeError('Pinned R4 GPU is occupied')


def run(output: Path, wait: bool):
    report = read(TRAINING / 'training_report.json')
    if report['status'] != 'TRAINING_GATE_PASS':
        raise ValueError('R3 trained model did not pass')
    for name, expected in report['model_files_sha256'].items():
        if sha256_file(TRAINING / 'model' / name) != expected:
            raise ValueError('R3 trained model changed')
    preflight()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TRAINING / 'model', local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        TRAINING / 'model', local_files_only=True, torch_dtype=torch.float16).to('cuda:0').eval()
    output.mkdir(parents=True, exist_ok=True)
    _, store, _, _ = environment()
    runtime = {'format': 'v4-r4-sparse-score-runtime-v1',
               'expansion_lock_sha256': sha256_file(EXPANSION / 'lock.json'),
               'training_report_sha256': sha256_file(TRAINING / 'training_report.json'),
               'model_files_sha256': report['model_files_sha256'], 'max_length': 512,
               'precision': 'fp16', 'gpu_uuid': GPU_UUID}
    write_once(output / 'runtime_lock.json', runtime)
    runtime_sha = sha256_file(output / 'runtime_lock.json')
    totals = {'queries': 0, 'pairs': 0, 'model_seconds': 0.0}
    for qid in range(1, 1201):
        source_path = EXPANSION / 'queries' / f'{qid}.json'
        waited = 0
        while not source_path.exists():
            if not wait or waited >= 7200:
                raise RuntimeError(f'R4 expansion query {qid} is unavailable')
            time.sleep(2); waited += 2
        source = read(source_path)
        path = output / 'scores' / f'{qid}.json'
        source_sha = sha256_file(source_path)
        if path.exists():
            result = read(path)
            if (result['runtime_sha256'] != runtime_sha or result['expansion_sha256'] != source_sha
                    or len(result['scores']) != len(source['hits'])):
                raise ValueError('R4 score resume differs')
        else:
            hits = source['hits']; sources = store.sources_batch({x['official_id'] for x in hits}) if hits else {}
            payloads = [f'{sources[x["official_id"]].title}\n{x["text"]}'
                        if sources[x['official_id']].title else x['text'] for x in hits]
            if any(hashlib.sha256(payload.encode()).hexdigest() != row['input_sha256']
                   for payload, row in zip(payloads, hits, strict=True)):
                raise ValueError('R4 payload hash differs')
            query = read(ROOT / 'outputs/v4/a0_fast_a6000_final_20261023/pairs'
                         / f'{qid}.json')['query']
            values = []; torch.cuda.synchronize(); started = time.monotonic()
            for offset in range(0, len(hits), 32):
                batch = payloads[offset:offset + 32]
                encoded = tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                    max_length=512, return_tensors='pt')
                encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**encoded).logits.float().reshape(-1)
                values.extend(float(x) for x in logits.cpu())
            torch.cuda.synchronize(); seconds = time.monotonic() - started
            if len(values) != len(hits) or any(not math.isfinite(x) for x in values):
                raise ValueError('R4 model scores invalid')
            result = {'format': 'v4-r4-sparse-scores-query-v1', 'id': qid,
                      'runtime_sha256': runtime_sha, 'expansion_sha256': source_sha,
                      'seconds': seconds,
                      'scores': [{'official_id': row['official_id'],
                                  'passage_id': row['passage_id'], 'score': score}
                                 for row, score in zip(hits, values, strict=True)]}
            write_once(path, result)
        totals['queries'] += 1; totals['pairs'] += len(result['scores'])
        totals['model_seconds'] += result['seconds']
        if qid % 25 == 0 or qid == 1200:
            status = {**totals, 'status': 'COMPLETE' if qid == 1200 else 'RUNNING',
                      'runtime_sha256': runtime_sha}
            write_atomic_replace(output / 'status.json', status)
            print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    run(args.output, args.wait)

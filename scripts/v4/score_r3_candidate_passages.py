"""Score every passage belonging to the frozen S2 document set with R3."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

from scripts.v4.diagnose_a0_quality import partition
from scripts.v4.refine_a0_submission import OUTPUT as CANDIDATES, environment
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.train_r3_chunk_reranker import GPU_UUID, OUTPUT as TRAINING
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.refinement import filter_documents, select_documents_by_confidence
from src.medrag.evidence.verifier import verify

OUTPUT = ROOT / 'outputs/v4/r3_candidate_scores_20261024'


def gpu_preflight() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('R3 scoring must expose only the pinned physical GPU 3 UUID')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    memory = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in rows.splitlines()}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if GPU_UUID not in memory or memory[GPU_UUID] > 512 or GPU_UUID in apps:
        raise RuntimeError('Pinned R3 GPU is occupied')


def s2_documents(final: dict) -> list[dict]:
    gated = filter_documents(final['selected'], -4.0)
    selected, _ = select_documents_by_confidence(
        gated, threshold=-2.0, min_documents=80, max_documents=140)
    return selected


def run(output: Path, scope: str) -> None:
    report = read(TRAINING / 'training_report.json')
    if report['status'] != 'TRAINING_GATE_PASS':
        raise ValueError('R3 training gate did not pass')
    for name, expected in report['model_files_sha256'].items():
        if sha256_file(TRAINING / 'model' / name) != expected:
            raise ValueError('R3 model checkpoint changed')
    gpu_preflight()
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_path = TRAINING / 'model'
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, torch_dtype=torch.float16).to('cuda:0').eval()
    output.mkdir(parents=True, exist_ok=True)
    config, store, _, _ = environment()
    runtime = {'format': 'v4-r3-candidate-passage-scores-v1',
               'training_report_sha256': sha256_file(TRAINING / 'training_report.json'),
               'model_files_sha256': report['model_files_sha256'],
               'candidate_manifest_sha256': sha256_file(CANDIDATES / 'run_manifest.json'),
               'document_policy': {'name': 'S2', 'pre_gate': -4.0,
                                   'score_threshold': -2.0, 'min_documents': 80,
                                   'max_documents': 140},
               'pair_scope': 'all passages of every S2 document',
               'input_format': 'query paired with title-newline-passage',
               'max_length': 512, 'precision': 'fp16', 'gpu_uuid': GPU_UUID}
    write_once(output / 'runtime_lock.json', runtime)
    runtime_sha = sha256_file(output / 'runtime_lock.json')
    wanted = [qid for qid in range(1, 1201)
              if scope == 'all' or partition(qid) == scope]
    totals = {'queries': 0, 'pairs': 0, 'model_seconds': 0.0}
    for qid in wanted:
        final_path = CANDIDATES / 'final' / f'{qid}.json'
        final = read(final_path)
        docs = s2_documents(final)
        candidates = {doc['official_id']: doc for doc in final['trace']['ranked_candidates']}
        pairs = [(doc['official_id'], evidence) for doc in docs
                 for evidence in candidates[doc['official_id']]['evidence']]
        path = output / 'scores' / f'{qid}.json'
        final_sha = sha256_file(final_path)
        if path.exists():
            result = read(path)
            if (result['runtime_sha256'] != runtime_sha
                    or result['candidate_final_sha256'] != final_sha
                    or len(result['scores']) != len(pairs)
                    or [(x['official_id'], x['passage_id']) for x in result['scores']]
                    != [(doc, evidence['passage_id']) for doc, evidence in pairs]):
                raise ValueError(f'R3 score checkpoint differs: {qid}')
        else:
            query = read(ROOT / 'outputs/v4/a0_fast_a6000_final_20261023/pairs' / f'{qid}.json')['query']
            sources = store.sources_batch({doc for doc, _ in pairs})
            passages = store.passages_batch([evidence['passage_id'] for _, evidence in pairs])
            payloads = []
            for doc, evidence in pairs:
                source, passage = sources[doc], passages[evidence['passage_id']]
                if (passage.official_id != doc or not verify(passage, source)
                        or (passage.text, passage.source_start, passage.source_end)
                        != (evidence['text'], evidence['start'], evidence['end'])):
                    raise ValueError(f'R3 input grounding differs: {qid}/{passage.passage_id}')
                payloads.append(f'{source.title}\n{passage.text}' if source.title else passage.text)
            values = []
            torch.cuda.synchronize(); started = time.monotonic()
            for offset in range(0, len(payloads), 32):
                batch = payloads[offset:offset + 32]
                encoded = tokenizer([query] * len(batch), batch, padding=True, truncation=True,
                                    max_length=512, return_tensors='pt')
                encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = model(**encoded).logits.float().reshape(-1)
                values.extend(float(x) for x in logits.cpu())
            torch.cuda.synchronize(); seconds = time.monotonic() - started
            if len(values) != len(pairs) or any(not math.isfinite(value) for value in values):
                raise ValueError('R3 model returned invalid scores')
            result = {'format': 'v4-r3-query-passage-scores-v1', 'id': qid,
                      'runtime_sha256': runtime_sha, 'candidate_final_sha256': final_sha,
                      'query_sha256': final['query_sha256'], 'seconds': seconds,
                      'scores': [{'official_id': doc, 'passage_id': evidence['passage_id'],
                                  'input_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                                  'score': value}
                                 for (doc, evidence), payload, value in
                                 zip(pairs, payloads, values, strict=True)]}
            write_once(path, result)
        totals['queries'] += 1; totals['pairs'] += len(result['scores'])
        totals['model_seconds'] += result['seconds']
        if totals['queries'] % 25 == 0 or totals['queries'] == len(wanted):
            status = {**totals, 'scope': scope, 'requested_queries': len(wanted),
                      'runtime_sha256': runtime_sha,
                      'status': 'COMPLETE' if totals['queries'] == len(wanted) else 'RUNNING'}
            write_atomic_replace(output / f'status_{scope}.json', status)
            print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--scope', choices=('development', 'validation', 'all'), default='all')
    args = parser.parse_args()
    run(args.output, args.scope)

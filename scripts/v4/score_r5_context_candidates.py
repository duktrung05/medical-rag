"""Score exact R5 candidate contexts with one frozen reranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from scripts.v4.prepare_r5_context_candidates import OUTPUT as CANDIDATES
from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r5_submission_chunks import MODEL_FILES, canonical, preflight
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file


def run(source: Path, output: Path, model: Path, gpu_uuid: str,
        batch_size: int, window_batch_size: int, window_stride: int, aggregation: str,
        start_query: int, end_query: int, wait: bool) -> None:
    source, output, model = source.resolve(), output.resolve(), model.resolve()
    if not model.is_dir() or not (source / 'lock.json').is_file():
        raise FileNotFoundError('Frozen model or R5 candidate lock is unavailable')
    preflight(gpu_uuid)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(
        model, local_files_only=True, torch_dtype=torch.float16).to('cuda:0').eval()
    _, store, _, _ = environment()
    runtime = {
        'format': 'v4-r5-context-score-runtime-v1',
        'candidate_lock_sha256': sha256_file(source / 'lock.json'),
        'model_path': str(model.relative_to(ROOT)),
        'model_files_sha256': {name: sha256_file(model / name) for name in MODEL_FILES},
        'max_length': 512,
        'precision': 'fp16',
        'gpu_uuid': gpu_uuid,
        'batch_size': batch_size,
        'window_batch_size': window_batch_size,
        'window_stride': window_stride,
        'score_aggregation': aggregation,
        'query_range': [start_query, end_query],
        'payload': 'query paired with exact title + final candidate context text',
    }
    if output.exists() and any(output.iterdir()):
        if not (output / 'runtime_lock.json').is_file() or read(output / 'runtime_lock.json') != runtime:
            raise ValueError('R5 score resume runtime differs')
    else:
        output.mkdir(parents=True, exist_ok=True)
        write_once(output / 'runtime_lock.json', runtime)
    runtime_sha = sha256_file(output / 'runtime_lock.json')
    totals = {'queries': 0, 'pairs': 0, 'truncated_pairs': 0, 'model_seconds': 0.0}

    completed = []
    for qid in range(start_query, end_query + 1):
        path = output / 'scores' / f'{qid}.json'
        if not path.exists():
            break
        result = read(path)
        candidate_path = source / 'queries' / f'{qid}.json'
        if (result['runtime_sha256'] != runtime_sha
                or result['candidate_sha256'] != sha256_file(candidate_path)):
            raise ValueError(f'R5 score resume lineage differs: {qid}')
        completed.append(qid)
        totals['queries'] += 1
        totals['pairs'] += len(result['scores'])
        totals['truncated_pairs'] += sum(row['input_tokens'] > 512 for row in result['scores'])
        totals['model_seconds'] += result['seconds']
    expected_existing = set(completed)
    actual_existing = {int(path.stem) for path in (output / 'scores').glob('*.json')}
    if actual_existing != expected_existing:
        raise ValueError('R5 score resume is not a contiguous query prefix')

    next_query = start_query + len(completed)
    for qid in range(next_query, end_query + 1):
        source_path = source / 'queries' / f'{qid}.json'
        waited = 0
        while not source_path.exists():
            if not wait or waited >= 3600:
                raise RuntimeError(f'R5 candidate query is unavailable: {qid}')
            time.sleep(1)
            waited += 1
        candidate = read(source_path)
        query = read(ROOT / 'outputs/v4/a0_fast_a6000_final_20261023/pairs'
                     / f'{qid}.json')['query']
        if hashlib.sha256(query.encode()).hexdigest() != candidate['query_sha256']:
            raise ValueError(f'R5 candidate/query identity differs: {qid}')
        contexts = candidate['contexts']
        sources = store.sources_batch({int(row['official_id']) for row in contexts})
        payloads = []
        for row in contexts:
            title = sources[int(row['official_id'])].title
            payloads.append(f'{title}\n{row["text"]}' if title else row['text'])

        values, token_lengths, window_counts = [], [], []
        started = time.monotonic()
        for offset in range(0, len(contexts), batch_size):
            batch = payloads[offset:offset + batch_size]
            queries = [query] * len(batch)
            lengths = tokenizer(queries, batch, padding=False, truncation=False,
                                return_length=True)['length']
            token_lengths.extend(int(x) for x in lengths)
            if aggregation == 'first_window':
                encoded = tokenizer(queries, batch, padding=True, truncation='only_second',
                                    max_length=512, return_tensors='pt')
                encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                    logits = reranker(**encoded).logits.float().reshape(-1)
                values.extend(float(x) for x in logits.cpu())
                window_counts.extend([1] * len(batch))
            else:
                encoded = tokenizer(queries, batch, padding=True, truncation='only_second',
                                    max_length=512, stride=window_stride,
                                    return_overflowing_tokens=True, return_tensors='pt')
                mapping = encoded.pop('overflow_to_sample_mapping')
                logits = []
                for window_offset in range(0, len(mapping), window_batch_size):
                    model_input = {key: value[window_offset:window_offset + window_batch_size].to('cuda:0')
                                   for key, value in encoded.items()}
                    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                        batch_logits = reranker(**model_input).logits.float().reshape(-1)
                    logits.extend(float(x) for x in batch_logits.cpu())
                grouped = [[] for _ in batch]
                for sample, score in zip(mapping.tolist(), logits, strict=True):
                    grouped[int(sample)].append(score)
                if any(not group for group in grouped):
                    raise ValueError('Tokenizer did not emit a window for every context')
                values.extend(max(group) for group in grouped)
                window_counts.extend(len(group) for group in grouped)
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        if (len(values) != len(contexts) or len(token_lengths) != len(contexts)
                or len(window_counts) != len(contexts)
                or any(not math.isfinite(x) for x in values)):
            raise ValueError(f'Invalid R5 candidate scores: {qid}')
        result = {
            'format': 'v4-r5-context-scores-query-v1',
            'id': qid,
            'candidate_sha256': sha256_file(source_path),
            'runtime_sha256': runtime_sha,
            'seconds': seconds,
            'scores': [
                {'official_id': int(row['official_id']), 'context_id': row['context_id'],
                 'text_sha256': row['text_sha256'],
                 'input_sha256': hashlib.sha256(canonical([query, payload])).hexdigest(),
                 'input_tokens': length, 'window_count': windows, 'score': score}
                for row, payload, length, windows, score in zip(
                    contexts, payloads, token_lengths, window_counts, values, strict=True)
            ],
        }
        write_once(output / 'scores' / f'{qid}.json', result)
        totals['queries'] += 1
        totals['pairs'] += len(values)
        totals['truncated_pairs'] += sum(x > 512 for x in token_lengths)
        totals['model_seconds'] += seconds
        if totals['queries'] % 25 == 0 or qid == end_query:
            status = {**totals, 'status': 'COMPLETE' if qid == end_query else 'RUNNING',
                      'runtime_sha256': runtime_sha}
            write_atomic_replace(output / 'status.json', status)
            print(json.dumps(status), flush=True)
    manifest = read(source / 'manifest.json')
    if (manifest['status'] != 'PASS' or manifest['queries'] != 1200
            or manifest['lock_sha256'] != runtime['candidate_lock_sha256']):
        raise ValueError('Completed R5 candidate manifest differs from score runtime')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=CANDIDATES)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--window-batch-size', type=int, default=128)
    parser.add_argument('--window-stride', type=int, default=128)
    parser.add_argument('--aggregation', choices=('first_window', 'window_max'), default='window_max')
    parser.add_argument('--start-query', type=int, default=1)
    parser.add_argument('--end-query', type=int, default=1200)
    parser.add_argument('--wait', action='store_true')
    args = parser.parse_args()
    if (args.batch_size <= 0 or args.window_batch_size <= 0
            or not 0 <= args.window_stride < 512
            or not 1 <= args.start_query <= args.end_query <= 1200):
        parser.error('invalid batch, stride, or query range')
    run(args.source, args.output, args.model, args.gpu_uuid,
        args.batch_size, args.window_batch_size, args.window_stride, args.aggregation,
        args.start_query, args.end_query, args.wait)

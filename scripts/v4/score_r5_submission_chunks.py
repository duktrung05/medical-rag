"""Score the exact exported S2 chunks with one frozen reranker."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time
import zipfile

from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_p5_2_split import ROOT, read
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file

S2 = ROOT / 'outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip'
ORIGINAL = (ROOT / '.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/snapshots'
            / '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e')
TRAINED = ROOT / 'outputs/v4/r3_chunk_reranker_20261024/model'
MODEL_FILES = ('config.json', 'model.safetensors', 'sentencepiece.bpe.model',
               'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json')


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def preflight(gpu_uuid: str) -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != gpu_uuid:
        raise ValueError('R5 must expose exactly the requested physical GPU UUID')
    rows = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv,noheader,nounits'],
        text=True)
    memory = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in rows.splitlines()}
    apps = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'], text=True)
    if gpu_uuid not in memory or memory[gpu_uuid] > 512 or gpu_uuid in apps:
        raise RuntimeError(f'Requested GPU is unavailable: {gpu_uuid}')


def load_submission() -> dict[int, dict]:
    with zipfile.ZipFile(S2) as archive:
        rows = json.loads(archive.read('predictions.json'))
    result = {int(row['id']): row for row in rows}
    if set(result) != set(range(1, 1201)):
        raise ValueError('S2 does not contain exactly the official 1200 queries')
    return result


def run(output: Path, model: Path, gpu_uuid: str, batch_size: int) -> None:
    output, model = output.resolve(), model.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    if not model.is_dir() or not S2.is_file():
        raise FileNotFoundError('Frozen model or S2 submission is unavailable')
    preflight(gpu_uuid)

    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    reranker = AutoModelForSequenceClassification.from_pretrained(
        model, local_files_only=True, torch_dtype=torch.float16).to('cuda:0').eval()
    predictions = load_submission()
    _, store, _, _ = environment()
    output.mkdir(parents=True, exist_ok=True)
    runtime = {
        'format': 'v4-r5-same-input-runtime-v1',
        'source_submission_sha256': sha256_file(S2),
        'model_path': str(model.relative_to(ROOT)),
        'model_files_sha256': {name: sha256_file(model / name) for name in MODEL_FILES},
        'max_length': 512,
        'precision': 'fp16',
        'gpu_uuid': gpu_uuid,
        'batch_size': batch_size,
        'payload': 'query paired with exact title + exported chunk_text',
    }
    write_once(output / 'runtime_lock.json', runtime)
    runtime_sha = sha256_file(output / 'runtime_lock.json')
    totals = {'queries': 0, 'pairs': 0, 'truncated_pairs': 0, 'model_seconds': 0.0}

    for qid in range(1, 1201):
        row = predictions[qid]
        chunks = row['relevant_chunks']
        if len(chunks) != 60 or len({x['doc_id'] for x in chunks}) != len(chunks):
            raise ValueError(f'Unexpected S2 chunk budget or duplicate parent: {qid}')
        query = read(ROOT / 'outputs/v4/a0_fast_a6000_final_20261023/pairs'
                     / f'{qid}.json')['query']
        sources = store.sources_batch({int(x['doc_id']) for x in chunks})
        payloads = []
        for chunk in chunks:
            doc = int(chunk['doc_id'])
            title = sources[doc].title
            payloads.append(f'{title}\n{chunk["chunk_text"]}' if title else chunk['chunk_text'])

        scores = []
        token_lengths = []
        started = time.monotonic()
        for offset in range(0, len(chunks), batch_size):
            batch = payloads[offset:offset + batch_size]
            queries = [query] * len(batch)
            lengths = tokenizer(queries, batch, padding=False, truncation=False,
                                return_length=True)['length']
            token_lengths.extend(int(x) for x in lengths)
            encoded = tokenizer(queries, batch, padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
            encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = reranker(**encoded).logits.float().reshape(-1)
            scores.extend(float(x) for x in logits.cpu())
        torch.cuda.synchronize()
        seconds = time.monotonic() - started
        if (len(scores) != len(chunks) or len(token_lengths) != len(chunks)
                or any(not math.isfinite(x) for x in scores)):
            raise ValueError(f'Invalid R5 score output: {qid}')
        result = {
            'format': 'v4-r5-same-input-query-v1',
            'id': qid,
            'query_sha256': hashlib.sha256(query.encode()).hexdigest(),
            'runtime_sha256': runtime_sha,
            'seconds': seconds,
            'scores': [
                {'doc_id': int(chunk['doc_id']),
                 'chunk_text_sha256': hashlib.sha256(chunk['chunk_text'].encode()).hexdigest(),
                 'input_sha256': hashlib.sha256(canonical([query, payload])).hexdigest(),
                 'input_tokens': length,
                 'score': score}
                for chunk, payload, length, score in zip(
                    chunks, payloads, token_lengths, scores, strict=True)
            ],
        }
        write_once(output / 'scores' / f'{qid}.json', result)
        totals['queries'] += 1
        totals['pairs'] += len(scores)
        totals['truncated_pairs'] += sum(x > 512 for x in token_lengths)
        totals['model_seconds'] += seconds
        if qid % 25 == 0 or qid == 1200:
            status = {**totals, 'status': 'COMPLETE' if qid == 1200 else 'RUNNING',
                      'runtime_sha256': runtime_sha}
            write_atomic_replace(output / 'status.json', status)
            print(json.dumps(status), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--gpu-uuid', required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error('--batch-size must be positive')
    run(args.output, args.model, args.gpu_uuid, args.batch_size)

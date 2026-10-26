"""Prepare and train the query-disjoint R3 evidence reranker pilot."""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sqlite3
import subprocess
import time
import zipfile

from scripts.v4.diagnose_a0_quality import ROOT, partition
from scripts.v4.run_p5_2_split import read
from src.medrag.core.model_revision import require_cached_revision
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file

OUTPUT = ROOT / 'outputs/v4/r3_chunk_reranker_20261024'
FEEDBACK = ROOT / 'outputs/scored_feedback_recheck_20261009/JOINED_SCORED_CHUNKS.csv'
OLD_ZIP = ROOT / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'
INDEX = ROOT / 'outputs/v4/task4/full_index_gpu3_20261019'
MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
GPU_UUID = 'GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7'
SEED = 20261024


def canonical(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()


def title_lookup(ids: set[int]) -> dict[int, str]:
    result = {}
    ordered = sorted(ids)
    with sqlite3.connect(f'file:{INDEX / "corpus.sqlite"}?mode=ro', uri=True) as db:
        for offset in range(0, len(ordered), 800):
            batch = ordered[offset:offset + 800]
            marks = ','.join('?' for _ in batch)
            result.update({int(doc): title for doc, title in db.execute(
                f'SELECT official_id,title FROM sources WHERE official_id IN ({marks})', batch)})
    return result


def prepare(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(OLD_ZIP) as archive:
        predictions = {int(row['id']): row for row in json.loads(archive.read('predictions.json'))}
    rows = []
    with FEEDBACK.open() as stream:
        for raw in csv.DictReader(stream):
            if raw['scoring_status'] != 'scored':
                continue
            qid, order = int(raw['query_id']), int(raw['chunk_order'])
            chunk = predictions[qid]['relevant_chunks'][order]
            doc = int(raw['baseline_doc_id_int'])
            text = chunk['chunk_text']
            if (chunk['doc_id'] != doc or int(raw['baseline_chunk_index']) < 0
                    or hashlib.sha256(text.encode()).hexdigest() != raw['chunk_text_sha256']):
                raise ValueError('Feedback row differs from the frozen scored submission')
            score = float(raw['rerank_logit'])
            if not math.isfinite(score):
                raise ValueError('Feedback contains a non-finite baseline logit')
            rows.append({'query_id': qid, 'query': raw['query_text'], 'doc_id': doc,
                         'chunk_order': order, 'text': text,
                         'label': raw['relevant'] == 'True', 'baseline_score': score,
                         'partition': partition(qid)})
    titles = title_lookup({row['doc_id'] for row in rows})
    for row in rows:
        title = titles.get(row['doc_id'], '')
        row['title'] = title
        row['payload'] = f'{title}\n{row["text"]}' if title else row['text']
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['query_id']].append(row)
    triples = []
    skipped_no_positive = []
    for qid in sorted(grouped):
        if partition(qid) != 'development':
            continue
        positives = sorted((r for r in grouped[qid] if r['label']),
                           key=lambda r: (r['chunk_order'], r['doc_id']))
        negatives = sorted((r for r in grouped[qid] if not r['label']),
                           key=lambda r: (-r['baseline_score'], r['chunk_order'], r['doc_id']))
        if not positives:
            skipped_no_positive.append(qid)
            continue
        if not negatives:
            raise ValueError(f'Positive query has no negative comparator: {qid}')
        # Four deterministic hard negatives per positive, rotated to avoid one repeated item.
        for p_index, positive in enumerate(positives):
            for offset in range(4):
                negative = negatives[(p_index * 4 + offset) % len(negatives)]
                triples.append({'query_id': qid, 'query': positive['query'],
                                'positive': positive['payload'], 'negative': negative['payload'],
                                'positive_doc_id': positive['doc_id'],
                                'negative_doc_id': negative['doc_id']})
    validation = [row for row in rows if row['partition'] == 'validation']
    train_queries = {row['query_id'] for row in triples}
    validation_queries = {row['query_id'] for row in validation}
    if train_queries & validation_queries:
        raise ValueError('Training and validation query IDs overlap')
    for name, values in (('train_triples.jsonl', triples), ('validation_examples.jsonl', validation)):
        with (output / name).open('x') as stream:
            for value in values:
                stream.write(json.dumps(value, ensure_ascii=False, separators=(',', ':')) + '\n')
    counts = {
        'feedback_rows': len(rows),
        'positive_rows': sum(r['label'] for r in rows),
        'negative_rows': sum(not r['label'] for r in rows),
        'train_triples': len(triples),
        'train_queries': len(train_queries),
        'validation_examples': len(validation),
        'validation_queries': len(validation_queries),
        'validation_positives': sum(r['label'] for r in validation),
        'validation_negatives': sum(not r['label'] for r in validation),
        'development_queries_without_positive_excluded': len(skipped_no_positive),
        'titles_missing_from_v4_store': sum(not r['title'] for r in rows),
    }
    manifest = {'format': 'v4-r3-query-disjoint-training-data-v1', 'seed': SEED,
                'model': MODEL, 'revision': REVISION, 'negative_policy':
                'four highest-baseline-score negatives per positive with deterministic rotation',
                'partition_function': 'scripts.v4.diagnose_a0_quality.partition',
                'counts': counts, 'inputs_sha256': {
                    str(FEEDBACK.relative_to(ROOT)): sha256_file(FEEDBACK),
                    str(OLD_ZIP.relative_to(ROOT)): sha256_file(OLD_ZIP),
                    str((INDEX / 'manifest.json').relative_to(ROOT)): sha256_file(INDEX / 'manifest.json')},
                'files_sha256': {name: sha256_file(output / name) for name in
                                 ('train_triples.jsonl', 'validation_examples.jsonl')}}
    write_once(output / 'data_manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


def gpu_preflight() -> None:
    if os.environ.get('CUDA_VISIBLE_DEVICES') != GPU_UUID:
        raise ValueError('R3 must expose only the pinned physical GPU 3 UUID')
    rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,memory.used',
                                    '--format=csv,noheader,nounits'], text=True)
    memory = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in rows.splitlines()}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader'], text=True)
    if GPU_UUID not in memory or memory[GPU_UUID] > 512 or GPU_UUID in apps:
        raise RuntimeError('Pinned R3 GPU is occupied')


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def metrics(labels, scores, query_ids) -> dict:
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    if len(labels) != len(scores) or not set(labels) == {0, 1} or not np.isfinite(scores).all():
        raise ValueError('Invalid evaluation labels or scores')
    by_query = defaultdict(list)
    for qid, label, score in zip(query_ids, labels, scores, strict=True):
        by_query[int(qid)].append((int(label), float(score)))
    aps = []
    reciprocal_ranks = []
    for values in by_query.values():
        qlabels = [x[0] for x in values]
        if not any(qlabels) or all(qlabels):
            continue
        qscores = [x[1] for x in values]
        aps.append(float(average_precision_score(qlabels, qscores)))
        ranked = sorted(values, key=lambda x: -x[1])
        reciprocal_ranks.append(1 / next(i for i, x in enumerate(ranked, 1) if x[0]))
    return {'roc_auc': float(roc_auc_score(labels, scores)),
            'average_precision': float(average_precision_score(labels, scores)),
            'query_macro_average_precision': sum(aps) / len(aps),
            'query_macro_mrr': sum(reciprocal_ranks) / len(reciprocal_ranks),
            'evaluated_queries_with_both_classes': len(aps), 'examples': len(labels),
            'positives': int(labels.sum())}


def score_examples(model, tokenizer, examples, *, batch_size: int, device: str) -> list[float]:
    import torch
    result = []
    model.eval()
    for offset in range(0, len(examples), batch_size):
        batch = examples[offset:offset + batch_size]
        encoded = tokenizer([row['query'] for row in batch], [row['payload'] for row in batch],
                            padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            values = model(**encoded).logits.float().reshape(-1)
        result.extend(float(x) for x in values.cpu())
    return result


def train(output: Path, epochs: int) -> None:
    manifest = read(output / 'data_manifest.json')
    for name, expected in manifest['files_sha256'].items():
        if sha256_file(output / name) != expected:
            raise ValueError('Prepared training data changed')
    gpu_preflight()
    import numpy as np
    import torch
    from torch.nn import functional as F
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              get_linear_schedule_with_warmup)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    snapshot = require_cached_revision(MODEL, REVISION)
    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        snapshot, local_files_only=True, torch_dtype=torch.float32).to('cuda:0')
    model.gradient_checkpointing_enable()
    model.config.use_cache = False
    triples = load_jsonl(output / 'train_triples.jsonl')
    validation = load_jsonl(output / 'validation_examples.jsonl')
    labels = [int(row['label']) for row in validation]
    qids = [row['query_id'] for row in validation]
    baseline = metrics(labels, [row['baseline_score'] for row in validation], qids)
    batch_size, accumulation = 4, 4
    updates_per_epoch = math.ceil(math.ceil(len(triples) / batch_size) / accumulation)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=max(1, updates_per_epoch // 10),
        num_training_steps=updates_per_epoch * epochs)
    scaler = torch.amp.GradScaler('cuda')
    history = []
    best_auc = -1.0
    model_dir = output / 'model'
    started = time.monotonic()
    for epoch in range(1, epochs + 1):
        order = list(range(len(triples)))
        random.Random(SEED + epoch).shuffle(order)
        model.train(); optimizer.zero_grad(set_to_none=True); losses = []
        for batch_index, start in enumerate(range(0, len(order), batch_size), 1):
            batch = [triples[i] for i in order[start:start + batch_size]]
            queries = [row['query'] for row in batch]
            payloads = [row['positive'] for row in batch] + [row['negative'] for row in batch]
            encoded = tokenizer(queries + queries, payloads, padding=True, truncation=True,
                                max_length=512, return_tensors='pt')
            encoded = {key: value.to('cuda:0') for key, value in encoded.items()}
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                logits = model(**encoded).logits.float().reshape(-1)
                size = len(batch)
                loss = F.softplus(-(logits[:size] - logits[size:])).mean() / accumulation
            scaler.scale(loss).backward(); losses.append(float(loss.detach()) * accumulation)
            if batch_index % accumulation == 0 or start + batch_size >= len(order):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
            if batch_index % 50 == 0:
                print(json.dumps({'epoch': epoch, 'batches': batch_index,
                                  'total_batches': math.ceil(len(order) / batch_size),
                                  'mean_loss': sum(losses[-50:]) / min(50, len(losses))}), flush=True)
        scores = score_examples(model, tokenizer, validation, batch_size=32, device='cuda:0')
        current = metrics(labels, scores, qids)
        row = {'epoch': epoch, 'mean_train_loss': sum(losses) / len(losses),
               'validation': current}
        history.append(row)
        print(json.dumps(row, indent=2), flush=True)
        if current['roc_auc'] > best_auc:
            best_auc = current['roc_auc']
            model.save_pretrained(model_dir, safe_serialization=True)
            tokenizer.save_pretrained(model_dir)
    trained = max((row for row in history), key=lambda row: row['validation']['roc_auc'])
    gate = (trained['validation']['roc_auc'] >= baseline['roc_auc'] + 0.03
            and trained['validation']['average_precision'] > baseline['average_precision'])
    files = {str(path.relative_to(model_dir)): sha256_file(path)
             for path in sorted(model_dir.rglob('*')) if path.is_file()}
    report = {'format': 'v4-r3-pairwise-reranker-training-v1',
              'status': 'TRAINING_GATE_PASS' if gate else 'TRAINING_GATE_FAIL',
              'model': MODEL, 'source_revision': REVISION, 'seed': SEED,
              'objective': 'pairwise softplus positive-minus-hard-negative',
              'epochs': epochs, 'batch_size_triples': batch_size,
              'gradient_accumulation': accumulation, 'learning_rate': 1e-6,
              'max_length': 512, 'baseline_validation': baseline,
              'best': trained, 'history': history, 'wall_seconds': time.monotonic() - started,
              'gpu_uuid': GPU_UUID, 'torch': torch.__version__,
              'transformers': importlib.metadata.version('transformers'),
              'data_manifest_sha256': sha256_file(output / 'data_manifest.json'),
              'model_files_sha256': files,
              'gate': {'roc_auc_minimum_delta': 0.03,
                       'average_precision_must_improve': True}}
    write_once(output / 'training_report.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('prepare', 'train'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    if args.epochs <= 0:
        parser.error('--epochs must be positive')
    if args.command == 'prepare':
        prepare(args.output)
    else:
        train(args.output, args.epochs)

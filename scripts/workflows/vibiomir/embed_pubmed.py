"""Encode only usable PubMed delta records on exactly two bounded GPU workers."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sqlite3
import subprocess
import time

import numpy as np

from scripts.workflows.vibiomir.collect_pubmed import atomic_json, file_sha, now, ResourceGate


def available_devices():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True)
    available = []
    for line in output.splitlines():
        index, free, utilization = [int(x.strip()) for x in line.split(',')]
        if free >= 16 * 1024 and utilization <= 10:
            available.append(index)
    if len(available) < 2:
        raise RuntimeError('Two available GPUs not found; leave embedding pending')
    return available[:2]


def part_fingerprint(rows, config):
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for row in rows:
        digest.update(f'{row[0]}:{row[2]}\n'.encode())
    return digest.hexdigest()


def encode_shard(shard, device, state_path, output_path, model_path, revision, cores):
    os.nice(10)
    os.sched_setaffinity(0, cores)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import torch
    from sentence_transformers import SentenceTransformer

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(device)
    free, total = torch.cuda.mem_get_info(device)
    if free < 16 * 1024**3:
        raise RuntimeError('Insufficient free GPU memory before load')
    torch.cuda.set_per_process_memory_fraction(min(0.25, 12 * 1024**3 / total), device)
    gate = ResourceGate()
    gate.wait()
    model = SentenceTransformer(str(model_path), device=f'cuda:{device}', local_files_only=True,
                                model_kwargs={'torch_dtype': torch.float16})
    model.max_seq_length = 384
    config = {'model': 'BAAI/bge-m3', 'model_revision': revision,
              'max_seq_length': 384, 'normalize_embeddings': True,
              'precision': 'fp16', 'representation': 'title_and_complete_abstract_v1'}
    con = sqlite3.connect(f'file:{state_path}?mode=ro', uri=True)
    con.execute('PRAGMA cache_size=-4096')
    cursor = con.execute("SELECT id,text,text_sha256 FROM documents WHERE gpu_shard=? AND status='usable_abstract' ORDER BY id", (shard,))
    output = Path(output_path) / f'shard_{shard}'
    output.mkdir(parents=True, exist_ok=True)
    count = part_number = 0
    started = time.monotonic()

    def prepare(row):
        ident, text, digest = row
        if hashlib.sha256(text.encode()).hexdigest() != digest:
            raise ValueError('Text fingerprint mismatch')
        tokens = len(model.tokenizer.encode(text, add_special_tokens=True, truncation=False))
        return ident, text, digest, tokens

    with ThreadPoolExecutor(max_workers=4) as pool:
        while rows := cursor.fetchmany(2000):
            fingerprint = part_fingerprint(rows, config)
            destination = output / f'part_{part_number:05d}.npz'
            metadata_path = output / f'part_{part_number:05d}.json'
            if destination.exists() and metadata_path.exists():
                metadata = json.loads(metadata_path.read_text())
                if metadata['input_fingerprint'] == fingerprint and metadata['sha256'] == file_sha(destination):
                    count += len(rows)
                    part_number += 1
                    continue
            gate.wait()
            prepared = list(pool.map(prepare, rows))
            vectors = []
            batch_size = 32
            position = 0
            while position < len(prepared):
                gate.wait()
                free, _ = torch.cuda.mem_get_info(device)
                if free < 8 * 1024**3:
                    print('GPU headroom pause', device, flush=True)
                    time.sleep(10)
                    continue
                subset = prepared[position:position+batch_size]
                try:
                    encoded = model.encode([x[1] for x in subset], batch_size=batch_size,
                                           normalize_embeddings=True, convert_to_numpy=True,
                                           show_progress_bar=False)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if batch_size <= 8:
                        raise
                    batch_size //= 2
                    continue
                encoded = np.asarray(encoded, dtype=np.float16)
                if encoded.shape != (len(subset), 1024) or not np.isfinite(encoded).all():
                    raise ValueError('Invalid embedding output')
                vectors.append(encoded)
                position += len(subset)
            arrays = np.concatenate(vectors)
            norms = np.linalg.norm(arrays.astype(np.float32), axis=1)
            if not np.allclose(norms, 1.0, atol=0.005):
                raise ValueError('Vectors are not normalized')
            temp = destination.with_suffix('.tmp')
            with temp.open('wb') as stream:
                np.savez(stream, ids=np.array([x[0] for x in prepared], dtype=np.int64), vectors=arrays,
                         token_counts=np.array([x[3] for x in prepared], dtype=np.int32),
                         text_sha256=np.array([x[2] for x in prepared]))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp, destination)
            count += len(rows)
            atomic_json(metadata_path, {'input_fingerprint': fingerprint, 'sha256': file_sha(destination),
                       'rows': len(rows), 'config': config,
                       'truncated_documents': sum(x[3] > 384 for x in prepared), 'finished_at': now()})
            atomic_json(output / 'progress.json', {'shard': shard, 'device': device, 'rows': count,
                        'preparation_workers': 4, 'encoder_processes': 1, 'batch_size': batch_size,
                        'peak_cuda_gib': round(torch.cuda.max_memory_reserved(device)/1024**3, 3),
                        'elapsed_seconds': round(time.monotonic()-started, 1), 'updated_at': now()})
            print(f'Shard {shard}, GPU {device}: encoded {count} documents', flush=True)
            part_number += 1
    atomic_json(output / 'complete.json', {'shard': shard, 'device': device, 'rows': count,
                'parts': part_number, 'model_revision': revision, 'complete': True, 'finished_at': now()})
    con.close()


def verify_and_merge(state_path, output, corpus_revision):
    con = sqlite3.connect(f'file:{state_path}?mode=ro', uri=True)
    expected_ids = np.array([x[0] for x in con.execute("SELECT id FROM documents WHERE status='usable_abstract' ORDER BY id")], dtype=np.int64)
    count = len(expected_ids)
    id_path = output / 'pubmed_document_ids.npy'
    vec_path = output / 'pubmed_document_vectors.npy'
    token_path = output / 'pubmed_document_token_counts.npy'
    vectors = np.lib.format.open_memmap(vec_path.with_suffix('.tmp.npy'), mode='w+', dtype=np.float16, shape=(count,1024))
    tokens = np.lib.format.open_memmap(token_path.with_suffix('.tmp.npy'), mode='w+', dtype=np.int32, shape=(count,))
    seen = np.zeros(count, dtype=np.bool_)
    parts = []
    devices = []
    truncated = 0
    for shard in range(2):
        complete = json.loads((output / f'shard_{shard}/complete.json').read_text())
        devices.append(complete['device'])
        for part_number in range(complete['parts']):
            path = output / f'shard_{shard}/part_{part_number:05d}.npz'
            metadata = json.loads(path.with_suffix('.json').read_text())
            truncated += metadata.get('truncated_documents', 0)
            if file_sha(path) != metadata['sha256']:
                raise ValueError('Part checksum mismatch')
            with np.load(path, allow_pickle=False) as part:
                ids = part['ids']
                locations = np.searchsorted(expected_ids, ids)
                if np.any(locations >= count) or not np.array_equal(expected_ids[locations], ids) or seen[locations].any():
                    raise ValueError('Missing, unexpected or duplicate embedding IDs')
                vectors[locations] = part['vectors']
                tokens[locations] = part['token_counts']
                seen[locations] = True
            parts.append({'path': str(path), 'sha256': metadata['sha256'], 'rows': metadata['rows']})
    if not seen.all():
        raise ValueError('Embedding IDs do not cover usable documents')
    vectors.flush()
    tokens.flush()
    del vectors, tokens
    os.replace(vec_path.with_suffix('.tmp.npy'), vec_path)
    os.replace(token_path.with_suffix('.tmp.npy'), token_path)
    np.save(id_path, expected_ids)
    norms_min, norms_max = 2.0, 0.0
    loaded = np.load(vec_path, mmap_mode='r')
    for start in range(0,count,10000):
        values = np.asarray(loaded[start:start+10000], dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError('Non-finite merged vectors')
        norms = np.linalg.norm(values,axis=1)
        if not np.allclose(norms,1,atol=.005):
            raise ValueError('Invalid merged normalization')
        norms_min, norms_max = min(norms_min,float(norms.min())),max(norms_max,float(norms.max()))
    atomic_json(output / 'embedding_report.json', {'complete': True, 'corpus_revision': corpus_revision,
                'rows': count, 'dimensions': 1024, 'gpu_count': 2, 'preparation_workers_per_gpu': 4,
                'physical_gpu_devices': devices, 'documents_truncated_in_encoder': truncated,
                'normalization_min_max': [norms_min,norms_max], 'parts': parts,
                'files': {str(p.name):file_sha(p) for p in [id_path,vec_path,token_path]},
                'finished_at': now(), 'scope': 'PubMed delta only; existing full retrieval runtime not switched'})
    con.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--model-path', type=Path, required=True)
    parser.add_argument('--model-revision', required=True)
    parser.add_argument('--corpus-revision', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(f'file:{args.state.resolve()}?mode=ro', uri=True)
    unfinished = con.execute("SELECT count(*) FROM documents WHERE status NOT IN ('usable_abstract','metadata_only','unavailable')").fetchone()[0]
    actual_revision = con.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
    if unfinished or actual_revision != args.corpus_revision:
        raise ValueError('Collection incomplete or revision mismatch')
    con.close()
    devices = available_devices()
    cores = sorted(os.sched_getaffinity(0))
    budget = min(8,max(2,len(cores)//2))
    os.nice(10)
    os.sched_setaffinity(0, cores[:budget])
    ctx = mp.get_context('spawn')
    processes = []
    for shard, device in enumerate(devices):
        subset = cores[shard*(budget//2):(shard+1)*(budget//2)]
        process = ctx.Process(target=encode_shard,args=(shard,device,args.state.resolve(),args.output.resolve(),
                    args.model_path.resolve(),args.model_revision,subset))
        process.start()
        processes.append(process)
    for process in processes:
        process.join()
    if any(p.exitcode != 0 for p in processes):
        raise RuntimeError('Encoder shard failed; checkpoints kept for resume')
    verify_and_merge(args.state.resolve(),args.output.resolve(),args.corpus_revision)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

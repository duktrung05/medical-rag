"""Prepare a fast high-IDF BM25 passage expansion outside the frozen R3 pool."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import time
from pathlib import Path

from scripts.v4.refine_a0_submission import OUTPUT as CANDIDATES
from scripts.v4.run_p5_2_split import ROOT, read
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.sql_aggregate_bm25 import SQLAggregateDiskBM25

OUTPUT = ROOT / 'outputs/v4/r4_sparse_expansion_20261024'
INDEX = ROOT / 'outputs/v4/task4/full_index_gpu3_20261019'
TOP_K = 500
MAX_NEW = 150
MAX_QUERY_TERMS = 12


def prepare_one(qid: int, output: str) -> dict:
    output = Path(output)
    manifest = read(INDEX / 'manifest.json')
    bm25 = SQLAggregateDiskBM25(INDEX / 'passages.sqlite', 'passage_bm25',
                               manifest['fingerprint'], max_query_terms=MAX_QUERY_TERMS)
    store = DiskCorpusStore(INDEX / 'corpus.sqlite', manifest['corpus_version'],
                            manifest['fingerprint'])
    final_path = CANDIDATES / 'final' / f'{qid}.json'
    final = read(final_path)
    pair = read(ROOT / 'outputs/v4/a0_fast_a6000_final_20261023/pairs' / f'{qid}.json')
    if hashlib.sha256(pair['query'].encode()).hexdigest() != final['query_sha256']:
        raise ValueError('R4 query identity differs')
    existing = {(doc['official_id'], evidence['passage_id'])
                for doc in final['trace']['ranked_candidates'] for evidence in doc['evidence']}
    started = time.monotonic()
    hits = bm25.search(pair['query'], TOP_K)
    fresh = []
    seen = set(existing)
    for hit in hits:
        identity = (hit.official_id, hit.passage_id)
        if identity in seen:
            continue
        seen.add(identity)
        fresh.append(hit)
        if len(fresh) >= MAX_NEW:
            break
    passages = store.passages_batch([hit.passage_id for hit in fresh]) if fresh else {}
    sources = store.sources_batch({hit.official_id for hit in fresh}) if fresh else {}
    rows = []
    for hit in fresh:
        passage, source = passages[hit.passage_id], sources[hit.official_id]
        if passage.official_id != hit.official_id or not verify(passage, source):
            raise ValueError('R4 BM25 hit is not grounded')
        payload = f'{source.title}\n{passage.text}' if source.title else passage.text
        rows.append({'official_id': hit.official_id, 'passage_id': hit.passage_id,
                     'text': passage.text, 'start': passage.source_start,
                     'end': passage.source_end, 'bm25_rank': hit.rank,
                     'bm25_score': hit.score,
                     'input_sha256': hashlib.sha256(payload.encode()).hexdigest()})
    result = {'format': 'v4-r4-sparse-expansion-query-v1', 'id': qid,
              'query_sha256': final['query_sha256'],
              'candidate_final_sha256': sha256_file(final_path),
              'top_k': TOP_K, 'max_new': MAX_NEW, 'max_query_terms': MAX_QUERY_TERMS,
              'retrieval_seconds': time.monotonic() - started,
              'hits': rows}
    write_once(output / 'queries' / f'{qid}.json', result)
    return {'id': qid, 'hits': len(rows), 'seconds': result['retrieval_seconds']}


def run(output: Path, workers: int):
    output.mkdir(parents=True, exist_ok=False)
    lock = {'format': 'v4-r4-sparse-expansion-lock-v1', 'top_k': TOP_K,
            'max_new_pairs_per_query': MAX_NEW, 'max_query_terms': MAX_QUERY_TERMS,
            'query_term_policy': 'highest corpus IDF, lexical tie break',
            'index_manifest_sha256': sha256_file(INDEX / 'manifest.json'),
            'candidate_manifest_sha256': sha256_file(CANDIDATES / 'run_manifest.json'),
            'implementation_sha256': {
                str(Path(__file__).resolve().relative_to(ROOT)): sha256_file(Path(__file__).resolve()),
                'src/medrag/retrieval/sql_aggregate_bm25.py': sha256_file(
                    ROOT / 'src/medrag/retrieval/sql_aggregate_bm25.py')}}
    write_once(output / 'lock.json', lock)
    totals = {'queries': 0, 'new_pairs': 0, 'retrieval_seconds': 0.0}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare_one, qid, str(output)): qid for qid in range(1, 1201)}
        for future in as_completed(futures):
            row = future.result()
            totals['queries'] += 1; totals['new_pairs'] += row['hits']
            totals['retrieval_seconds'] += row['seconds']
            if totals['queries'] % 25 == 0 or totals['queries'] == 1200:
                status = {**totals, 'workers': workers,
                          'status': 'COMPLETE' if totals['queries'] == 1200 else 'RUNNING'}
                write_atomic_replace(output / 'status.json', status)
                print(json.dumps(status), flush=True)
    hashes = {str(qid): sha256_file(output / 'queries' / f'{qid}.json')
              for qid in range(1, 1201)}
    manifest = {'format': 'v4-r4-sparse-expansion-manifest-v1', 'status': 'PASS',
                **totals, 'lock_sha256': sha256_file(output / 'lock.json'),
                'query_sha256': hashes}
    write_once(output / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k != 'query_sha256'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('--workers must be between 1 and 16')
    run(args.output, args.workers)

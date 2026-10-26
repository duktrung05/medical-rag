"""Materialize exact export contexts for every evidence seed inside S2 documents."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import zipfile

from scripts.v4.refine_a0_submission import OUTPUT as CANDIDATES, batch_store, environment
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r5_submission_chunks import S2, canonical
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import derive_adjacent_evidence
from src.medrag.evidence.refinement import expand_existing_context

OUTPUT = ROOT / 'outputs/v4/r5_s2_context_candidates_20261024'
_STORE = None
_POLICY = None
_S2 = None
_OUTPUT = None
_LOCK_SHA = None


def load_s2() -> dict[int, dict]:
    with zipfile.ZipFile(S2) as archive:
        return {int(row['id']): row for row in json.loads(archive.read('predictions.json'))}


def initialize(output: str, lock_sha: str) -> None:
    global _STORE, _POLICY, _S2, _OUTPUT, _LOCK_SHA
    _, _STORE, _POLICY, _ = environment()
    _S2 = load_s2()
    _OUTPUT, _LOCK_SHA = Path(output), lock_sha


def prepare_query(qid: int) -> dict:
    started = time.monotonic()
    final_path = CANDIDATES / 'final' / f'{qid}.json'
    final = read(final_path)
    candidates = {int(row['official_id']): row for row in final['trace']['ranked_candidates']}
    docs = [int(doc) for doc in _S2[qid]['relevant_docs']]
    if len(docs) != len(set(docs)) or any(doc not in candidates for doc in docs):
        raise ValueError(f'S2 documents are missing or duplicated in candidate pool: {qid}')
    selected = [candidates[doc] for doc in docs]
    cached = batch_store(_STORE, selected, set(docs))
    contexts = []
    for document_rank, doc in enumerate(docs, start=1):
        source = cached.source(doc)
        passages = cached.passages_for_parent(doc)
        unique = {}
        for evidence in candidates[doc]['evidence']:
            adjacent = derive_adjacent_evidence(evidence, source, passages, _POLICY)
            expanded = expand_existing_context(adjacent, source, passages)
            if expanded['text'] not in source.text:
                raise ValueError(f'Context is not a source substring: {qid}/{doc}')
            text_sha = hashlib.sha256(expanded['text'].encode()).hexdigest()
            value = unique.get(text_sha)
            seed = {'passage_id': evidence['passage_id'],
                    'baseline_rerank_score': float(evidence['rerank_score'])}
            if value is None:
                unique[text_sha] = {
                    'official_id': doc,
                    'document_rank': document_rank,
                    'context_id': hashlib.sha256(canonical([doc, expanded['text']])).hexdigest(),
                    'text': expanded['text'],
                    'text_sha256': text_sha,
                    'start': expanded['start'],
                    'end': expanded['end'],
                    'source_sha256': source.text_sha256,
                    'seed_passages': [seed],
                }
            else:
                value['seed_passages'].append(seed)
        for value in unique.values():
            value['seed_passages'].sort(
                key=lambda row: (-row['baseline_rerank_score'], row['passage_id']))
            contexts.append(value)
    if {row['official_id'] for row in contexts} != set(docs):
        raise ValueError(f'Every S2 document must have a context candidate: {qid}')
    result = {
        'format': 'v4-r5-context-candidates-query-v1',
        'id': qid,
        'query_sha256': final['query_sha256'],
        'candidate_final_sha256': sha256_file(final_path),
        'lock_sha256': _LOCK_SHA,
        'documents': len(docs),
        'contexts': contexts,
        'seconds': time.monotonic() - started,
    }
    path = _OUTPUT / 'queries' / f'{qid}.json'
    write_once(path, result)
    return {'id': qid, 'documents': len(docs), 'contexts': len(contexts),
            'bytes': path.stat().st_size, 'seconds': result['seconds'],
            'sha256': sha256_file(path)}


def run(output: Path, workers: int) -> None:
    if output.exists():
        raise FileExistsError(output)
    _, _, policy, _ = environment()
    if (policy.chunk_context != 'adjacent' or policy.context_window != 1
            or policy.context_max_characters != 4096):
        raise ValueError('R5 requires the frozen S2 adjacent-context policy')
    output.mkdir(parents=True, exist_ok=False)
    lock = {
        'format': 'v4-r5-context-candidates-lock-v1',
        'source_submission_sha256': sha256_file(S2),
        'candidate_manifest_sha256': sha256_file(CANDIDATES / 'run_manifest.json'),
        'context_policy': {'derive_from_registered_seed': True,
                           'selection_context_window': policy.context_window,
                           'selection_context_max_characters': policy.context_max_characters,
                           'final_extra_neighbors': 1,
                           'final_max_characters': 5500},
        'scope': 'Every evidence seed of every document exported by S2; duplicate final texts deduplicated per parent',
    }
    write_once(output / 'lock.json', lock)
    lock_sha = sha256_file(output / 'lock.json')
    totals = {'queries': 0, 'documents': 0, 'contexts': 0, 'bytes': 0,
              'worker_seconds': 0.0}
    hashes = {}
    context = mp.get_context('spawn')
    with context.Pool(workers, initializer=initialize,
                      initargs=(str(output), lock_sha)) as pool:
        for row in pool.imap(prepare_query, range(1, 1201), chunksize=1):
            totals['queries'] += 1
            for key in ('documents', 'contexts', 'bytes'):
                totals[key] += row[key]
            totals['worker_seconds'] += row['seconds']
            hashes[str(row['id'])] = row['sha256']
            if totals['queries'] % 25 == 0 or totals['queries'] == 1200:
                write_atomic_replace(output / 'status.json', {
                    **totals, 'status': 'COMPLETE' if totals['queries'] == 1200 else 'RUNNING'})
                print(json.dumps(totals), flush=True)
    manifest = {'format': 'v4-r5-context-candidates-manifest-v1', 'status': 'PASS',
                **totals, 'lock_sha256': lock_sha, 'query_sha256': hashes}
    write_once(output / 'manifest.json', manifest)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error('--workers must be positive')
    run(args.output, args.workers)

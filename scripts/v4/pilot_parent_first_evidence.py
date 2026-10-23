"""Bounded full-index parent-first pilot with branch traces and exact score reuse."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import time
import zipfile

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import src.medrag.core.pipeline as pipeline_module
from src.medrag.core.config import load_config
from src.medrag.core.disk_pipeline import DiskPipeline
from src.medrag.evidence.selector import select_results, SelectionPolicy
from src.medrag.ranking.score_cache import ScoreCache
from scripts.v4.smoke_full_index import score_contract
from scripts.v4.audit_evidence_survival import normalize, sha

QUERY_IDS = [53, 130, 380, 754, 771, 1144]
GPU_UUID = 'GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    temporary = path.with_name(f'.{path.name}.tmp-{os.getpid()}')
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def copy_sqlite(source, target):
    with sqlite3.connect(f'file:{source}?mode=ro', uri=True) as old:
        with sqlite3.connect(target) as new:
            old.backup(new)


class CapturedScores(ScoreCache):
    def score(self, query, rows, reranker):
        requested = []
        with sqlite3.connect(self.path) as db:
            for doc_id, passage_id, payload in rows:
                key = self._key(query, doc_id, passage_id, payload)
                prior = db.execute('SELECT score FROM scores WHERE cache_key=?', (key,)).fetchone()
                requested.append({'official_id': doc_id, 'passage_id': passage_id,
                                  'input_sha256': hashlib.sha256(payload.encode()).hexdigest(),
                                  'cache_hit': prior is not None, 'prior_score': prior[0] if prior else None})
        values = super().score(query, rows, reranker)
        self.last_pairs = [{**row, 'score': value} for row, value in zip(requested, values, strict=True)]
        return values


def captured_retrieve(pipeline, query):
    stage = {'branch_hits': {}, 'fallback_hits': [], 'fused': []}
    originals = (pipeline.doc_bm25.search, pipeline.passage_bm25.search,
                 pipeline.passage_bm25.search_parent, pipeline.dense.search,
                 pipeline_module.fuse)

    def wrap_branch(name, method):
        def call(text, limit, **kwargs):
            hits = method(text, limit, **kwargs)
            if not kwargs:
                stage['branch_hits'][name] = [asdict(hit) for hit in hits]
            return hits
        return call

    def parent_call(text, parent_id, limit):
        hits = originals[2](text, parent_id, limit)
        stage['fallback_hits'].append({'official_id': parent_id,
                                       'hits': [asdict(hit) for hit in hits]})
        return hits

    def fuse_call(rankings, *, k, limit):
        items = originals[4](rankings, k=k, limit=limit)
        stage['fused'] = [asdict(item) for item in items]
        return items

    pipeline.doc_bm25.search = wrap_branch('document_bm25', originals[0])
    pipeline.passage_bm25.search = wrap_branch('passage_bm25', originals[1])
    pipeline.passage_bm25.search_parent = parent_call
    pipeline.dense.search = wrap_branch('dense', originals[3])
    pipeline_module.fuse = fuse_call
    try:
        selected, diagnostics = pipeline.retrieve_with_diagnostics(query, 200, include_candidates=True)
    finally:
        pipeline.doc_bm25.search, pipeline.passage_bm25.search = originals[:2]
        pipeline.passage_bm25.search_parent, pipeline.dense.search = originals[2:4]
        pipeline_module.fuse = originals[4]
    return selected, diagnostics, stage


def positive_anchors(root, query_ids, queries):
    feedback = root / 'outputs/scored_feedback_recheck_20261009/JOINED_SCORED_CHUNKS.csv'
    archive = root / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'
    assert sha(archive) == '7eeacdd8273502fe5d9ba1dcb76d0689d9c3ba9c737749c4b39bc3c0e89a6e20'
    with zipfile.ZipFile(archive) as bundle:
        predictions = {row['id']: row for row in json.loads(bundle.read('predictions.json'))}
    result = {qid: [] for qid in query_ids}
    with feedback.open(encoding='utf-8') as stream:
        for row in csv.DictReader(stream):
            qid = int(row['query_id'])
            if qid not in result or row['scoring_status'] != 'scored' or row['relevant'] != 'True':
                continue
            chunk = predictions[qid]['relevant_chunks'][int(row['chunk_order'])]
            assert chunk['doc_id'] == int(row['baseline_doc_id_int'])
            assert hashlib.sha256(chunk['chunk_text'].encode()).hexdigest() == row['chunk_text_sha256']
            assert row['query_text'] == queries[qid]
            result[qid].append(chunk)
    return result


def evaluate(anchors, selected, diagnostics, stage, old_checkpoint):
    # This checks only preservation of old judged predictions, never official F2.
    by_parent = {d['official_id']: d for d in diagnostics['ranked_candidates']}
    selected_parents = {d['official_id']: d for d in selected}
    fused_parents = {item['official_id'] for item in stage['fused']}
    old_parents = {d['official_id'] for d in old_checkpoint['selected']}
    known = []
    for chunk in anchors:
        parent = chunk['doc_id']
        old_text = normalize(chunk['chunk_text'])
        candidates = by_parent.get(parent, {}).get('evidence', [])
        exported = selected_parents.get(parent, {}).get('evidence', [])
        known.append({'official_id': parent, 'in_fused': parent in fused_parents,
                      'was_selected_a0': parent in old_parents,
                      'selected_parent_first': parent in selected_parents,
                      'old_anchor_contained_in_candidate': any(old_text in normalize(e['text']) for e in candidates),
                      'old_anchor_contained_in_export': any(old_text in normalize(e['text']) for e in exported)})
    old_fused = {d['official_id']: d['score'] for d in old_checkpoint['trace']['ranked_candidates']}
    new_fused = {d['official_id']: d['score'] for d in diagnostics['ranked_candidates']}
    assert old_fused == new_fused, 'Retrieval/fused parent pool differs from locked A0 baseline'
    for branch, expected_count in [('document_bm25', 400), ('passage_bm25', 200), ('dense', 200)]:
        assert len(stage['branch_hits'][branch]) == expected_count
    assert len(stage['fused']) == old_checkpoint['trace']['fused_items']
    assert diagnostics['rerank_pairs'] == 200
    assert len({p['official_id'] for p in diagnostics['_pilot_pairs']}) == 200
    assert len(selected) == 200 and sum(len(d['evidence']) for d in selected) == 60
    return {'known_positives': known,
            'old_positive_parents_in_fused': sum(r['in_fused'] for r in known),
            'old_positive_parents_selected_a0': sum(r['was_selected_a0'] for r in known),
            'old_positive_parents_selected_parent_first': sum(r['selected_parent_first'] for r in known),
            'old_anchors_contained_in_parent_first_export': sum(r['old_anchor_contained_in_export'] for r in known)}


def main(root, index, config_path, reports):
    root, index, config_path, reports = (path.resolve() for path in (root, index, config_path, reports))
    smoke = root / 'outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550'
    assert read_json(smoke / 'full_index_acceptance.json')['status'] == 'FULL_INDEX_PASS'
    smoke_lock = read_json(smoke / 'smoke_lock.json')
    assert sha(index / 'manifest.json') == smoke_lock['index_manifest_sha256']
    for name, digest in smoke_lock['code_sha256'].items():
        assert sha(root / name) == digest, f'Protected runtime code changed: {name}'
    config = load_config(config_path)
    a0 = load_config(root / 'configs/v4_phase2_full_a0_gpu3.yaml')
    for key in ('corpus_version', 'registry', 'index_namespace', 'lexical', 'dense', 'reranker', 'fusion'):
        assert config[key] == a0[key], f'Pilot changed locked model/index contract: {key}'
    assert sha(root / 'configs/v4_phase2_full_a0_gpu3.yaml') == smoke_lock['config_sha256']
    assert config['retrieval'] == {**a0['retrieval'], 'rerank_allocation': 'parent_first'}
    assert config['selection'] == {'max_docs': 200, 'max_chunks': 60, 'chunks_per_doc': 1,
                                   'document_selection': 'independent'}
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == GPU_UUID
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').replace('-', '').lower() == GPU_UUID.removeprefix('GPU-').replace('-', '').lower()
    migration = read_json(root / 'configs/vibiomir/phase2_batch_migration_v1.json')
    query_path = root / migration['official_queries']
    assert sha(query_path) == smoke_lock['official_queries_sha256']
    rows = pq.read_table(query_path, columns=['id', 'query']).to_pylist()
    queries = {int(row['id']): str(row['query']) for row in rows}
    assert len(queries) == len(rows)
    for qid in QUERY_IDS:
        assert hashlib.sha256(queries[qid].encode()).hexdigest() == smoke_lock['query_sha256'][str(qid)]
    anchors = positive_anchors(root, QUERY_IDS, queries)
    lock = {'format': 'v4-parent-first-evidence-pilot-v1', 'query_ids': QUERY_IDS,
            'full_index_manifest_sha256': smoke_lock['index_manifest_sha256'],
            'a0_smoke_lock_sha256': sha(smoke / 'smoke_lock.json'),
            'pilot_config_sha256': sha(config_path),
            'pilot_script_sha256': sha(Path(__file__).resolve()),
            'baseline_submission_sha256': '7eeacdd8273502fe5d9ba1dcb76d0689d9c3ba9c737749c4b39bc3c0e89a6e20'}
    if (reports / 'run_lock.json').exists():
        assert read_json(reports / 'run_lock.json') == lock, 'Pilot resume lock differs'
    else:
        reports.mkdir(parents=True, exist_ok=True)
        write_json(reports / 'run_lock.json', lock)
    query_dir = reports / 'queries'
    query_dir.mkdir(exist_ok=True)
    copied_cache = reports / 'scores.sqlite'
    if not copied_cache.exists():
        copy_sqlite(smoke / 'smoke_scores.sqlite', copied_cache)
    cache = CapturedScores(copied_cache, score_contract(index / 'manifest.json', read_json(index / 'manifest.json'), config))
    pending = [qid for qid in QUERY_IDS if not (query_dir / f'{qid}.json').exists()]
    if pending:
        pipeline = DiskPipeline(index, config, score_cache=cache)
        assert pipeline.rerank_allocation == 'parent_first'
        for position, qid in enumerate(QUERY_IDS, 1):
            checkpoint_path = query_dir / f'{qid}.json'
            if checkpoint_path.exists():
                continue
            started = time.perf_counter()
            selected, diagnostics, stage = captured_retrieve(pipeline, queries[qid])
            elapsed = time.perf_counter() - started
            pairs = cache.last_pairs
            diagnostics['_pilot_pairs'] = pairs
            assert len(pairs) == 200
            previous = read_json(smoke / 'smoke_queries' / f'{qid}.json')
            checks = evaluate(anchors[qid], selected, diagnostics, stage, previous)
            diagnostics.pop('_pilot_pairs')
            write_json(checkpoint_path, {'id': qid, 'query_sha256': smoke_lock['query_sha256'][str(qid)],
                                         'status': 'complete', 'elapsed_seconds': elapsed,
                                         'selection': selected, 'diagnostics': diagnostics,
                                         'branch_trace': stage, 'pair_trace': pairs, 'checks': checks})
            print(f'pilot query {position}/{len(QUERY_IDS)} id={qid} complete '
                  f'cache_hits={sum(p["cache_hit"] for p in pairs)} '
                  f'new_scores={sum(not p["cache_hit"] for p in pairs)} '
                  f'elapsed={elapsed:.1f}s', flush=True)
    checkpoints = [read_json(query_dir / f'{qid}.json') for qid in QUERY_IDS]
    assert all(row['status'] == 'complete' for row in checkpoints)
    assert all(row['query_sha256'] == smoke_lock['query_sha256'][str(row['id'])] for row in checkpoints)
    for row in checkpoints:
        old = read_json(smoke / 'smoke_queries' / f'{row["id"]}.json')
        assert len(row['pair_trace']) == 200
        diagnostics = dict(row['diagnostics'])
        diagnostics['_pilot_pairs'] = row['pair_trace']
        assert evaluate(anchors[row['id']], row['selection'], diagnostics, row['branch_trace'], old) == row['checks']
        replay, _ = select_results(row['diagnostics']['ranked_candidates'],
                                   SelectionPolicy(document_selection='independent'),
                                   reranker_enabled=True, limit=200)
        assert replay == row['selection']
    result = {'status': 'PILOT_COMPLETE', 'query_ids': QUERY_IDS,
              'rerank_pairs_total': sum(len(row['pair_trace']) for row in checkpoints),
              'cached_pairs_reused': sum(p['cache_hit'] for row in checkpoints for p in row['pair_trace']),
              'new_pairs_scored': sum(not p['cache_hit'] for row in checkpoints for p in row['pair_trace']),
              'selected_documents': sum(len(row['selection']) for row in checkpoints),
              'selected_chunks': sum(sum(len(d['evidence']) for d in row['selection']) for row in checkpoints),
              'historical_positive_chunks': sum(len(anchors[qid]) for qid in QUERY_IDS),
              'old_positive_parents_selected_a0': sum(row['checks']['old_positive_parents_selected_a0'] for row in checkpoints),
              'old_positive_parents_selected_parent_first': sum(row['checks']['old_positive_parents_selected_parent_first'] for row in checkpoints),
              'old_anchors_contained_in_parent_first_export': sum(row['checks']['old_anchors_contained_in_parent_first_export'] for row in checkpoints),
              'total_seconds': sum(row['elapsed_seconds'] for row in checkpoints),
              'official_f2': 'UNMEASURED', 'score_cache_sha256': sha(copied_cache),
              'run_lock_sha256': sha(reports / 'run_lock.json')}
    write_json(reports / 'summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--index', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.index, args.config, args.reports)

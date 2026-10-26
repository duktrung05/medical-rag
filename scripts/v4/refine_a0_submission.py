"""Build and export cached A0 quality experiments with full provenance checks."""
from __future__ import annotations

import argparse
from collections import defaultdict
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import statistics
import time

from scripts.v4.run_full_baseline import _all_official_ids
from scripts.v4.run_p5_2_split import ROOT, CONFIG, read
from scripts.v4.score_a0_remaining_passages import BASE, OUTPUT as SCORES
from src.medrag.adapters.validation import (canonical_payload, package_zip, validate_payload,
                                            validate_selected, validate_zip)
from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.core.contracts import Passage
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.pair_lineage import write_atomic_replace, write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import apply_chunk_context
from src.medrag.evidence.refinement import (cap_documents_preserving_chunks,
                                            expand_existing_context, filter_documents,
                                            merge_supplemental_scores,
                                            select_documents_by_confidence)
from src.medrag.evidence.selector import SelectionPolicy, select_results
from src.medrag.evidence.verifier import verify

OUTPUT = ROOT / 'outputs/v4/a0_refined_candidates_20261024'
EXPORTS = ROOT / 'outputs/v4/a0_quality_submissions_20261024'
S1 = 's1_docs_fixed120'
S2 = 's2_docs_adaptive'


class CachedStore:
    def __init__(self, sources, passages, by_parent):
        self.sources, self.passages, self.by_parent = sources, passages, by_parent

    def source(self, parent):
        return self.sources[parent]

    def passage(self, key):
        return self.passages[key]

    def passages_for_parent(self, parent):
        return self.by_parent.get(parent, [])

    def for_parent(self, parent):
        return self.passages_for_parent(parent)


def batch_store(store, candidates, context_parents):
    sources = store.sources_batch({d['official_id'] for d in candidates})
    passage_ids = list({e['passage_id'] for d in candidates for e in d['evidence']})
    passages = store.passages_batch(passage_ids)
    by_parent = defaultdict(list)
    if context_parents:
        with sqlite3.connect(f'file:{store.path}?mode=ro', uri=True) as db:
            rows = db.execute('SELECT passage_id,official_id,url,text,source_start,source_end,'
                              'source_sha256,language,provenance FROM passages WHERE official_id IN ('
                              + ','.join('?' for _ in context_parents)
                              + ') ORDER BY official_id,source_start,source_end,passage_id',
                              sorted(context_parents)).fetchall()
        for row in rows:
            passage = Passage(row[0], store.corpus_version, *row[1:])
            by_parent[passage.official_id].append(passage)
            passages[passage.passage_id] = passage
    return CachedStore(sources, passages, by_parent)


def environment():
    config = load_config(CONFIG)
    index = ROOT / config['index_namespace']
    manifest = read(index / 'manifest.json')
    store = DiskCorpusStore(index / 'corpus.sqlite', manifest['corpus_version'], manifest['fingerprint'])
    policy = SelectionPolicy(**config['selection'])
    migration = read(ROOT / 'configs/vibiomir/phase2_batch_migration_v1.json')
    official_ids = _all_official_ids(ROOT / migration['official_corpus'])
    return config, store, policy, official_ids


def build(output, wait):
    output.mkdir(parents=True, exist_ok=True)
    config, store, policy, official_ids = environment()
    lock = {'format': 'a0-supplemental-replay-v1',
            'baseline_manifest_sha256': sha256_file(BASE / 'run_manifest.json'),
            'baseline_config_sha256': sha256_file(CONFIG),
            'implementation_sha256': {str(p.relative_to(ROOT)): sha256_file(p) for p in (
                Path(__file__).resolve(), ROOT / 'src/medrag/evidence/refinement.py')},
            'candidate_policy': 'score every cached passage; max score per parent; original d200/c60 selection',
            'expansion_policy': {'max_characters': 5500, 'extra_neighbors': 1,
                                 'preserve_existing_span': True}}
    write_once(output / 'lock.json', lock)
    lock_sha = sha256_file(output / 'lock.json')
    counts = {'queries': 0, 'new_pairs': 0, 'total_pairs': 0}
    hashes = {}
    for qid in range(1, 1201):
        score_path = SCORES / 'scores' / f'{qid}.json'
        waited = 0
        while not score_path.exists():
            if not wait or waited >= 3600:
                raise RuntimeError(f'Supplemental scores are not ready for query {qid}')
            if waited % 30 == 0:
                print(f'Waiting for supplemental score query {qid}', flush=True)
            time.sleep(5)
            waited += 5
        score = read(score_path)
        original_path = BASE / 'final' / f'{qid}.json'
        original = read(original_path)
        score_sha = sha256_file(score_path)
        if (score['baseline_final_sha256'] != sha256_file(original_path)
                or score['query_sha256'] != original['query_sha256']
                or score['runtime_sha256'] != sha256_file(SCORES / 'runtime_lock.json')):
            raise ValueError('Supplemental score lineage differs')
        path = output / 'final' / f'{qid}.json'
        if path.exists():
            final = read(path)
            if final['lock_sha256'] != lock_sha or final['supplemental_scores_sha256'] != score_sha:
                raise ValueError('Refined checkpoint lineage differs on resume')
        else:
            candidates = merge_supplemental_scores(original['trace']['ranked_candidates'], score['scores'])
            selected, selection = select_results(candidates, policy, reranker_enabled=True, limit=200)
            context_parents = {d['official_id'] for d in selected if d['evidence']}
            context_parents.update(d['official_id'] for d in original['selected'] if d['evidence'])
            cached = batch_store(store, candidates, context_parents)
            for row in score['scores']:
                source, passage = cached.source(row['official_id']), cached.passage(row['passage_id'])
                payload = f'{source.title}\n{passage.text}' if source.title else passage.text
                if (not verify(passage, source) or passage.official_id != row['official_id']
                        or hashlib.sha256(payload.encode()).hexdigest() != row['input_sha256']):
                    raise ValueError('Supplemental score payload/source mismatch')
            selected, context = apply_chunk_context(selected, policy, cached.sources, cached)
            validate_selected([qid], {qid: selected}, cached, policy, result_limit=200,
                              reranker_enabled=True, official_ids=official_ids)
            expanded, original_expanded = {}, {}
            for documents, target in ((selected, expanded), (original['selected'], original_expanded)):
                for doc in documents:
                    for evidence in doc['evidence']:
                        value = expand_existing_context(evidence, cached.source(doc['official_id']),
                                                        cached.passages_for_parent(doc['official_id']))
                        if evidence['text'] not in value['text']:
                            raise ValueError('Context expansion removed old content')
                        target[str(doc['official_id'])] = value
            final = {'format': 'a0-supplemental-final-query-v1', 'id': qid,
                     'query_sha256': original['query_sha256'], 'lock_sha256': lock_sha,
                     'baseline_final_sha256': score['baseline_final_sha256'],
                     'supplemental_scores_sha256': score_sha, 'selected': selected,
                     'expanded_evidence': expanded, 'original_expanded_evidence': original_expanded,
                     'status': 'complete', 'trace': {**original['trace'], 'ranked_candidates': candidates,
                         'selection': selection, 'context': context,
                         'rerank_pairs': 200 + len(score['scores']), 'rerank_allocation': 'all_cached_passages'}}
            write_once(path, final)
        hashes[str(qid)] = sha256_file(path)
        counts['queries'] += 1
        counts['new_pairs'] += len(score['scores'])
        counts['total_pairs'] += 200 + len(score['scores'])
        if qid % 25 == 0:
            print(json.dumps(counts), flush=True)
            write_atomic_replace(output / 'status.json', counts)
    manifest = {'status': 'PASS', 'query_count': counts['queries'], 'pair_count': counts['total_pairs'],
                'new_pairs': counts['new_pairs'], 'lock_sha256': lock_sha, 'final_sha256': hashes,
                'official_f2': 'UNMEASURED'}
    write_once(output / 'run_manifest.json', manifest)
    write_atomic_replace(output / 'status.json', {**counts, 'status': 'COMPLETE'})
    print(json.dumps({k: v for k, v in manifest.items() if k != 'final_sha256'}), flush=True)


def export(source, destination, variant):
    destination.mkdir(parents=True, exist_ok=False)
    _, _, policy, official_ids = environment()
    manifest = read(source / 'run_manifest.json')
    if manifest['status'] != 'PASS' or manifest['query_count'] != 1200:
        raise ValueError('Full refined candidate coverage is required')
    predictions, per_query = [], []
    document_policy = None
    for qid in range(1, 1201):
        path = source / 'final' / f'{qid}.json'
        if sha256_file(path) != manifest['final_sha256'][str(qid)]:
            raise ValueError('Refined query changed since validation')
        final = read(path)
        original_path = BASE / 'final' / f'{qid}.json'
        if sha256_file(original_path) != final['baseline_final_sha256']:
            raise ValueError('Original query changed')
        original = read(original_path)
        use_original = variant in ('docs_gate_only', 'context_only', 'baseline_refined')
        documents = original['selected'] if use_original else final['selected']
        gate = -4.0 if variant in ('docs_gate_only', 'baseline_refined', 'rerank_refined', S1, S2) else None
        selected = copy.deepcopy(filter_documents(documents, gate))
        document_target = len(selected)
        if variant == S1:
            document_target = min(120, len(selected))
            selected = cap_documents_preserving_chunks(selected, 120)
            document_policy = {'mode': 'fixed_rank_prefix', 'max_documents': 120,
                               'pre_gate': -4.0, 'protect_chunk_parents': True}
        elif variant == S2:
            selected, document_target = select_documents_by_confidence(
                selected, threshold=-2.0, min_documents=80, max_documents=140)
            document_policy = {'mode': 'score_count_adaptive_rank_prefix',
                               'score_threshold': -2.0, 'min_documents': 80,
                               'max_documents': 140, 'pre_gate': -4.0,
                               'protect_chunk_parents': True}
        context_changed = 0
        if variant in ('context_only', 'baseline_refined', 'rerank_refined', S1, S2):
            expanded = final['original_expanded_evidence'] if use_original else final['expanded_evidence']
            for doc in selected:
                if doc['evidence']:
                    new = expanded[str(doc['official_id'])]
                    if doc['evidence'][0]['text'] not in new['text']:
                        raise ValueError('Export lost previously selected text')
                    context_changed += new['text'] != doc['evidence'][0]['text']
                    doc['evidence'] = [new]
        prediction = records([qid], {qid: selected})[0]
        base_prediction = records([qid], {qid: original['selected']})[0]
        if variant == 'docs_gate_only' and prediction['relevant_chunks'] != base_prediction['relevant_chunks']:
            raise ValueError('Document-only experiment changed chunks')
        if variant == 'context_only' and prediction['relevant_docs'] != base_prediction['relevant_docs']:
            raise ValueError('Context-only experiment changed documents')
        if variant in (S1, S2):
            reference = copy.deepcopy(filter_documents(final['selected'], -4.0))
            for doc in reference:
                if doc['evidence']:
                    doc['evidence'] = [final['expanded_evidence'][str(doc['official_id'])]]
            reference_prediction = records([qid], {qid: reference})[0]
            if prediction['relevant_chunks'] != reference_prediction['relevant_chunks']:
                raise ValueError('Document selection experiment changed chunks')
        validate_payload(canonical_payload([prediction]), query_ids={qid}, official_ids=official_ids,
                         policy=policy, result_limit=200)
        predictions.append(prediction)
        per_query.append({'id': qid, 'documents': len(prediction['relevant_docs']),
                          'document_target': document_target,
                          'chunks': len(prediction['relevant_chunks']), 'contexts_changed': context_changed})
    payload = canonical_payload(predictions)
    (destination / 'predictions.json').write_bytes(payload)
    zip_path = destination / 'submission.zip'
    package_zip(zip_path, payload)
    validation = validate_zip(zip_path, query_ids=set(range(1, 1201)), official_ids=official_ids,
                              policy=policy, result_limit=200, max_bytes=100_000_000)
    if validation['payload_sha256'] != hashlib.sha256(payload).hexdigest():
        raise ValueError('ZIP payload differs from generated predictions')
    document_counts = [row['documents'] for row in per_query]
    report = {'status': 'LOCAL_EXPORT_PASS', 'variant': variant, 'validation': validation,
              'source_manifest_sha256': sha256_file(source / 'run_manifest.json'),
              'documents': sum(r['documents'] for r in per_query),
              'documents_per_query': {'minimum': min(document_counts),
                                      'median': statistics.median(document_counts),
                                      'maximum': max(document_counts),
                                      'mean': sum(document_counts) / len(document_counts)},
              'chunks': sum(r['chunks'] for r in per_query),
              'contexts_changed': sum(r['contexts_changed'] for r in per_query),
              'document_threshold': gate, 'document_policy': document_policy,
              'reference_submission_sha256': sha256_file(EXPORTS / 'rerank_refined' / 'submission.zip')
                  if variant in (S1, S2) else None,
              'official_f2': 'UNMEASURED',
              'quality_claim': 'Experimental candidate; official score requires BTC evaluation'}
    write_once(destination / 'manifest.json', report)
    write_once(destination / 'per_query_counts.json', per_query)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('build', 'export'))
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--variant', choices=('docs_gate_only', 'context_only', 'baseline_refined',
                                              'rerank_only', 'rerank_refined', S1, S2))
    parser.add_argument('--destination', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.output, args.wait)
    else:
        if not args.variant or not args.destination:
            parser.error('export requires --variant and --destination')
        export(args.output, args.destination, args.variant)

"""Summarize the bounded parent-first pilot without model or scorer calls."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.medrag.evidence.selector import SelectionPolicy, select_results
from scripts.v4.audit_evidence_survival import normalize, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main(root, reports):
    root, reports = root.resolve(), reports.resolve()
    summary = read(reports / 'summary.json')
    assert summary['status'] == 'PILOT_COMPLETE'
    pilot_lock = read(reports / 'run_lock.json')
    assert pilot_lock['query_ids'] == summary['query_ids']
    smoke = root / 'outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550'
    assert sha(smoke / 'smoke_lock.json') == pilot_lock['a0_smoke_lock_sha256']
    with (root / 'outputs/v4/evidence_survival_audit_20261021_v2/historical_positive_stages.csv').open() as stream:
        historical = [row for row in csv.DictReader(stream)
                      if int(row['query_id']) in summary['query_ids']]
    assert len(historical) == summary['historical_positive_chunks']
    archive = root / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'
    assert sha(archive) == pilot_lock['baseline_submission_sha256']
    with zipfile.ZipFile(archive) as bundle:
        baseline = {row['id']: row for row in json.loads(bundle.read('predictions.json'))}
    by_query = {}
    contrasts = []
    for qid in summary['query_ids']:
        old = read(smoke / 'smoke_queries' / f'{qid}.json')
        current = read(reports / 'queries' / f'{qid}.json')
        old_independent, _ = select_results(old['trace']['ranked_candidates'],
                                            SelectionPolicy(document_selection='independent'),
                                            reranker_enabled=True, limit=200)
        arm_docs = {'a0_with_chunks': old['selected'],
                    'independent_on_a0_scores': old_independent,
                    'independent_parent_first': current['selection']}
        row = {'query_id': qid, 'pairs_cached': sum(p['cache_hit'] for p in current['pair_trace']),
               'pairs_new': sum(not p['cache_hit'] for p in current['pair_trace']),
               'candidate_parents': len(current['diagnostics']['ranked_candidates']),
               'elapsed_seconds': current['elapsed_seconds']}
        anchors = [r for r in historical if int(r['query_id']) == qid]
        for arm, docs in arm_docs.items():
            by_parent = {d['official_id']: d for d in docs}
            row[f'{arm}_documents'] = len(docs)
            row[f'{arm}_chunks'] = sum(len(d['evidence']) for d in docs)
            row[f'{arm}_old_positive_parents'] = sum(int(r['doc_id']) in by_parent for r in anchors)
            row[f'{arm}_old_anchor_contained'] = sum(any(
                normalize(r_text) in normalize(e['text'])
                for e in by_parent.get(int(r['doc_id']), {}).get('evidence', []))
                for r, r_text in [(r, baseline[qid]['relevant_chunks'][int(r['baseline_chunk_order'])]['chunk_text'])
                                  for r in anchors])
        by_query[qid] = row
        branch = current['branch_trace']['branch_hits']
        fused = {x['official_id'] for x in current['branch_trace']['fused']}
        for r in anchors:
            parent = int(r['doc_id'])
            arm_parents = {name: {d['official_id'] for d in docs} for name, docs in arm_docs.items()}
            contrasts.append({'query_id': qid, 'doc_id': parent,
                              'historical_parent_stage_a0': r['parent_stage'],
                              'document_bm25': any(x['official_id'] == parent for x in branch['document_bm25']),
                              'passage_bm25': any(x['official_id'] == parent for x in branch['passage_bm25']),
                              'dense': any(x['official_id'] == parent for x in branch['dense']),
                              'fused': parent in fused,
                              **{f'selected_{arm}': parent in ids for arm, ids in arm_parents.items()}})
    assert sum(row['pairs_new'] for row in by_query.values()) == summary['new_pairs_scored']
    assert sum(row['pairs_cached'] for row in by_query.values()) == summary['cached_pairs_reused']
    missing_fused = [r for r in contrasts if not r['fused']]
    analysis = {'status': 'PILOT_ANALYSIS_COMPLETE',
                'query_ids': summary['query_ids'],
                'per_query': list(by_query.values()),
                'historical_positive_parents': len(contrasts),
                'not_fused_branch_diagnosis': missing_fused,
                'arms': {arm: {'documents': sum(row[f'{arm}_documents'] for row in by_query.values()),
                               'chunks': sum(row[f'{arm}_chunks'] for row in by_query.values()),
                               'old_positive_parents': sum(row[f'{arm}_old_positive_parents'] for row in by_query.values()),
                               'old_anchor_contained': sum(row[f'{arm}_old_anchor_contained'] for row in by_query.values())}
                         for arm in ('a0_with_chunks', 'independent_on_a0_scores', 'independent_parent_first')},
                'cached_pairs_reused': summary['cached_pairs_reused'],
                'new_pairs_scored': summary['new_pairs_scored'],
                'official_f2': 'UNMEASURED'}
    path = reports / 'analysis.json'
    if path.exists():
        assert read(path) == analysis
    else:
        path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    contrast_path = reports / 'positive_parent_contrast.csv'
    if not contrast_path.exists() and contrasts:
        with contrast_path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(contrasts[0]))
            writer.writeheader()
            writer.writerows(contrasts)
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reports)

"""CPU-only follow-up analysis for the completed parent-first evidence pilot."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sqlite3
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.medrag.core.contracts import Hit
from src.medrag.retrieval.core import fuse
from scripts.v4.audit_evidence_survival import normalize, sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def branch_hits(trace):
    return [[Hit(**row) for row in trace['branch_hits'][name]]
            for name in ('document_bm25', 'passage_bm25', 'dense')]


def parent_order(items):
    docs = {}
    for item in items:
        row = docs.setdefault(item.official_id, {'document': 0.0, 'passage': 0.0})
        channel = 'document' if item.passage_id is None else 'passage'
        row[channel] = max(row[channel], item.score)
    return sorted(docs, key=lambda parent: (
        -(docs[parent]['document'] + docs[parent]['passage']), parent))


def main(root, reports):
    root, reports = root.resolve(), reports.resolve()
    summary = read(reports / 'summary.json')
    analysis = read(reports / 'analysis.json')
    lock = read(reports / 'run_lock.json')
    assert summary['status'] == 'PILOT_COMPLETE' and analysis['status'] == 'PILOT_ANALYSIS_COMPLETE'
    assert lock['query_ids'] == summary['query_ids']
    archive = root / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'
    assert sha(archive) == lock['baseline_submission_sha256']
    with zipfile.ZipFile(archive) as bundle:
        baseline = {row['id']: row for row in json.loads(bundle.read('predictions.json'))}
    evidence_csv = root / 'outputs/v4/evidence_survival_audit_20261021_v2/historical_positive_stages.csv'
    with evidence_csv.open() as stream:
        anchors = [row for row in csv.DictReader(stream) if int(row['query_id']) in summary['query_ids']]
    store = sqlite3.connect(f'file:{root / "outputs/v4/task4/full_index_gpu3_20261019/corpus.sqlite"}?mode=ro', uri=True)
    fusion_rows, context_rows = [], []
    for qid in summary['query_ids']:
        checkpoint = read(reports / 'queries' / f'{qid}.json')
        raw = branch_hits(checkpoint['branch_trace'])
        fused400 = fuse(raw, k=60, limit=400)
        fused800 = fuse(raw, k=60, limit=800)
        assert [item.__dict__ for item in fused400] == checkpoint['branch_trace']['fused']
        order400, order800 = parent_order(fused400), parent_order(fused800)
        runtime_order = [doc['official_id'] for doc in sorted(
            checkpoint['diagnostics']['ranked_candidates'],
            key=lambda doc: (-doc['score'], doc['official_id']))]
        assert order400 == runtime_order, 'Diagnostic parent order differs from runtime'
        for row in [x for x in anchors if int(x['query_id']) == qid]:
            parent = int(row['doc_id'])
            fusion_rows.append({'query_id': qid, 'doc_id': parent,
                                'in_fused400': parent in order400,
                                'parent_rank_fused400': order400.index(parent) + 1 if parent in order400 else '',
                                'in_fused800': parent in order800,
                                'parent_rank_fused800': order800.index(parent) + 1 if parent in order800 else '',
                                'in_top200_parents_fused400': parent in order400[:200],
                                'in_top200_parents_fused800': parent in order800[:200]})
        selected = {d['official_id']: d for d in checkpoint['selection']}
        for row in [x for x in anchors if int(x['query_id']) == qid]:
            parent = int(row['doc_id'])
            old_text = baseline[qid]['relevant_chunks'][int(row['baseline_chunk_order'])]['chunk_text']
            anchor = normalize(old_text)
            evidence = selected.get(parent, {}).get('evidence', [])
            seed_contains = any(anchor in normalize(e['text']) for e in evidence)
            expanded = []
            for item in evidence:
                spans = store.execute('SELECT passage_id,source_start,source_end FROM passages '
                                      'WHERE official_id=? ORDER BY source_start,source_end,passage_id',
                                      (parent,)).fetchall()
                index = next(i for i, span in enumerate(spans) if span[0] == item['passage_id'])
                adjacent = spans[max(0, index - 1):index + 2]
                start, end = min(x[1] for x in adjacent), max(x[2] for x in adjacent)
                source = store.execute('SELECT text FROM sources WHERE official_id=?', (parent,)).fetchone()[0]
                expanded.append((source[start:end], start, end))
            context_rows.append({'query_id': qid, 'doc_id': parent,
                                 'parent_selected': parent in selected,
                                 'seed_contains_old_anchor': seed_contains,
                                 'adjacent_context_contains_old_anchor': any(anchor in normalize(x[0]) for x in expanded),
                                 'maximum_adjacent_context_characters': max((x[2] - x[1] for x in expanded), default=0)})
    result = {'status': 'REMEDIATION_ANALYSIS_COMPLETE',
              'parent_score_formula': 'max(document_fused_score) + max(passage_fused_score)',
              'baseline_parent_order_matches_runtime_queries': len(summary['query_ids']),
              'fusion': {'positive_parents': len(fusion_rows),
                         'present_fused400': sum(r['in_fused400'] for r in fusion_rows),
                         'present_fused800': sum(r['in_fused800'] for r in fusion_rows),
                         'top200_parents_fused400': sum(r['in_top200_parents_fused400'] for r in fusion_rows),
                         'top200_parents_fused800': sum(r['in_top200_parents_fused800'] for r in fusion_rows),
                         'absent_fused400_recovered_fused800': [r for r in fusion_rows if not r['in_fused400'] and r['in_fused800']]},
              'context': {'positive_anchors': len(context_rows),
                          'selected_parents': sum(r['parent_selected'] for r in context_rows),
                          'seed_contains_old_anchor': sum(r['seed_contains_old_anchor'] for r in context_rows),
                          'adjacent_context_contains_old_anchor': sum(r['adjacent_context_contains_old_anchor'] for r in context_rows),
                          'newly_recovered_by_context': [r for r in context_rows if not r['seed_contains_old_anchor'] and r['adjacent_context_contains_old_anchor']]},
              'model_calls': 0, 'official_f2': 'UNMEASURED'}
    # Preserve the earlier diagnostic artifact, whose sum-of-passages aggregation
    # differed from runtime. Version 2 checks exact baseline parent-order parity.
    path = reports / 'remediation_analysis_v2.json'
    if path.exists():
        assert read(path) == result
    else:
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--reports', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.reports)

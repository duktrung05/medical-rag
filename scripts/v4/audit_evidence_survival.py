"""CPU-only diagnostics on immutable full-corpus smoke traces and partial feedback.

Historical matched predictions are anchors, NOT gold reference texts. Token LCS
here uses whitespace and divides by anchor length; it is NOT the official scorer.
No models, retrieval searches, index writes or production policy changes occur.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import sqlite3
import statistics
import sys
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.medrag.evidence.selector import SelectionPolicy, select_results


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def normalize(text):
    return ' '.join(html.unescape(unicodedata.normalize('NFKC', text)).lower().split())


def anchor_coverage(anchor, texts):
    """Best whitespace-token LCS / old prediction length, NOT gold overlap."""
    words = anchor.split()
    masks = defaultdict(int)
    for position, word in enumerate(words):
        masks[word] |= 1 << position
    best = 0
    for text in texts:
        state = 0
        for word in normalize(text).split():
            x = state | masks.get(word, 0)
            state = x & ~(x - ((state << 1) | 1))
        best = max(best, state.bit_count())
    return best / len(words) if words else 0.0


def write_csv(path, rows):
    if rows:
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def global_chunk_replay(documents):
    """Diagnostic arm only: keep d200 parents, spend 60 chunks by global score."""
    chosen, _ = select_results(documents, SelectionPolicy(document_selection='independent'),
                               reranker_enabled=True, limit=200)
    output = {d['official_id']: {**d, 'evidence': []} for d in chosen}
    pool = [(d, e) for d in documents if d['official_id'] in output
            for e in d['evidence'] if 'rerank_score' in e]
    pool.sort(key=lambda pair: (-pair[1]['rerank_score'], -pair[0]['score'],
                                pair[0]['official_id'], pair[1]['passage_id']))
    used_text, count = set(), 0
    for doc, evidence in pool:
        parent = doc['official_id']
        if count >= 60:
            break
        key = (parent, evidence['text'])
        if len(output[parent]['evidence']) >= 2 or key in used_text:
            continue
        output[parent]['evidence'].append(evidence)
        used_text.add(key)
        count += 1
    return list(output.values())


def main(root, output):
    root = root.resolve()
    smoke = root / 'outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550'
    feedback = root / 'outputs/scored_feedback_recheck_20261009'
    baseline = root / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'
    index = root / 'outputs/v4/task4/full_index_gpu3_20261019'
    lock = json.loads((smoke / 'smoke_lock.json').read_text())
    query_ids = lock['query_ids']
    protected = [root / name for name in lock['code_sha256']]
    inputs = protected + [Path(__file__).resolve(), smoke / 'smoke_lock.json', smoke / 'smoke_scores.sqlite',
                          index / 'manifest.json', baseline,
                          feedback / 'JOINED_SCORED_CHUNKS.csv',
                          feedback / 'PRIORITY_61_QUERY_AUDIT.csv']
    inputs += [smoke / 'smoke_queries' / f'{qid}.json' for qid in query_ids]
    before = {str(p.relative_to(root)): sha(p) for p in inputs}
    assert all(sha(root / name) == expected for name, expected in lock['code_sha256'].items())
    assert sha(index / 'manifest.json') == lock['index_manifest_sha256']
    assert sha(baseline) == '7eeacdd8273502fe5d9ba1dcb76d0689d9c3ba9c737749c4b39bc3c0e89a6e20'
    with zipfile.ZipFile(baseline) as archive:
        predictions = {r['id']: r for r in json.loads(archive.read('predictions.json'))}
    with (feedback / 'JOINED_SCORED_CHUNKS.csv').open() as stream:
        judged = list(csv.DictReader(stream))
    with (feedback / 'PRIORITY_61_QUERY_AUDIT.csv').open() as stream:
        priority = {int(row['query_id']) for row in csv.DictReader(stream)}
    anchors = defaultdict(list)
    for row in judged:
        qid = int(row['query_id'])
        if qid not in query_ids or row['scoring_status'] != 'scored' or row['relevant'] != 'True':
            continue
        chunk = predictions[qid]['relevant_chunks'][int(row['chunk_order'])]
        assert chunk['doc_id'] == int(row['baseline_doc_id_int'])
        assert hashlib.sha256(chunk['chunk_text'].encode()).hexdigest() == row['chunk_text_sha256']
        assert hashlib.sha256(row['query_text'].encode()).hexdigest() == lock['query_sha256'][str(qid)]
        anchors[qid].append((row, chunk))
    db = sqlite3.connect(f'file:{index / "corpus.sqlite"}?mode=ro', uri=True)
    cache = sqlite3.connect(f'file:{smoke / "smoke_scores.sqlite"}?mode=ro', uri=True)
    per_query, anchor_rows, replay_rows = [], [], []
    cache_checks = 0
    for qid in query_ids:
        checkpoint = json.loads((smoke / 'smoke_queries' / f'{qid}.json').read_text())
        trace = checkpoint['trace']
        docs = trace['ranked_candidates']
        assert checkpoint['status'] == 'complete'
        assert checkpoint['query_sha256'] == lock['query_sha256'][str(qid)]
        assert canonical_sha(docs) == checkpoint['checks']['ranked_candidates_sha256']
        assert canonical_sha(checkpoint['selected']) == checkpoint['checks']['selected_sha256']
        selected, _ = select_results(docs, SelectionPolicy(), reranker_enabled=True, limit=200)
        assert selected == checkpoint['selected']
        by_parent = {d['official_id']: d for d in docs}
        assert len(by_parent) == len(docs)
        scored_parents = {d['official_id'] for d in docs if any('rerank_score' in e for e in d['evidence'])}
        passage_parent_count = Counter(row[0] for row in db.execute(
            f'SELECT official_id FROM passages WHERE official_id IN ({",".join("?" for _ in docs)})',
            list(by_parent)))
        doc_only = [d for d in docs if set(d['ranks']) == {'document_bm25'}]
        for doc in docs:
            title, source = db.execute('SELECT title,text FROM sources WHERE official_id=?',
                                       (doc['official_id'],)).fetchone()
            for evidence in doc['evidence']:
                stored = db.execute('SELECT official_id,text,source_start,source_end FROM passages '
                                    'WHERE passage_id=?', (evidence['passage_id'],)).fetchone()
                assert stored == (doc['official_id'], evidence['text'], evidence['start'], evidence['end'])
                assert source[evidence['start']:evidence['end']] == evidence['text']
                if 'rerank_score' in evidence:
                    payload = f'{title}\n{evidence["text"]}' if title else evidence['text']
                    rows = cache.execute('SELECT score,input_sha256 FROM scores WHERE '
                                         'query_sha256=? AND official_id=? AND passage_id=?',
                                         (checkpoint['query_sha256'], doc['official_id'], evidence['passage_id'])).fetchall()
                    assert rows == [(evidence['rerank_score'], hashlib.sha256(payload.encode()).hexdigest())]
                    cache_checks += 1
        # Parent membership for a breadth-first first round is recoverable even
        # though exact pre-rerank evidence order and branch hit lists are absent.
        pre_ranked = sorted(docs, key=lambda d: (-d['score'], d['official_id']))
        breadth_parents = [d['official_id'] for d in pre_ranked if d['evidence']][:200]
        source_by_parent, passages_by_parent = {}, {}
        for _, chunk in anchors[qid]:
            parent = chunk['doc_id']
            source = db.execute('SELECT text FROM sources WHERE official_id=?', (parent,)).fetchone()
            source_by_parent[parent] = source[0] if source else ''
            passages_by_parent[parent] = [row[0] for row in db.execute(
                'SELECT text FROM passages WHERE official_id=?', (parent,))]
        arms = {}
        for name, policy in [('a0', SelectionPolicy()),
                             ('independent_d120_c1', SelectionPolicy(max_docs=120, document_selection='independent')),
                             ('independent_d200_c1', SelectionPolicy(document_selection='independent')),
                             ('independent_d200_c2', SelectionPolicy(chunks_per_doc=2, document_selection='independent')),
                             ('diagnostic_global_c2', None)]:
            if policy is None:
                chosen = global_chunk_replay(docs)
            else:
                chosen, _ = select_results(docs, policy, reranker_enabled=True, limit=200)
            arms[name] = {d['official_id']: d for d in chosen}
            assert sum(len(d['evidence']) for d in chosen) <= 60
            replay_rows.append({'query_id': qid, 'arm': name, 'documents': len(chosen),
                                'chunks': sum(len(d['evidence']) for d in chosen),
                                'parents_with_exported_chunks': sum(bool(d['evidence']) for d in chosen),
                                'known_positive_parents_selected': sum(c['doc_id'] in arms[name] for _, c in anchors[qid]),
                                'known_anchor_contained_in_output': sum(any(normalize(c['chunk_text']) in normalize(e['text'])
                                    for e in arms[name].get(c['doc_id'], {}).get('evidence', [])) for _, c in anchors[qid]),
                                'selected_sha256': canonical_sha(chosen)})
        assert set(arms['independent_d200_c1']) == set(arms['independent_d200_c2'])
        for row, chunk in anchors[qid]:
            parent = chunk['doc_id']
            old = normalize(chunk['chunk_text'])
            candidate_evidence = by_parent.get(parent, {}).get('evidence', [])
            scored_evidence = [e for e in candidate_evidence if 'rerank_score' in e]
            a0_evidence = arms['a0'].get(parent, {}).get('evidence', [])
            texts_by_stage = {'store_passages': passages_by_parent[parent],
                             'fused_candidates': [e['text'] for e in candidate_evidence],
                             'reranked': [e['text'] for e in scored_evidence],
                             'a0_export': [e['text'] for e in a0_evidence]}
            failure = ('parent_absent_store' if not source_by_parent[parent] else
                       'parent_absent_fused' if parent not in by_parent else
                       'parent_not_reranked' if parent not in scored_parents else
                       'parent_not_selected_a0' if parent not in arms['a0'] else 'parent_selected_a0')
            entry = {'query_id': qid, 'doc_id': parent, 'baseline_chunk_order': row['chunk_order'],
                     'baseline_best_overlap': row['best_overlap'], 'parent_stage': failure,
                     'fused_parent_rank': next((i for i, d in enumerate(pre_ranked, 1) if d['official_id'] == parent), ''),
                     'parent_in_breadth_first_200': parent in breadth_parents,
                     'parent_selected_d120': parent in arms['independent_d120_c1'],
                     'parent_selected_d200': parent in arms['independent_d200_c1'],
                     'old_anchor_contained_in_source': old in normalize(source_by_parent[parent])}
            for stage, texts in texts_by_stage.items():
                entry[f'{stage}_contains_old_anchor'] = any(old in normalize(t) for t in texts)
                entry[f'{stage}_whitespace_lcs_over_old_anchor'] = anchor_coverage(old, texts)
            # Fixed previous+seed+next context for diagnosis only. This is not
            # registered/scored evidence and cannot pass the existing validator.
            ordered_spans = db.execute('SELECT passage_id,source_start,source_end FROM passages '
                                       'WHERE official_id=? ORDER BY source_start,source_end,passage_id',
                                       (parent,)).fetchall()
            expanded_texts = []
            expanded_characters = []
            for evidence in a0_evidence:
                position = next(i for i, span in enumerate(ordered_spans) if span[0] == evidence['passage_id'])
                adjacent = ordered_spans[max(0, position - 1):position + 2]
                start, end = min(s[1] for s in adjacent), max(s[2] for s in adjacent)
                expanded_texts.append(source_by_parent[parent][start:end])
                expanded_characters.append(end - start)
            entry['unscored_adjacent_context_contains_old_anchor'] = any(old in normalize(t) for t in expanded_texts)
            entry['unscored_adjacent_context_whitespace_lcs_over_old_anchor'] = anchor_coverage(old, expanded_texts)
            entry['unscored_adjacent_context_chars'] = max(expanded_characters, default=0)
            global_texts = [e['text'] for e in arms['diagnostic_global_c2'].get(parent, {}).get('evidence', [])]
            entry['global_c2_contains_old_anchor'] = any(old in normalize(t) for t in global_texts)
            anchor_rows.append(entry)
        token_info = trace['reranker_tokens']
        per_query.append({'query_id': qid, 'historical_positive_chunks': len(anchors[qid]),
                          'historical_doc_correct_chunk_zero': qid in priority,
                          'fused_parents': len(docs), 'candidate_evidence': sum(len(d['evidence']) for d in docs),
                          'rerank_pairs': trace['rerank_pairs'], 'scored_parents': len(scored_parents),
                          'unscored_parents': len(docs) - len(scored_parents),
                          'document_only_parents': len(doc_only),
                          'document_only_with_one_fallback': sum(len(d['evidence']) == 1 for d in doc_only),
                          'document_only_with_more_store_passages': sum(passage_parent_count[d['official_id']] > 1 for d in doc_only),
                          'breadth_first_parent_count': len(breadth_parents),
                          'breadth_first_new_parent_scores_minimum': len(set(breadth_parents) - scored_parents),
                          'truncated_pairs': token_info['truncated_pairs'],
                          'elapsed_seconds': checkpoint['elapsed_seconds']})
    assert cache_checks == 4000
    assert cache.execute('SELECT count(*) FROM scores').fetchone()[0] == cache_checks
    after = {str(p.relative_to(root)): sha(p) for p in inputs}
    assert before == after, 'Input or production code changed during audit'
    counts = Counter(row['parent_stage'] for row in anchor_rows)
    arm_summary = {}
    for arm in sorted({row['arm'] for row in replay_rows}):
        rows = [row for row in replay_rows if row['arm'] == arm]
        fields = ['documents', 'chunks', 'parents_with_exported_chunks',
                  'known_positive_parents_selected', 'known_anchor_contained_in_output']
        arm_summary[arm] = {key: sum(row[key] for row in rows) for key in fields}
    summary = {'status': 'CPU_DIAGNOSTIC_COMPLETE', 'query_count': len(query_ids),
               'query_ids': query_ids, 'queries_with_historical_positive_chunks': len([q for q in query_ids if anchors[q]]),
               'queries_from_priority_61': sorted(set(query_ids) & priority),
               'historical_positive_chunks': len(anchor_rows), 'parent_stage_counts': dict(counts),
               'known_positive_parents_in_breadth_first200': sum(r['parent_in_breadth_first_200'] for r in anchor_rows),
               'parent_not_reranked_but_in_breadth_first200': sum(r['parent_stage'] == 'parent_not_reranked' and r['parent_in_breadth_first_200'] for r in anchor_rows),
               'stage_exact_old_anchor_containment': {stage: sum(r[f'{stage}_contains_old_anchor'] for r in anchor_rows)
                   for stage in ['store_passages', 'fused_candidates', 'reranked', 'a0_export']},
               'old_anchor_contained_in_source': sum(r['old_anchor_contained_in_source'] for r in anchor_rows),
               'unscored_adjacent_context_contains_old_anchor': sum(r['unscored_adjacent_context_contains_old_anchor'] for r in anchor_rows),
               'stage_whitespace_lcs_over_old_anchor_median': {stage: statistics.median(r[f'{stage}_whitespace_lcs_over_old_anchor'] for r in anchor_rows)
                   for stage in ['store_passages', 'fused_candidates', 'reranked', 'a0_export']},
               'totals': {key: sum(row[key] for row in per_query) for key in per_query[0]
                          if key not in {'query_id', 'historical_doc_correct_chunk_zero', 'elapsed_seconds'}},
               'replay_arms_same_a0_scores': arm_summary,
               'score_cache_pairs_verified': cache_checks, 'input_hashes_unchanged': True,
               'model_calls': 0, 'official_f2': 'UNMEASURED',
               'limitations': ['Old matched predictions are anchors, not gold; missing anchors are not proof of failed LCS.',
                               'Whitespace LCS / old prediction length is a preservation proxy, not official token LCS / gold length.',
                               'Parent counts/stage counts use positive chunk rows, not unique parents.',
                               'Smoke set is not a representative quality sample; no unjudged candidates treated as negatives.',
                               'Branch hit lists and original evidence ordering are absent. Parent-first membership is computed, not reranked; required new pair count is only a lower bound.',
                               'Selection replays use A0 document_order scores; they do not evaluate parent_first configurations.']}
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'per_query.csv', per_query)
    write_csv(output / 'historical_positive_stages.csv', anchor_rows)
    write_csv(output / 'selection_replays.csv', replay_rows)
    for name, value in [('summary.json', summary), ('input_hashes.json', before)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert anchor_coverage('a b a c', ['a a x c']) == .75
    assert anchor_coverage('a b c', ['c b a']) == 1 / 3
    main(args.root, args.output)

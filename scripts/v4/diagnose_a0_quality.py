"""Audit full-run candidate survival using explicitly incomplete diagnostic labels."""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import unicodedata
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.refinement import expand_existing_context, filter_documents

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'outputs/v4/a0_fast_a6000_final_20261023'
OUT = ROOT / 'outputs/v4/a0_quality_diagnosis_20261024'
THRESHOLDS = (-6, -4, -3, -2, -1, 0)
CAPS = (80, 100, 120, 150)


def read(path):
    return json.loads(path.read_text())


def normalize(text):
    return ' '.join(html.unescape(unicodedata.normalize('NFKC', text)).lower().split())


def partition(qid):
    from src.medrag.validation.evidence_experiment import historical_partition
    return historical_partition(qid)


def policies(documents):
    yield 'baseline', documents
    for threshold in THRESHOLDS:
        yield f'doc_threshold_{threshold}', filter_documents(documents, threshold)
    for cap in CAPS:
        # Preserve all chunk parents even if the source ordering ever changes.
        yield f'doc_cap_{cap}', [d for rank, d in enumerate(documents)
                                 if rank < cap or d['evidence']]


def load_labels():
    old_zip = ROOT / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'
    feedback = ROOT / 'outputs/scored_feedback_recheck_20261009/JOINED_SCORED_CHUNKS.csv'
    qrels = ROOT / 'configs/vibiomir/phase2_retrieval_qrels_v2_20261015.json'
    with zipfile.ZipFile(old_zip) as archive:
        predictions = {r['id']: r for r in json.loads(archive.read('predictions.json'))}
    labels = defaultdict(list)
    with feedback.open() as stream:
        for row in csv.DictReader(stream):
            if row['scoring_status'] != 'scored' or row['relevant'] != 'True':
                continue
            qid = int(row['query_id'])
            chunk = predictions[qid]['relevant_chunks'][int(row['chunk_order'])]
            if (chunk['doc_id'] != int(row['baseline_doc_id_int'])
                    or hashlib.sha256(chunk['chunk_text'].encode()).hexdigest()
                    != row['chunk_text_sha256']):
                raise ValueError('Historical positive feedback does not match frozen prediction')
            labels[qid].append({'label_set': 'historical_matched_chunks',
                                'doc_id': chunk['doc_id'], 'text': chunk['chunk_text'],
                                'query_text': row['query_text']})
    for row in read(qrels)['queries']:
        labels[int(row['query_id'])].append({'label_set': 'manual_107',
                                             'doc_id': row['doc_id'], 'text': row['evidence_text'],
                                             'query_text': row.get('query')})
    return labels, {str(p.relative_to(ROOT)): sha256_file(p) for p in (old_zip, feedback, qrels)}


def main(output, base=BASE):
    output.mkdir(parents=True, exist_ok=False)
    labels, inputs = load_labels()
    index = ROOT / 'outputs/v4/task4/full_index_gpu3_20261019'
    manifest = read(index / 'manifest.json')
    store = DiskCorpusStore(index / 'corpus.sqlite', manifest['corpus_version'], manifest['fingerprint'])
    inputs[str(base.relative_to(ROOT) / 'run_manifest.json')] = sha256_file(base / 'run_manifest.json')
    totals = Counter()
    stages = defaultdict(Counter)
    policy_counts = defaultdict(Counter)
    rows = []
    compact = []
    for qid in range(1, 1201):
        path = base / 'final' / f'{qid}.json'
        data = read(path)
        selected = data['selected']
        candidates = {d['official_id']: d for d in data['trace']['ranked_candidates']}
        chosen = {d['official_id']: d for d in selected}
        split = partition(qid)
        ids_by_policy = {name: {d['official_id'] for d in docs}
                         for name, docs in policies(selected)}
        baseline_chunks = {d['official_id'] for d in selected if d['evidence']}
        for name, ids in ids_by_policy.items():
            if not baseline_chunks <= ids:
                raise ValueError('Document gate removed an existing chunk parent')
            policy_counts[split, name]['queries'] += 1
            policy_counts[split, name]['documents'] += len(ids)
        compact.append({'id': qid, 'partition': split,
                        'documents': [{'id': d['official_id'], 'score': d['rerank_score'],
                                       'has_chunk': bool(d['evidence'])} for d in selected]})
        for label in labels[qid]:
            if (label['query_text'] is not None and hashlib.sha256(label['query_text'].encode()).hexdigest()
                    != data['query_sha256']):
                raise ValueError(f'Label/query identity mismatch: {qid}')
            parent = label['doc_id']
            needle = normalize(label['text'])
            try:
                source = store.source(parent)
            except KeyError:
                source = None
            candidate = candidates.get(parent)
            final = chosen.get(parent)
            chunks = final['evidence'] if final else []
            base_contains = any(needle in normalize(e['text']) for e in chunks)
            expanded_contains = base_contains
            if source and chunks:
                passages = store.passages_for_parent(parent)
                expanded = [expand_existing_context(e, source, passages) for e in chunks]
                if any(e['text'] not in n['text'] for e, n in zip(chunks, expanded)):
                    raise ValueError('Expansion lost original exported evidence')
                expanded_contains = any(needle in normalize(e['text']) for e in expanded)
            in_source = bool(source and needle in normalize(source.text))
            if source is None:
                failure = 'source_absent'
            elif candidate is None:
                failure = 'absent_from_fused_candidates'
            elif not final:
                failure = 'candidate_not_scored_or_selected'
            elif not chunks:
                failure = 'selected_document_but_no_chunk_slot'
            elif base_contains:
                failure = 'anchor_contained'
            elif expanded_contains:
                failure = 'adjacent_context_too_narrow'
            elif in_source:
                failure = 'other_passage_or_more_distant_context'
            else:
                failure = 'historical_anchor_not_exact_in_current_source'
            row = {'query_id': qid, 'partition': split, 'label_set': label['label_set'],
                   'doc_id': parent, 'source_present': source is not None,
                   'anchor_in_source': in_source, 'in_fused': candidate is not None,
                   'selected_document': final is not None, 'exported_chunk': bool(chunks),
                   'baseline_contains_anchor': base_contains,
                   'expanded_contains_anchor': expanded_contains, 'failure_stage': failure,
                   'document_rank': next((i + 1 for i, d in enumerate(selected)
                                          if d['official_id'] == parent), None)}
            rows.append(row)
            stages[label['label_set']][failure] += 1
            for name, ids in ids_by_policy.items():
                key = f"{label['label_set']}_positive_rows"
                policy_counts[split, name][key] += parent in ids
        totals['queries'] += 1
        totals['documents'] += len(selected)
        totals['chunks'] += sum(len(d['evidence']) for d in selected)
        if qid % 100 == 0:
            print(f'Audited {qid}/1200 queries', flush=True)
    def dump(name, value):
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    summary = {'format': 'a0-quality-diagnostic-v1', 'totals': dict(totals),
               'label_counts': dict(Counter(r['label_set'] for r in rows)),
               'excluded_manual_labels': '20 synthetic EN/ZH query IDs outside the 1200 official IDs; 87/107 manual labels apply',
               'failure_stages': {k: dict(v) for k, v in stages.items()},
               'context_policy': {'max_characters': 5500, 'extra_neighbors': 1,
                                  'preserve_existing_span': True},
               'inputs_sha256': inputs,
               'scope': 'Incomplete historical positives/manual labels; exact containment proxy, not official F2',
               'partition_note': 'Query-disjoint development/validation; historical labels were previously inspected, not a pristine holdout',
               'development_policies': {name: dict(counts) for (part, name), counts in policy_counts.items()
                                        if part == 'development'}}
    dump('summary.json', summary)
    dump('all_policy_counts.json', {f'{part}/{name}': dict(counts)
                                   for (part, name), counts in policy_counts.items()})
    with (output / 'positive_paths.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / 'compact_rankings.jsonl').open('w') as stream:
        for row in compact:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--base', type=Path, default=BASE)
    args = parser.parse_args()
    main(args.output, args.base)

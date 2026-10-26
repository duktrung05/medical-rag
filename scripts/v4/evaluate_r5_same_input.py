"""Compare original BGE and R3 on identical exported S2 chunk inputs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.v4.diagnose_a0_quality import partition
from scripts.v4.refine_a0_submission import environment
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r5_submission_chunks import S2
from scripts.v4.train_r3_chunk_reranker import metrics
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import judgment_key

ORIGINAL = ROOT / 'outputs/v4/r5_s2_same_input_original_20261024'
TRAINED = ROOT / 'outputs/v4/r5_s2_same_input_r3_20261024'
OUTPUT = ROOT / 'outputs/v4/r5_same_input_evaluation_20261024'
FEEDBACK = ROOT / 'outputs/scored_feedback_recheck_20261009/JOINED_SCORED_CHUNKS.csv'
OLD_ZIP = ROOT / 'outputs/submissions/vibiomir_threshold_neg2_20261008/submission.zip'


def submission(path: Path) -> dict[int, dict]:
    with zipfile.ZipFile(path) as archive:
        return {int(row['id']): row for row in json.loads(archive.read('predictions.json'))}


def feedback_labels() -> dict[tuple[int, int, str], bool]:
    old = submission(OLD_ZIP)
    result = {}
    with FEEDBACK.open() as stream:
        for raw in csv.DictReader(stream):
            if raw['scoring_status'] != 'scored':
                continue
            qid, order = int(raw['query_id']), int(raw['chunk_order'])
            chunk = old[qid]['relevant_chunks'][order]
            doc = int(raw['baseline_doc_id_int'])
            if (chunk['doc_id'] != doc
                    or hashlib.sha256(chunk['chunk_text'].encode()).hexdigest()
                    != raw['chunk_text_sha256']):
                raise ValueError('Historical feedback differs from its frozen submission')
            if raw['relevant'] not in ('True', 'False'):
                raise ValueError('Scored feedback must have an explicit boolean label')
            key, label = judgment_key(qid, doc, chunk['chunk_text']), raw['relevant'] == 'True'
            if key in result and result[key] != label:
                raise ValueError(f'Conflicting historical feedback label: {key}')
            result[key] = label
    return result


def measured(rows: list[dict], score_field: str) -> dict:
    return metrics([row['label'] for row in rows], [row[score_field] for row in rows],
                   [row['query_id'] for row in rows])


def subset(rows: list[dict], predicate) -> dict:
    values = [row for row in rows if predicate(row)]
    labels = {row['label'] for row in values}
    result = {'examples': len(values), 'positives': sum(row['label'] for row in values),
              'queries': len({row['query_id'] for row in values})}
    if labels == {False, True}:
        result['original'] = measured(values, 'original_score')
        result['r3'] = measured(values, 'r3_score')
    return result


def evaluate(original: Path, trained: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    labels = feedback_labels()
    s2 = submission(S2)
    original_runtime, trained_runtime = read(original / 'runtime_lock.json'), read(trained / 'runtime_lock.json')
    if (original_runtime['source_submission_sha256'] != sha256_file(S2)
            or trained_runtime['source_submission_sha256'] != sha256_file(S2)
            or original_runtime['max_length'] != trained_runtime['max_length']):
        raise ValueError('Same-input runtime lineage differs')

    _, store, _, _ = environment()
    all_docs = {int(chunk['doc_id']) for row in s2.values() for chunk in row['relevant_chunks']}
    sources = store.sources_batch(all_docs)
    languages = {doc: sources[doc].metadata.get('detected_language', 'unknown')
                 for doc in all_docs}
    rows = []
    for qid in range(1, 1201):
        original_row = read(original / 'scores' / f'{qid}.json')
        trained_row = read(trained / 'scores' / f'{qid}.json')
        expected = s2[qid]['relevant_chunks']
        if len(original_row['scores']) != len(expected) or len(trained_row['scores']) != len(expected):
            raise ValueError(f'R5 query coverage differs: {qid}')
        for chunk, base, r3 in zip(expected, original_row['scores'], trained_row['scores'], strict=True):
            doc = int(chunk['doc_id'])
            text_sha = hashlib.sha256(chunk['chunk_text'].encode()).hexdigest()
            if (base['doc_id'] != doc or r3['doc_id'] != doc
                    or base['chunk_text_sha256'] != text_sha or r3['chunk_text_sha256'] != text_sha
                    or base['input_sha256'] != r3['input_sha256']
                    or base['input_tokens'] != r3['input_tokens']):
                raise ValueError(f'R5 same-input identity differs: {qid}/{doc}')
            key = judgment_key(qid, doc, chunk['chunk_text'])
            if key in labels:
                rows.append({'query_id': qid, 'partition': partition(qid), 'doc_id': doc,
                             'chunk_text_sha256': text_sha,
                             'language': languages[doc], 'label': labels[key],
                             'input_tokens': base['input_tokens'],
                             'original_score': base['score'], 'r3_score': r3['score']})

    groups = {
        'all': subset(rows, lambda _: True),
        'development': subset(rows, lambda row: row['partition'] == 'development'),
        'validation': subset(rows, lambda row: row['partition'] == 'validation'),
        'not_truncated': subset(rows, lambda row: row['input_tokens'] <= 512),
        'truncated': subset(rows, lambda row: row['input_tokens'] > 512),
    }
    for language in sorted({row['language'] for row in rows}):
        groups[f'language/{language}'] = subset(rows, lambda row, lang=language: row['language'] == lang)

    base = groups['validation'].get('original')
    r3 = groups['validation'].get('r3')
    gate_fields = ('roc_auc', 'average_precision', 'query_macro_average_precision', 'query_macro_mrr')
    sufficient = base is not None and r3 is not None
    gate = sufficient and all(r3[field] >= base[field] for field in gate_fields) and any(
        r3[field] > base[field] for field in gate_fields)
    report = {
        'format': 'v4-r5-same-input-evaluation-v2',
        'status': ('INSUFFICIENT_EXACT_IDENTITY_LABELS' if not sufficient else
                   'R3_SAME_INPUT_GATE_PASS' if gate else 'R3_SAME_INPUT_GATE_FAIL'),
        'gate': {'partition': 'validation', 'non_decrease_fields': list(gate_fields),
                 'requires_at_least_one_strict_gain': True},
        'groups': groups,
        'joined_feedback_rows': len(rows),
        'feedback_rows_without_exact_s2_identity': len(labels) - len(rows),
        'unjudged_s2_chunks': sum(len(row['relevant_chunks']) for row in s2.values()) - len(rows),
        'input_lineage': {
            's2_zip_sha256': sha256_file(S2),
            'old_feedback_zip_sha256': sha256_file(OLD_ZIP),
            'feedback_csv_sha256': sha256_file(FEEDBACK),
            'original_runtime_sha256': sha256_file(original / 'runtime_lock.json'),
            'r3_runtime_sha256': sha256_file(trained / 'runtime_lock.json'),
        },
        'scope': ('Historical judgments joined only by query, document and exact UTF-8 text hash; '
                  'changed contexts are unjudged. This is not independent evidence relevance or official F2.'),
    }
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'evaluation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original', type=Path, default=ORIGINAL)
    parser.add_argument('--trained', type=Path, default=TRAINED)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluate(args.original, args.trained, args.output)

"""Evaluate sliding-window max scoring on the exact S2 baseline contexts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.v4.diagnose_a0_quality import partition
from scripts.v4.evaluate_r5_same_input import feedback_labels
from scripts.v4.prepare_r5_context_candidates import OUTPUT as CANDIDATES
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r5_submission_chunks import S2
from scripts.v4.train_r3_chunk_reranker import metrics
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.identity import judgment_key

FIRST = ROOT / 'outputs/v4/r5_s2_same_input_r3_20261024'
WINDOW = (
    ROOT / 'outputs/v4/r5_context_scores_r3_a_20261024',
    ROOT / 'outputs/v4/r5_context_scores_r3_b_20261024',
)
OUTPUT = ROOT / 'outputs/v4/r5_window_scoring_evaluation_20261024'


def load_s2() -> dict[int, dict]:
    with zipfile.ZipFile(S2) as archive:
        return {int(row['id']): row for row in json.loads(archive.read('predictions.json'))}


def measured(rows: list[dict], field: str) -> dict:
    return metrics([row['label'] for row in rows], [row[field] for row in rows],
                   [row['query_id'] for row in rows])


def window_source(qid: int) -> Path:
    return WINDOW[0] if qid <= 600 else WINDOW[1]


def evaluate(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    refs, labels = load_s2(), feedback_labels()
    rows = []
    for qid in range(1, 1201):
        candidate_path = CANDIDATES / 'queries' / f'{qid}.json'
        candidates = read(candidate_path)['contexts']
        scored = read(window_source(qid) / 'scores' / f'{qid}.json')
        if scored['candidate_sha256'] != sha256_file(candidate_path):
            raise ValueError(f'Window score candidate lineage differs: {qid}')
        window_rows = {(row['official_id'], row['text_sha256']): score
                       for row, score in zip(candidates, scored['scores'], strict=True)}
        first_rows = {row['doc_id']: row for row in
                      read(FIRST / 'scores' / f'{qid}.json')['scores']}
        for chunk in refs[qid]['relevant_chunks']:
            doc = int(chunk['doc_id'])
            key = judgment_key(qid, doc, chunk['chunk_text'])
            if key not in labels:
                continue
            text_sha = hashlib.sha256(chunk['chunk_text'].encode()).hexdigest()
            value = {'query_id': qid, 'partition': partition(qid), 'doc_id': doc,
                     'label': labels[key], 'chunk_text_sha256': text_sha}
            first = first_rows[doc]
            window = window_rows.get((doc, text_sha))
            if window is None or first['chunk_text_sha256'] != text_sha:
                raise ValueError(f'Exact S2 context score is unavailable: {qid}/{doc}')
            value['r3_first'] = first['score']
            value['r3_window'] = window['score']
            value['r3_windows'] = window['window_count']
            rows.append(value)

    groups = {}
    for part in ('development', 'validation'):
        values = [row for row in rows if row['partition'] == part]
        groups[part] = {'examples': len(values), 'positives': sum(row['label'] for row in values),
                        'queries': len({row['query_id'] for row in values})}
        if {row['label'] for row in values} == {False, True}:
            groups[part]['r3'] = {'first_512': measured(values, 'r3_first'),
                                  'window_max': measured(values, 'r3_window')}
    fields = ('average_precision', 'query_macro_average_precision', 'query_macro_mrr')
    model_gate = {}
    for model in ('r3',):
        if model not in groups['validation']:
            model_gate[model] = False
            continue
        first = groups['validation'][model]['first_512']
        window = groups['validation'][model]['window_max']
        model_gate[model] = all(window[field] >= first[field] for field in fields)
    report = {
        'format': 'v4-r5-window-scoring-evaluation-v2',
        'status': ('INSUFFICIENT_EXACT_IDENTITY_LABELS' if 'r3' not in groups['validation'] else
                   'WINDOW_GATE_PASS' if any(model_gate.values()) else 'WINDOW_GATE_FAIL'),
        'model_gate': model_gate,
        'gate_fields': list(fields),
        'gate_partition': 'validation',
        'groups': groups,
        'joined_feedback_rows': len(rows),
        'window_count': {'r3': {'minimum': min((row['r3_windows'] for row in rows), default=None),
                                'maximum': max((row['r3_windows'] for row in rows), default=None),
                                'mean': sum(row['r3_windows'] for row in rows) / len(rows) if rows else None}},
        'input_lineage': {
            's2_zip_sha256': sha256_file(S2),
            'candidate_manifest_sha256': sha256_file(CANDIDATES / 'manifest.json'),
            'first_r3_runtime_sha256': sha256_file(FIRST / 'runtime_lock.json'),
            'window_r3_a_runtime_sha256': sha256_file(WINDOW[0] / 'runtime_lock.json'),
            'window_r3_b_runtime_sha256': sha256_file(WINDOW[1] / 'runtime_lock.json'),
        },
        'scope': 'Historical judged S2 chunks; window aggregation is evaluated on exact final text, not official F2.',
    }
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'evaluation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    evaluate(args.output)

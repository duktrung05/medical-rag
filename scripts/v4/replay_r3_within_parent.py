"""Conservative R3: retain all S2 chunk parents and rerank only their passages."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from scripts.v4.diagnose_a0_quality import load_labels, normalize, partition
from scripts.v4.refine_a0_submission import environment
from scripts.v4.replay_r3_chunk_selection import (CANDIDATES, SCORES, TRAINING, contexts,
    reference_predictions, scored_candidates)
from scripts.v4.run_p5_2_split import ROOT, read
from src.medrag.adapters.validation import canonical_payload, package_zip, validate_payload, validate_zip
from src.medrag.adapters.vibiomir_submission import records
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file

OUTPUT = ROOT / 'outputs/v4/r3_within_parent_selection_20261024'
SUBMISSION = ROOT / 'outputs/v4/r3_within_parent_submission_20261024'
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def choose_parent_passages(candidates: list[dict], parents: list[int], alpha: float) -> dict[int, dict]:
    by_parent = {doc['official_id']: doc for doc in candidates}
    selected = {}
    for parent in parents:
        evidence = by_parent[parent]['evidence']
        baseline_order = sorted(evidence, key=lambda e: (-e['rerank_score'], e['passage_id']))
        trained_order = sorted(evidence, key=lambda e: (-e['r3_score'], e['passage_id']))
        b_rank = {e['passage_id']: (len(evidence) - i) / len(evidence)
                  for i, e in enumerate(baseline_order)}
        r_rank = {e['passage_id']: (len(evidence) - i) / len(evidence)
                  for i, e in enumerate(trained_order)}
        selected[parent] = copy.deepcopy(max(evidence, key=lambda e: (
            (1 - alpha) * b_rank[e['passage_id']] + alpha * r_rank[e['passage_id']],
            e['r3_score'], e['rerank_score'], e['passage_id'])))
    return selected


def outcome(mapping, labels):
    result = Counter()
    for label in labels:
        group, parent = label['label_set'], label['doc_id']
        result[group + '/labels'] += 1
        result[group + '/chunk_parent'] += parent in mapping
        result[group + '/exact_anchor'] += (parent in mapping
            and normalize(label['text']) in normalize(mapping[parent]))
    return result


def evaluate(part, alphas, labels, refs, store, policy):
    counts = defaultdict(Counter)
    for qid in range(1, 1201):
        if partition(qid) != part or not labels[qid]:
            continue
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        _, candidates = scored_candidates(qid, final)
        reference = {row['doc_id']: row['chunk_text'] for row in refs[qid]['relevant_chunks']}
        parents = list(reference)
        counts[0.0].update(outcome(reference, labels[qid])); counts[0.0]['queries'] += 1
        for alpha in alphas:
            expanded = contexts(final, choose_parent_passages(candidates, parents, alpha), store, policy)
            mapping = {doc: evidence['text'] for doc, evidence in expanded.items()}
            if set(mapping) != set(reference):
                raise ValueError('Within-parent policy changed chunk parents')
            counts[alpha].update(outcome(mapping, labels[qid])); counts[alpha]['queries'] += 1
        if counts[0.0]['queries'] % 50 == 0:
            print(f'within-parent {part}: {counts[0.0]["queries"]} queries', flush=True)
    return {str(alpha): dict(value) for alpha, value in counts.items()}


def build(output: Path, destination: Path):
    output.mkdir(parents=True, exist_ok=False)
    labels, inputs = load_labels(); refs = reference_predictions()
    _, store, policy, official_ids = environment()
    development = evaluate('development', ALPHAS, labels, refs, store, policy)
    base = development['0.0']
    eligible = [a for a in ALPHAS
        if development[str(a)].get('historical_matched_chunks/exact_anchor', 0)
            > base.get('historical_matched_chunks/exact_anchor', 0)
        and development[str(a)].get('manual_107/exact_anchor', 0)
            >= base.get('manual_107/exact_anchor', 0)]
    alpha = max(eligible, key=lambda a: (development[str(a)].get(
        'historical_matched_chunks/exact_anchor', 0),
        development[str(a)].get('manual_107/exact_anchor', 0), -a)) if eligible else 0.0
    lock = {'format': 'v4-r3-within-parent-policy-v1', 'selected_alpha': alpha,
            'candidate_alphas': list(ALPHAS),
            'rule': 'maximize development historical exact anchors while manual exact anchors do not decrease; chunk parents are immutable',
            'development': development, 'label_inputs_sha256': inputs,
            'training_report_sha256': sha256_file(TRAINING / 'training_report.json'),
            'score_runtime_sha256': sha256_file(SCORES / 'runtime_lock.json')}
    write_once(output / 'policy_lock.json', lock)
    validation = evaluate('validation', (alpha,) if alpha else (), labels, refs, store, policy)
    vb, vc = validation['0.0'], validation.get(str(alpha), validation['0.0'])
    gate = (alpha > 0
            and vc.get('historical_matched_chunks/exact_anchor', 0)
                >= vb.get('historical_matched_chunks/exact_anchor', 0)
            and vc.get('manual_107/exact_anchor', 0) >= vb.get('manual_107/exact_anchor', 0)
            and vc.get('historical_matched_chunks/chunk_parent', 0)
                == vb.get('historical_matched_chunks/chunk_parent', 0)
            and vc.get('manual_107/chunk_parent', 0) == vb.get('manual_107/chunk_parent', 0))
    evaluation = {'format': 'v4-r3-within-parent-evaluation-v1',
                  'status': 'VALIDATION_GATE_PASS' if gate else 'VALIDATION_GATE_FAIL',
                  'selected_alpha': alpha, 'development': development,
                  'validation': validation,
                  'scope': 'Incomplete known-positive exact-containment proxy; not official F2'}
    write_once(output / 'evaluation.json', evaluation)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2), flush=True)
    if not gate:
        return
    destination.mkdir(parents=True, exist_ok=False)
    predictions = []; changed = 0
    for qid in range(1, 1201):
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        docs, candidates = scored_candidates(qid, final)
        parents = [row['doc_id'] for row in refs[qid]['relevant_chunks']]
        expanded = contexts(final, choose_parent_passages(candidates, parents, alpha), store, policy)
        selected = copy.deepcopy(docs)
        for doc in selected:
            doc['evidence'] = [expanded[doc['official_id']]] if doc['official_id'] in expanded else []
        prediction = records([qid], {qid: selected})[0]
        if (prediction['relevant_docs'] != refs[qid]['relevant_docs']
                or [x['doc_id'] for x in prediction['relevant_chunks']] != parents):
            raise ValueError('R3 within-parent export changed S2 documents or chunk parents')
        changed += sum(a['chunk_text'] != b['chunk_text'] for a, b in
                       zip(prediction['relevant_chunks'], refs[qid]['relevant_chunks'], strict=True))
        validate_payload(canonical_payload([prediction]), query_ids={qid}, official_ids=official_ids,
                         policy=policy, result_limit=200)
        predictions.append(prediction)
    payload = canonical_payload(predictions)
    (destination / 'predictions.json').write_bytes(payload)
    package_zip(destination / 'submission.zip', payload)
    zip_report = validate_zip(destination / 'submission.zip', query_ids=set(range(1, 1201)),
                              official_ids=official_ids, policy=policy,
                              result_limit=200, max_bytes=100_000_000)
    report = {'status': 'LOCAL_EXPORT_PASS', 'variant': 'r3_within_parent_plus_s2_docs',
              'selected_alpha': alpha, 'documents': sum(len(x['relevant_docs']) for x in predictions),
              'chunks': sum(len(x['relevant_chunks']) for x in predictions),
              'changed_chunk_texts_vs_s2': changed, 'validation': zip_report,
              'policy_lock_sha256': sha256_file(output / 'policy_lock.json'),
              'evaluation_sha256': sha256_file(output / 'evaluation.json'),
              'official_f2': 'UNMEASURED'}
    write_once(destination / 'manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--destination', type=Path, default=SUBMISSION)
    args = parser.parse_args()
    build(args.output, args.destination)

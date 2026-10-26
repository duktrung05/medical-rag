"""R4c high-confidence sparse replacement with development-locked thresholds."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from scripts.v4.diagnose_a0_quality import load_labels, partition
from scripts.v4.refine_a0_submission import OUTPUT as CANDIDATES, environment
from scripts.v4.replay_r3_chunk_selection import contexts, reference_predictions
from scripts.v4.replay_r4_rank_fusion import percentile
from scripts.v4.replay_r4_sparse_expansion import (EXPANSION, SCORES, document_list,
                                                    inputs, outcome)
from scripts.v4.run_p5_2_split import ROOT, read
from src.medrag.adapters.validation import canonical_payload, package_zip, validate_payload, validate_zip
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file

OUTPUT = ROOT / 'outputs/v4/r4_high_confidence_selection_20261024'
SUBMISSION = ROOT / 'outputs/v4/r4_high_confidence_submission_20261024'
POLICIES = tuple((score, gap) for score in (4.0, 5.0, 6.0, 7.0) for gap in (1.0, 2.0, 3.0))


def one_replacement(baseline, additions, minimum_score, minimum_gap):
    if not additions or not baseline:
        return []
    candidate = max(additions, key=lambda x: (x['r3_score'], -x['official_id']))
    old = list(baseline.items())
    r3 = percentile(old, lambda x: (x[1]['r3_score'], -x[0]))
    bge = percentile(old, lambda x: (x[1]['rerank_score'], -x[0]))
    removed, seed = min(old, key=lambda x: (0.5 * r3[id(x)] + 0.5 * bge[id(x)], x[0]))
    if (candidate['r3_score'] < minimum_score
            or candidate['r3_score'] < seed['r3_score'] + minimum_gap):
        return []
    return [(removed, candidate)]


def evaluate(part, policies, labels, refs, store, selection_policy):
    counts = defaultdict(Counter)
    for qid in range(1, 1201):
        if partition(qid) != part or not labels[qid]:
            continue
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        _, baseline, additions, _ = inputs(qid, final)
        ref_docs = refs[qid]['relevant_docs']
        ref_chunks = {x['doc_id']: x['chunk_text'] for x in refs[qid]['relevant_chunks']}
        counts['baseline'].update(outcome(ref_docs, ref_chunks, labels[qid])); counts['baseline']['queries'] += 1
        decisions = {p: one_replacement(baseline, additions, *p) for p in policies}
        union = {item['official_id']: item for value in decisions.values() for _, item in value}
        expanded = contexts(final, union, store, selection_policy) if union else {}
        for policy, accepted in decisions.items():
            chunks = dict(ref_chunks)
            for removed, item in accepted:
                chunks.pop(removed); chunks[item['official_id']] = expanded[item['official_id']]['text']
            docs = document_list(ref_docs, chunks, accepted)
            key = f'score{policy[0]}_gap{policy[1]}'
            counts[key].update(outcome(docs, chunks, labels[qid])); counts[key]['queries'] += 1
            counts[key]['replacements'] += len(accepted)
        if counts['baseline']['queries'] % 50 == 0:
            print(f'R4c {part}: {counts["baseline"]["queries"]} labeled queries', flush=True)
    return {key: dict(value) for key, value in counts.items()}


def guards(candidate, baseline, strict_gain):
    fields = [f'{group}/{metric}' for group in ('historical_matched_chunks', 'manual_107')
              for metric in ('document', 'chunk_parent', 'exact_anchor')]
    return (all(candidate.get(field, 0) >= baseline.get(field, 0) for field in fields)
            and (not strict_gain or candidate.get('historical_matched_chunks/exact_anchor', 0)
                 > baseline.get('historical_matched_chunks/exact_anchor', 0)))


def build(output: Path, destination: Path):
    output.mkdir(parents=True, exist_ok=False)
    labels, label_inputs = load_labels(); refs = reference_predictions()
    _, store, policy, official_ids = environment()
    development = evaluate('development', POLICIES, labels, refs, store, policy)
    eligible = [p for p in POLICIES if guards(
        development[f'score{p[0]}_gap{p[1]}'], development['baseline'], True)]
    selected = max(eligible, key=lambda p: (
        development[f'score{p[0]}_gap{p[1]}'].get('historical_matched_chunks/exact_anchor', 0),
        development[f'score{p[0]}_gap{p[1]}'].get('historical_matched_chunks/chunk_parent', 0),
        -development[f'score{p[0]}_gap{p[1]}'].get('replacements', 0))) if eligible else None
    lock = {'format': 'v4-r4-high-confidence-policy-v1', 'selected_policy': selected,
            'candidate_policies': list(POLICIES), 'maximum_replacements_per_query': 1,
            'rule': 'maximize development historical exact anchors subject to strict non-decrease in both label groups; then minimize replacements',
            'development': development,
            'expansion_manifest_sha256': sha256_file(EXPANSION / 'manifest.json'),
            'score_runtime_sha256': sha256_file(SCORES / 'runtime_lock.json'),
            'label_inputs_sha256': label_inputs}
    write_once(output / 'policy_lock.json', lock)
    validation = evaluate('validation', (selected,) if selected else (), labels, refs, store, policy)
    key = f'score{selected[0]}_gap{selected[1]}' if selected else 'baseline'
    gate = bool(selected and guards(validation[key], validation['baseline'], False))
    result = {'format': 'v4-r4-high-confidence-evaluation-v1',
              'status': 'VALIDATION_GATE_PASS' if gate else 'VALIDATION_GATE_FAIL',
              'selected_policy': selected, 'development': development, 'validation': validation,
              'scope': 'Incomplete known-positive exact-containment proxy; not official F2'}
    write_once(output / 'evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not gate:
        return
    score, gap = selected; destination.mkdir(parents=True, exist_ok=False)
    predictions = []; total = 0
    for qid in range(1, 1201):
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        _, baseline, additions, _ = inputs(qid, final)
        accepted = one_replacement(baseline, additions, score, gap)
        expanded = contexts(final, {x['official_id']: x for _, x in accepted}, store, policy) if accepted else {}
        chunks = {x['doc_id']: x['chunk_text'] for x in refs[qid]['relevant_chunks']}
        for removed, item in accepted:
            chunks.pop(removed); chunks[item['official_id']] = expanded[item['official_id']]['text']
        docs = document_list(refs[qid]['relevant_docs'], chunks, accepted)
        prediction = {'id': qid, 'relevant_docs': docs,
                      'relevant_chunks': [{'doc_id': doc, 'chunk_text': chunks[doc]}
                                          for doc in docs if doc in chunks]}
        validate_payload(canonical_payload([prediction]), query_ids={qid}, official_ids=official_ids,
                         policy=policy, result_limit=200)
        predictions.append(prediction); total += len(accepted)
    payload = canonical_payload(predictions); (destination / 'predictions.json').write_bytes(payload)
    package_zip(destination / 'submission.zip', payload)
    zip_report = validate_zip(destination / 'submission.zip', query_ids=set(range(1, 1201)),
                              official_ids=official_ids, policy=policy,
                              result_limit=200, max_bytes=100_000_000)
    report = {'status': 'LOCAL_EXPORT_PASS', 'variant': 'r4c_high_confidence_plus_s2',
              'selected_policy': selected, 'replacements': total,
              'documents': sum(len(x['relevant_docs']) for x in predictions),
              'chunks': sum(len(x['relevant_chunks']) for x in predictions),
              'validation': zip_report, 'policy_lock_sha256': sha256_file(output / 'policy_lock.json'),
              'evaluation_sha256': sha256_file(output / 'evaluation.json'), 'official_f2': 'UNMEASURED'}
    write_once(destination / 'manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--destination', type=Path, default=SUBMISSION)
    args = parser.parse_args(); build(args.output, args.destination)

"""Evaluate conservative R4 sparse additions and export only after validation passes."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
from pathlib import Path

from scripts.v4.diagnose_a0_quality import load_labels, normalize, partition
from scripts.v4.prepare_r4_sparse_expansion import OUTPUT as EXPANSION
from scripts.v4.refine_a0_submission import OUTPUT as CANDIDATES, environment
from scripts.v4.replay_r3_chunk_selection import contexts, reference_predictions, scored_candidates
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r4_sparse_expansion import OUTPUT as SCORES
from src.medrag.adapters.validation import canonical_payload, package_zip, validate_payload, validate_zip
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file

OUTPUT = ROOT / 'outputs/v4/r4_sparse_selection_20261024'
SUBMISSION = ROOT / 'outputs/v4/r4_sparse_submission_20261024'
POLICIES = tuple((cap, margin) for cap in (2, 5, 10, 20) for margin in (0.0, 0.5, 1.0))


def inputs(qid, final):
    docs, candidates = scored_candidates(qid, final)
    r3 = read(ROOT / 'outputs/v4/r3_candidate_scores_20261024/scores' / f'{qid}.json')
    r3_values = {(x['official_id'], x['passage_id']): x['score'] for x in r3['scores']}
    raw = {(doc['official_id'], evidence['passage_id']): evidence
           for doc in final['trace']['ranked_candidates'] for evidence in doc['evidence']}
    baseline = {}
    for doc in docs:
        if not doc['evidence']:
            continue
        seed_id = doc['evidence'][0]['seed_passage_id']
        seed = copy.deepcopy(raw[(doc['official_id'], seed_id)])
        seed['r3_score'] = r3_values[(doc['official_id'], seed_id)]
        baseline[doc['official_id']] = seed
    expansion_path = EXPANSION / 'queries' / f'{qid}.json'
    score_path = SCORES / 'scores' / f'{qid}.json'
    expansion, score = read(expansion_path), read(score_path)
    if score['expansion_sha256'] != sha256_file(expansion_path):
        raise ValueError('R4 expansion score lineage differs')
    values = {(x['official_id'], x['passage_id']): x['score'] for x in score['scores']}
    fresh = []
    for row in expansion['hits']:
        item = copy.deepcopy(row); item['r3_score'] = values[(row['official_id'], row['passage_id'])]
        item['rerank_score'] = item['r3_score']; item['score_status'] = 'reranked'
        fresh.append(item)
    best = {}
    for item in fresh:
        parent = item['official_id']
        if parent in baseline:
            continue
        if parent not in best or (item['r3_score'], item['passage_id']) > (
                best[parent]['r3_score'], best[parent]['passage_id']):
            best[parent] = item
    additions = sorted(best.values(), key=lambda x: (-x['r3_score'], x['official_id'], x['passage_id']))
    removals = sorted(baseline, key=lambda p: (baseline[p]['r3_score'], p))
    return docs, baseline, additions, removals


def replacements(baseline, additions, removals, cap, margin):
    accepted = []
    for added, removed in zip(additions, removals):
        if len(accepted) >= cap or added['r3_score'] < baseline[removed]['r3_score'] + margin:
            break
        accepted.append((removed, added))
    return accepted


def document_list(reference_docs, chunk_parents, accepted):
    result = list(reference_docs)
    for _, item in accepted:
        if item['official_id'] not in result:
            result.append(item['official_id'])
    required = set(chunk_parents)
    while len(result) > len(reference_docs):
        index = next((i for i in range(len(result) - 1, -1, -1) if result[i] not in required), None)
        if index is None:
            raise ValueError('R4 cannot preserve document count and all chunk parents')
        result.pop(index)
    return result


def outcome(documents, chunks, labels):
    result = Counter(); docset = set(documents)
    for label in labels:
        group, parent = label['label_set'], label['doc_id']
        result[group + '/labels'] += 1
        result[group + '/document'] += parent in docset
        result[group + '/chunk_parent'] += parent in chunks
        result[group + '/exact_anchor'] += (parent in chunks
            and normalize(label['text']) in normalize(chunks[parent]))
    return result


def evaluate(part, policies, labels, refs, store, selection_policy):
    counts = defaultdict(Counter)
    for qid in range(1, 1201):
        if partition(qid) != part or not labels[qid]:
            continue
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        _, baseline, additions, removals = inputs(qid, final)
        ref_docs = refs[qid]['relevant_docs']
        ref_chunks = {x['doc_id']: x['chunk_text'] for x in refs[qid]['relevant_chunks']}
        counts['baseline'].update(outcome(ref_docs, ref_chunks, labels[qid])); counts['baseline']['queries'] += 1
        union = {item['official_id']: item for item in additions[:20]}
        expanded = contexts(final, union, store, selection_policy) if union else {}
        for cap, margin in policies:
            accepted = replacements(baseline, additions, removals, cap, margin)
            chunks = dict(ref_chunks)
            for removed, item in accepted:
                chunks.pop(removed); chunks[item['official_id']] = expanded[item['official_id']]['text']
            docs = document_list(ref_docs, chunks, accepted)
            key = f'cap{cap}_margin{margin}'
            counts[key].update(outcome(docs, chunks, labels[qid])); counts[key]['queries'] += 1
            counts[key]['replacements'] += len(accepted)
        if counts['baseline']['queries'] % 50 == 0:
            print(f'R4 {part}: {counts["baseline"]["queries"]} labeled queries', flush=True)
    return {key: dict(value) for key, value in counts.items()}


def guards(candidate, baseline, *, strict_gain):
    groups = ('historical_matched_chunks', 'manual_107')
    metrics = ('document', 'chunk_parent', 'exact_anchor')
    return (all(candidate.get(f'{g}/{m}', 0) >= baseline.get(f'{g}/{m}', 0)
                for g in groups for m in metrics)
            and (not strict_gain or candidate.get('historical_matched_chunks/exact_anchor', 0)
                 > baseline.get('historical_matched_chunks/exact_anchor', 0)))


def build(output: Path, destination: Path):
    output.mkdir(parents=True, exist_ok=False)
    labels, label_inputs = load_labels(); refs = reference_predictions()
    _, store, policy, official_ids = environment()
    development = evaluate('development', POLICIES, labels, refs, store, policy)
    eligible = [(cap, margin) for cap, margin in POLICIES
                if guards(development[f'cap{cap}_margin{margin}'], development['baseline'], strict_gain=True)]
    selected = max(eligible, key=lambda x: (
        development[f'cap{x[0]}_margin{x[1]}'].get('historical_matched_chunks/exact_anchor', 0),
        development[f'cap{x[0]}_margin{x[1]}'].get('historical_matched_chunks/chunk_parent', 0),
        -development[f'cap{x[0]}_margin{x[1]}'].get('replacements', 0))) if eligible else None
    lock = {'format': 'v4-r4-sparse-selection-policy-v1', 'selected_policy': selected,
            'candidate_policies': list(POLICIES),
            'rule': 'maximize development historical exact anchors subject to non-decrease of document/chunk-parent/exact outcomes in historical and manual groups',
            'development': development,
            'expansion_manifest_sha256': sha256_file(EXPANSION / 'manifest.json'),
            'score_runtime_sha256': sha256_file(SCORES / 'runtime_lock.json'),
            'label_inputs_sha256': label_inputs}
    write_once(output / 'policy_lock.json', lock)
    validation = evaluate('validation', (selected,) if selected else (), labels, refs, store, policy)
    key = f'cap{selected[0]}_margin{selected[1]}' if selected else 'baseline'
    gate = bool(selected and guards(validation[key], validation['baseline'], strict_gain=False))
    result = {'format': 'v4-r4-sparse-selection-evaluation-v1',
              'status': 'VALIDATION_GATE_PASS' if gate else 'VALIDATION_GATE_FAIL',
              'selected_policy': selected, 'development': development, 'validation': validation,
              'scope': 'Incomplete known-positive exact-containment proxy; not official F2'}
    write_once(output / 'evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not gate:
        return
    cap, margin = selected; destination.mkdir(parents=True, exist_ok=False); predictions = []
    total_replacements = 0
    for qid in range(1, 1201):
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        _, baseline, additions, removals = inputs(qid, final)
        accepted = replacements(baseline, additions, removals, cap, margin)
        expanded = contexts(final, {x['official_id']: x for _, x in accepted}, store, policy) if accepted else {}
        chunks = {x['doc_id']: x['chunk_text'] for x in refs[qid]['relevant_chunks']}
        for removed, item in accepted:
            chunks.pop(removed); chunks[item['official_id']] = expanded[item['official_id']]['text']
        docs = document_list(refs[qid]['relevant_docs'], chunks, accepted)
        ordered_chunks = [{'doc_id': doc, 'chunk_text': chunks[doc]} for doc in docs if doc in chunks]
        prediction = {'id': qid, 'relevant_docs': docs, 'relevant_chunks': ordered_chunks}
        validate_payload(canonical_payload([prediction]), query_ids={qid}, official_ids=official_ids,
                         policy=policy, result_limit=200)
        predictions.append(prediction); total_replacements += len(accepted)
    payload = canonical_payload(predictions); (destination / 'predictions.json').write_bytes(payload)
    package_zip(destination / 'submission.zip', payload)
    zip_report = validate_zip(destination / 'submission.zip', query_ids=set(range(1, 1201)),
                              official_ids=official_ids, policy=policy,
                              result_limit=200, max_bytes=100_000_000)
    report = {'status': 'LOCAL_EXPORT_PASS', 'variant': 'r4_sparse_expansion_plus_s2',
              'selected_policy': selected, 'documents': sum(len(x['relevant_docs']) for x in predictions),
              'chunks': sum(len(x['relevant_chunks']) for x in predictions),
              'replacements': total_replacements, 'validation': zip_report,
              'policy_lock_sha256': sha256_file(output / 'policy_lock.json'),
              'evaluation_sha256': sha256_file(output / 'evaluation.json'),
              'official_f2': 'UNMEASURED'}
    write_once(destination / 'manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--destination', type=Path, default=SUBMISSION)
    args = parser.parse_args(); build(args.output, args.destination)

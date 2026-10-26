"""Tune R3/BGE rank blending, validate query-disjoint, and export with S2 docs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import zipfile

from scripts.v4.diagnose_a0_quality import load_labels, normalize, partition
from scripts.v4.refine_a0_submission import (EXPORTS, OUTPUT as CANDIDATES, batch_store,
                                             environment)
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r3_candidate_passages import OUTPUT as SCORES, s2_documents
from scripts.v4.train_r3_chunk_reranker import OUTPUT as TRAINING
from src.medrag.adapters.validation import (canonical_payload, package_zip,
                                            validate_payload, validate_zip)
from src.medrag.adapters.vibiomir_submission import records
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.evidence.context import derive_adjacent_evidence
from src.medrag.evidence.refinement import expand_existing_context

OUTPUT = ROOT / 'outputs/v4/r3_chunk_selection_20261024'
SUBMISSION = ROOT / 'outputs/v4/r3_submission_20261024'
ALPHAS = (0.25, 0.5, 0.75, 1.0)


def reference_predictions() -> dict[int, dict]:
    path = EXPORTS / 's2_docs_adaptive' / 'submission.zip'
    with zipfile.ZipFile(path) as archive:
        return {int(row['id']): row for row in json.loads(archive.read('predictions.json'))}


def scored_candidates(qid: int, final: dict) -> tuple[list[dict], dict]:
    docs = s2_documents(final)
    candidates = {doc['official_id']: copy.deepcopy(doc)
                  for doc in final['trace']['ranked_candidates']}
    score_path = SCORES / 'scores' / f'{qid}.json'
    score = read(score_path)
    if (score['candidate_final_sha256'] != sha256_file(CANDIDATES / 'final' / f'{qid}.json')
            or score['query_sha256'] != final['query_sha256']):
        raise ValueError(f'R3 score lineage differs: {qid}')
    values = {(row['official_id'], row['passage_id']): row['score'] for row in score['scores']}
    expected = {(doc['official_id'], evidence['passage_id']) for doc in docs
                for evidence in candidates[doc['official_id']]['evidence']}
    if set(values) != expected:
        raise ValueError(f'R3 score coverage differs from S2 passages: {qid}')
    result = []
    for selected in docs:
        candidate = candidates[selected['official_id']]
        for evidence in candidate['evidence']:
            evidence['r3_score'] = values[(candidate['official_id'], evidence['passage_id'])]
        result.append(candidate)
    return docs, result


def rank_values(rows: list[tuple[int, dict]], field: str) -> dict[tuple[int, str], float]:
    ordered = sorted(rows, key=lambda item: (-item[1][field], item[0], item[1]['passage_id']))
    size = len(ordered)
    return {(doc, evidence['passage_id']): (size - rank) / size
            for rank, (doc, evidence) in enumerate(ordered)}


def choose(rows: list[dict], alpha: float) -> dict[int, dict]:
    flat = [(doc['official_id'], evidence) for doc in rows for evidence in doc['evidence']]
    baseline = rank_values(flat, 'rerank_score')
    trained = rank_values(flat, 'r3_score')
    best = {}
    for doc, evidence in flat:
        key = (doc, evidence['passage_id'])
        value = (1 - alpha) * baseline[key] + alpha * trained[key]
        candidate = (value, evidence['r3_score'], evidence['rerank_score'], evidence['passage_id'])
        if doc not in best or candidate > best[doc][0]:
            best[doc] = (candidate, evidence)
    parents = sorted(best, key=lambda doc: (-best[doc][0][0], -best[doc][0][1], doc))[:60]
    return {doc: copy.deepcopy(best[doc][1]) for doc in parents}


def contexts(final: dict, selected_seeds: dict[int, dict], store, policy) -> dict[int, dict]:
    candidate_docs = [{'official_id': doc, 'evidence': [seed]} for doc, seed in selected_seeds.items()]
    cached = batch_store(store, candidate_docs, set(selected_seeds))
    expanded = {}
    for doc, seed in selected_seeds.items():
        source = cached.source(doc)
        updated = dict(seed)
        updated['rerank_score'] = float(seed['r3_score'])
        updated['score_status'] = 'reranked'
        adjacent = derive_adjacent_evidence(updated, source, cached.passages_for_parent(doc), policy)
        value = expand_existing_context(adjacent, source, cached.passages_for_parent(doc))
        if value['text'] not in source.text:
            raise ValueError('R3 context is not a registered source substring')
        expanded[doc] = value
    return expanded


def outcomes(mapping: dict[int, str], labels: list[dict]) -> Counter:
    result = Counter()
    for label in labels:
        group, doc = label['label_set'], label['doc_id']
        result[group + '/labels'] += 1
        result[group + '/chunk_parent'] += doc in mapping
        result[group + '/exact_anchor'] += (doc in mapping
            and normalize(label['text']) in normalize(mapping[doc]))
    return result


def evaluate(part: str, alphas: tuple[float, ...], labels, refs, store, policy):
    counts = defaultdict(Counter)
    for qid in range(1, 1201):
        if partition(qid) != part or not labels[qid]:
            continue
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        docs, candidates = scored_candidates(qid, final)
        if [d['official_id'] for d in docs] != refs[qid]['relevant_docs']:
            raise ValueError(f'S2 document replay differs: {qid}')
        reference = {row['doc_id']: row['chunk_text'] for row in refs[qid]['relevant_chunks']}
        if len(reference) != len(refs[qid]['relevant_chunks']):
            raise ValueError('S2 has more than one chunk for a parent')
        counts[0.0].update(outcomes(reference, labels[qid]))
        counts[0.0]['queries'] += 1; counts[0.0]['chunks'] += len(reference)
        for alpha in alphas:
            expanded = contexts(final, choose(candidates, alpha), store, policy)
            mapping = {doc: evidence['text'] for doc, evidence in expanded.items()}
            counts[alpha].update(outcomes(mapping, labels[qid]))
            counts[alpha]['queries'] += 1; counts[alpha]['chunks'] += len(mapping)
        if counts[0.0]['queries'] % 50 == 0:
            print(f'evaluated {part}: {counts[0.0]["queries"]} queries', flush=True)
    return {str(alpha): dict(value) for alpha, value in counts.items()}


def select_alpha(development: dict) -> float:
    base = development['0.0']
    eligible = []
    for alpha in ALPHAS:
        value = development[str(alpha)]
        if (value.get('historical_matched_chunks/exact_anchor', 0)
                > base.get('historical_matched_chunks/exact_anchor', 0)
                and value.get('manual_107/exact_anchor', 0)
                >= base.get('manual_107/exact_anchor', 0)
                and value.get('historical_matched_chunks/chunk_parent', 0)
                >= base.get('historical_matched_chunks/chunk_parent', 0)
                and value.get('manual_107/chunk_parent', 0)
                >= base.get('manual_107/chunk_parent', 0)):
            eligible.append(alpha)
    if not eligible:
        return 0.0
    return max(eligible, key=lambda alpha: (
        development[str(alpha)].get('historical_matched_chunks/exact_anchor', 0),
        development[str(alpha)].get('manual_107/exact_anchor', 0),
        development[str(alpha)].get('historical_matched_chunks/chunk_parent', 0), -alpha))


def validation_gate(alpha: float, validation: dict) -> bool:
    base, candidate = validation['0.0'], validation.get(str(alpha), validation['0.0'])
    return (alpha > 0
            and candidate.get('historical_matched_chunks/exact_anchor', 0)
                >= base.get('historical_matched_chunks/exact_anchor', 0)
            and candidate.get('manual_107/exact_anchor', 0)
                >= base.get('manual_107/exact_anchor', 0)
            and candidate.get('historical_matched_chunks/chunk_parent', 0)
                >= base.get('historical_matched_chunks/chunk_parent', 0)
            and candidate.get('manual_107/chunk_parent', 0)
                >= base.get('manual_107/chunk_parent', 0))


def export_submission(alpha, destination, refs, store, policy, official_ids, output):
    destination.mkdir(parents=True, exist_ok=False)
    predictions = []
    changed = 0
    for qid in range(1, 1201):
        final = read(CANDIDATES / 'final' / f'{qid}.json')
        docs, candidates = scored_candidates(qid, final)
        expanded = contexts(final, choose(candidates, alpha), store, policy)
        selected = copy.deepcopy(docs)
        for doc in selected:
            doc['evidence'] = [expanded[doc['official_id']]] if doc['official_id'] in expanded else []
        prediction = records([qid], {qid: selected})[0]
        if prediction['relevant_docs'] != refs[qid]['relevant_docs']:
            raise ValueError('R3 changed S2 documents')
        changed += sum(a != b for a, b in zip(prediction['relevant_chunks'],
                                               refs[qid]['relevant_chunks'], strict=True))
        validate_payload(canonical_payload([prediction]), query_ids={qid}, official_ids=official_ids,
                         policy=policy, result_limit=200)
        predictions.append(prediction)
    payload = canonical_payload(predictions)
    (destination / 'predictions.json').write_bytes(payload)
    package_zip(destination / 'submission.zip', payload)
    validation_zip = validate_zip(destination / 'submission.zip', query_ids=set(range(1, 1201)),
                                  official_ids=official_ids, policy=policy,
                                  result_limit=200, max_bytes=100_000_000)
    report = {'status': 'LOCAL_EXPORT_PASS', 'variant': 'r3_chunk_reranker_plus_s2_docs',
              'selected_alpha': alpha, 'documents': sum(len(x['relevant_docs']) for x in predictions),
              'chunks': sum(len(x['relevant_chunks']) for x in predictions),
              'changed_chunk_slots_vs_s2': changed, 'validation': validation_zip,
              'policy_lock_sha256': sha256_file(output / ('fallback_policy_lock.json'
                  if (output / 'fallback_policy_lock.json').exists() else 'policy_lock.json')),
              'evaluation_sha256': sha256_file(output / ('fallback_evaluation.json'
                  if (output / 'fallback_evaluation.json').exists() else 'evaluation.json')),
              'official_f2': 'UNMEASURED'}
    write_once(destination / 'manifest.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


def build(output: Path, destination: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    labels, label_inputs = load_labels()
    refs = reference_predictions()
    _, store, policy, official_ids = environment()
    development = evaluate('development', ALPHAS, labels, refs, store, policy)
    alpha = select_alpha(development)
    lock = {'format': 'v4-r3-chunk-blend-policy-v1', 'selected_alpha': alpha,
            'candidate_alphas': list(ALPHAS), 'selection_split': 'development',
            'rule': 'maximize historical exact anchors subject to strict non-decrease of historical/manual chunk-parent and manual exact-anchor counts',
            'development': development,
            'training_report_sha256': sha256_file(TRAINING / 'training_report.json'),
            'score_runtime_sha256': sha256_file(SCORES / 'runtime_lock.json'),
            's2_submission_sha256': sha256_file(EXPORTS / 's2_docs_adaptive' / 'submission.zip'),
            'label_inputs_sha256': label_inputs}
    write_once(output / 'policy_lock.json', lock)
    validation = evaluate('validation', (alpha,) if alpha else (), labels, refs, store, policy)
    gate = validation_gate(alpha, validation)
    evaluation = {'format': 'v4-r3-chunk-blend-evaluation-v1',
                  'status': 'VALIDATION_GATE_PASS' if gate else 'VALIDATION_GATE_FAIL',
                  'selected_alpha': alpha, 'development': development,
                  'validation': validation,
                  'scope': 'Incomplete known-positive exact-containment proxy; not official F2'}
    write_once(output / 'evaluation.json', evaluation)
    print(json.dumps(evaluation, ensure_ascii=False, indent=2), flush=True)
    if not gate:
        return
    export_submission(alpha, destination, refs, store, policy, official_ids, output)


def fallback(output: Path, destination: Path) -> None:
    first = read(output / 'evaluation.json')
    if first['status'] != 'VALIDATION_GATE_FAIL' or destination.exists():
        raise ValueError('R3 fallback requires a failed primary evaluation and no export')
    development = first['development']; base = development['0.0']
    eligible = [alpha for alpha in ALPHAS if alpha != first['selected_alpha']
        and development[str(alpha)].get('historical_matched_chunks/exact_anchor', 0)
            > base.get('historical_matched_chunks/exact_anchor', 0)
        and development[str(alpha)].get('manual_107/exact_anchor', 0)
            >= base.get('manual_107/exact_anchor', 0)
        and development[str(alpha)].get('historical_matched_chunks/chunk_parent', 0)
            >= base.get('historical_matched_chunks/chunk_parent', 0)
        and development[str(alpha)].get('manual_107/chunk_parent', 0)
            >= base.get('manual_107/chunk_parent', 0)]
    alpha = min(eligible) if eligible else 0.0
    lock = {'format': 'v4-r3-conservative-fallback-policy-v1', 'selected_alpha': alpha,
            'rule': 'smallest nonzero development alpha satisfying all original development guardrails after primary validation failure',
            'primary_evaluation_sha256': sha256_file(output / 'evaluation.json'),
            'validation_unseen_for_this_alpha_when_locked': True}
    write_once(output / 'fallback_policy_lock.json', lock)
    labels, _ = load_labels(); refs = reference_predictions()
    _, store, policy, official_ids = environment()
    validation = evaluate('validation', (alpha,) if alpha else (), labels, refs, store, policy)
    gate = validation_gate(alpha, validation)
    result = {'format': 'v4-r3-conservative-fallback-evaluation-v1',
              'status': 'VALIDATION_GATE_PASS' if gate else 'VALIDATION_GATE_FAIL',
              'selected_alpha': alpha, 'validation': validation,
              'scope': 'Incomplete known-positive exact-containment proxy; not official F2'}
    write_once(output / 'fallback_evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if gate:
        export_submission(alpha, destination, refs, store, policy, official_ids, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--destination', type=Path, default=SUBMISSION)
    parser.add_argument('--fallback', action='store_true')
    args = parser.parse_args()
    if args.fallback:
        fallback(args.output, args.destination)
    else:
        build(args.output, args.destination)

"""Evaluate exact-context BGE/R3 ablations inside the frozen S2 document set."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import zipfile

from scripts.v4.diagnose_a0_quality import load_labels, partition
from scripts.v4.prepare_r5_context_candidates import OUTPUT as CANDIDATES
from scripts.v4.refine_a0_submission import environment
from scripts.v4.replay_r4_sparse_expansion import outcome
from scripts.v4.run_p5_2_split import ROOT, read
from scripts.v4.score_r5_submission_chunks import S2
from src.medrag.adapters.validation import canonical_payload, package_zip, validate_payload, validate_zip
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file

TRAINED = (
    ROOT / 'outputs/v4/r5_context_scores_r3_first_a_20261024',
    ROOT / 'outputs/v4/r5_context_scores_r3_first_b_20261024',
)
SAME_INPUT = ROOT / 'outputs/v4/r5_same_input_evaluation_20261024/evaluation.json'
WINDOW_EVALUATION = ROOT / 'outputs/v4/r5_window_scoring_evaluation_20261024/evaluation.json'
OUTPUT = ROOT / 'outputs/v4/r5_context_selection_20261024'
SUBMISSION = ROOT / 'outputs/v4/r5_context_submission_20261024'

POLICIES = tuple(
    [{'mode': 'within_parent', 'model': 'r3'}]
    + [{'mode': 'parent_replace', 'model': 'r3', 'cap': cap, 'margin': margin}
       for cap in (1, 2) for margin in (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)]
)
MAX_REPLACEMENT_RATE = 0.10


def load_s2() -> dict[int, dict]:
    with zipfile.ZipFile(S2) as archive:
        return {int(row['id']): row for row in json.loads(archive.read('predictions.json'))}


def policy_name(policy: dict) -> str:
    return '_'.join(f'{key}-{value}' for key, value in policy.items())


def percentile(rows: list[dict], field: str) -> dict[str, float]:
    ordered = sorted(rows, key=lambda row: (-row[field], row['official_id'], row['context_id']))
    size = len(ordered)
    return {row['context_id']: (size - rank) / size for rank, row in enumerate(ordered)}


def inputs(qid: int, refs: dict[int, dict]) -> tuple[list[int], dict[int, dict], list[dict]]:
    candidate_path = CANDIDATES / 'queries' / f'{qid}.json'
    trained_path = (TRAINED[0] if qid <= 600 else TRAINED[1]) / 'scores' / f'{qid}.json'
    candidate, trained = read(candidate_path), read(trained_path)
    if (trained['candidate_sha256'] != sha256_file(candidate_path)
            or len(candidate['contexts']) != len(trained['scores'])):
        raise ValueError(f'R5 context score lineage differs: {qid}')
    contexts = []
    for row, r3 in zip(candidate['contexts'], trained['scores'], strict=True):
        if (row['official_id'] != r3['official_id']
                or row['context_id'] != r3['context_id']
                or row['text_sha256'] != r3['text_sha256']):
            raise ValueError(f'R5 exact context identity differs: {qid}')
        contexts.append({**row, 'r3_score': r3['score'], 'input_tokens': r3['input_tokens'],
                         'r3_windows': r3['window_count']})
    documents = [int(doc) for doc in refs[qid]['relevant_docs']]
    baseline = {}
    by_identity = {(row['official_id'], row['text']): row for row in contexts}
    for chunk in refs[qid]['relevant_chunks']:
        key = (int(chunk['doc_id']), chunk['chunk_text'])
        if key not in by_identity:
            raise ValueError(f'S2 final context is absent from R5 candidates: {qid}/{key[0]}')
        baseline[key[0]] = by_identity[key]
    if len(baseline) != 60 or not set(baseline) <= set(documents):
        raise ValueError(f'Invalid S2 baseline chunk mapping: {qid}')
    return documents, baseline, contexts


def decisions(baseline: dict[int, dict], contexts: list[dict], policy: dict) -> list[tuple[int, dict]]:
    grouped = defaultdict(list)
    for row in contexts:
        grouped[row['official_id']].append(row)
    mode = policy['mode']
    if mode == 'within_parent':
        field = policy['model'] + '_score'
        return [(doc, max(grouped[doc], key=lambda row: (row[field], row['context_id'])))
                for doc, old in baseline.items()
                if max(grouped[doc], key=lambda row: (row[field], row['context_id']))['context_id']
                != old['context_id']]

    best = {}
    if mode == 'parent_replace':
        field = policy['model'] + '_score'
        for doc, rows in grouped.items():
            best[doc] = max(rows, key=lambda row: (row[field], row['context_id']))
        additions = sorted((row for doc, row in best.items() if doc not in baseline),
                           key=lambda row: (-row[field], row['official_id']))
        removals = sorted(baseline.items(), key=lambda item: (item[1][field], item[0]))
        accepted = []
        for added, (removed, old) in zip(additions, removals):
            if (len(accepted) >= policy['cap']
                    or added[field] < old[field] + policy['margin']):
                break
            accepted.append((removed, added))
        return accepted

    if mode != 'consensus_replace':
        raise ValueError(f'Unsupported R5 selection mode: {mode}')
    for doc, rows in grouped.items():
        original_rank = percentile(rows, 'original_score')
        r3_rank = percentile(rows, 'r3_score')
        best[doc] = max(rows, key=lambda row: (
            original_rank[row['context_id']] + r3_rank[row['context_id']], row['context_id']))
    pool = [row for doc, row in best.items() if doc not in baseline] + list(baseline.values())
    original_rank, r3_rank = percentile(pool, 'original_score'), percentile(pool, 'r3_score')
    combined = lambda row: (original_rank[row['context_id']] + r3_rank[row['context_id']]) / 2
    additions = sorted((row for doc, row in best.items() if doc not in baseline),
                       key=lambda row: (-combined(row), row['official_id']))
    removals = sorted(baseline.items(), key=lambda item: (combined(item[1]), item[0]))
    accepted = []
    for added, (removed, old) in zip(additions, removals):
        if (len(accepted) >= policy['cap']
                or combined(added) < combined(old) + policy['percentile_margin']
                or added['original_score'] <= old['original_score']
                or added['r3_score'] <= old['r3_score']):
            break
        accepted.append((removed, added))
    return accepted


def apply(baseline: dict[int, dict], accepted: list[tuple[int, dict]]) -> dict[int, str]:
    chunks = {doc: row['text'] for doc, row in baseline.items()}
    for removed, added in accepted:
        chunks.pop(removed)
        chunks[added['official_id']] = added['text']
    if len(chunks) != 60:
        raise ValueError('R5 selection changed the frozen chunk budget')
    return chunks


def evaluate(part: str, policies: tuple[dict, ...], labels, refs) -> dict:
    counts = defaultdict(Counter)
    for qid in range(1, 1201):
        if partition(qid) != part or not labels[qid]:
            continue
        documents, baseline, contexts = inputs(qid, refs)
        baseline_chunks = {doc: row['text'] for doc, row in baseline.items()}
        counts['baseline'].update(outcome(documents, baseline_chunks, labels[qid]))
        counts['baseline']['queries'] += 1
        for policy in policies:
            accepted = decisions(baseline, contexts, policy)
            chunks = apply(baseline, accepted)
            key = policy_name(policy)
            counts[key].update(outcome(documents, chunks, labels[qid]))
            counts[key]['queries'] += 1
            counts[key]['replacements'] += len(accepted)
        if counts['baseline']['queries'] % 50 == 0:
            print(f'R5 {part}: {counts["baseline"]["queries"]} labeled queries', flush=True)
    return {key: dict(value) for key, value in counts.items()}


def nondecrease(candidate: dict, baseline: dict) -> bool:
    return all(candidate.get(f'{group}/{metric}', 0) >= baseline.get(f'{group}/{metric}', 0)
               for group in ('historical_matched_chunks', 'manual_107')
               for metric in ('chunk_parent', 'exact_anchor'))


def build(output: Path, destination: Path) -> None:
    if output.exists() or destination.exists():
        raise FileExistsError(output if output.exists() else destination)
    same_input = read(SAME_INPUT)
    if same_input['status'] != 'R3_SAME_INPUT_GATE_PASS':
        raise ValueError('R3 did not pass the frozen same-input gate')
    window_evaluation = read(WINDOW_EVALUATION)
    if window_evaluation['status'] != 'WINDOW_GATE_FAIL':
        raise ValueError('Unexpected full-window gate result; inspect before selection')
    labels, label_inputs = load_labels()
    refs = load_s2()
    development = evaluate('development', POLICIES, labels, refs)
    baseline = development['baseline']
    eligible = []
    for policy in POLICIES:
        value = development[policy_name(policy)]
        if (nondecrease(value, baseline)
                and value.get('historical_matched_chunks/exact_anchor', 0)
                >= baseline.get('historical_matched_chunks/exact_anchor', 0) + 3
                and value.get('replacements', 0)
                <= math.ceil(MAX_REPLACEMENT_RATE * value.get('queries', 0))):
            eligible.append(policy)
    selected = max(eligible, key=lambda policy: (
        development[policy_name(policy)].get('historical_matched_chunks/exact_anchor', 0),
        development[policy_name(policy)].get('manual_107/exact_anchor', 0),
        development[policy_name(policy)].get('historical_matched_chunks/chunk_parent', 0),
        -development[policy_name(policy)].get('replacements', 0))) if eligible else None
    lock = {
        'format': 'v4-r5-context-selection-lock-v1',
        'selected_policy': selected,
        'candidate_policies': list(POLICIES),
        'development_gate': ('no parent/exact loss in either label group, >=3 historical exact gains, '
                             'and replacements in <=10% of labeled queries'),
        'development': development,
        'same_input_evaluation_sha256': sha256_file(SAME_INPUT),
        'window_evaluation_sha256': sha256_file(WINDOW_EVALUATION),
        'score_mode': 'R3 first 512 tokens of exact context; full-window max failed validation AP gate',
        'candidate_manifest_sha256': sha256_file(CANDIDATES / 'manifest.json'),
        'r3_score_runtime_sha256': [sha256_file(path / 'runtime_lock.json') for path in TRAINED],
        'label_inputs_sha256': label_inputs,
    }
    output.mkdir(parents=True, exist_ok=False)
    write_once(output / 'policy_lock.json', lock)
    validation = evaluate('validation', (selected,) if selected else (), labels, refs)
    key = policy_name(selected) if selected else 'baseline'
    base, candidate = validation['baseline'], validation[key]
    gate = bool(selected and nondecrease(candidate, base)
                and candidate.get('historical_matched_chunks/exact_anchor', 0)
                >= base.get('historical_matched_chunks/exact_anchor', 0) + 2
                and candidate.get('replacements', 0)
                <= math.ceil(MAX_REPLACEMENT_RATE * candidate.get('queries', 0)))
    global_replacements = None
    if gate:
        global_replacements = 0
        for qid in range(1, 1201):
            _, baseline_chunks, contexts = inputs(qid, refs)
            global_replacements += len(decisions(baseline_chunks, contexts, selected))
        gate = global_replacements <= math.ceil(MAX_REPLACEMENT_RATE * 1200)
    result = {
        'format': 'v4-r5-context-selection-evaluation-v1',
        'status': 'VALIDATION_GATE_PASS' if gate else 'VALIDATION_GATE_FAIL',
        'selected_policy': selected,
        'development': development,
        'validation': validation,
        'validation_gate': ('no parent/exact loss in either label group, >=2 historical exact gains, '
                            'replacements in <=10% of labeled queries and <=120 globally'),
        'global_replacements': global_replacements,
        'scope': 'Incomplete known-positive proxy; document set is byte-for-byte S2 and official F2 is unmeasured',
    }
    write_once(output / 'evaluation.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if not gate:
        return

    _, _, selection_policy, official_ids = environment()
    predictions, total = [], 0
    for qid in range(1, 1201):
        documents, baseline_chunks, contexts = inputs(qid, refs)
        accepted = decisions(baseline_chunks, contexts, selected)
        chunks = apply(baseline_chunks, accepted)
        prediction = {'id': qid, 'relevant_docs': documents,
                      'relevant_chunks': [{'doc_id': doc, 'chunk_text': chunks[doc]}
                                          for doc in documents if doc in chunks]}
        if documents != refs[qid]['relevant_docs']:
            raise ValueError(f'R5 changed the frozen S2 document list: {qid}')
        validate_payload(canonical_payload([prediction]), query_ids={qid}, official_ids=official_ids,
                         policy=selection_policy, result_limit=200)
        predictions.append(prediction)
        total += len(accepted)
    if total != global_replacements:
        raise ValueError('R5 global replacement replay differs from validation gate')
    destination.mkdir(parents=True, exist_ok=False)
    payload = canonical_payload(predictions)
    (destination / 'predictions.json').write_bytes(payload)
    package_zip(destination / 'submission.zip', payload)
    validation_report = validate_zip(
        destination / 'submission.zip', query_ids=set(range(1, 1201)), official_ids=official_ids,
        policy=selection_policy, result_limit=200, max_bytes=100_000_000)
    manifest = {
        'status': 'LOCAL_EXPORT_PASS', 'variant': 'r5_exact_context_selection_plus_s2_docs',
        'selected_policy': selected, 'replacements': total,
        'documents': sum(len(row['relevant_docs']) for row in predictions),
        'chunks': sum(len(row['relevant_chunks']) for row in predictions),
        'validation': validation_report,
        'policy_lock_sha256': sha256_file(output / 'policy_lock.json'),
        'evaluation_sha256': sha256_file(output / 'evaluation.json'),
        'official_f2': 'UNMEASURED',
    }
    write_once(destination / 'manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--destination', type=Path, default=SUBMISSION)
    args = parser.parse_args()
    build(args.output, args.destination)

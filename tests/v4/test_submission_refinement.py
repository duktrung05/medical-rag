from copy import deepcopy

import pytest

from src.medrag.evidence.context import derive_adjacent_evidence
from src.medrag.evidence.refinement import (cap_documents_preserving_chunks,
                                            expand_existing_context, filter_documents,
                                            merge_supplemental_scores,
                                            select_documents_by_confidence)
from src.medrag.evidence.selector import SelectionPolicy
from tests.v4.test_adjacent_context import fixture
from scripts.v4.replay_r3_chunk_selection import choose, select_alpha
from scripts.v4.replay_r5_context_selection import decisions


def test_document_gate_preserves_chunk_parents_order_and_input():
    documents = [
        {'official_id': 1, 'rerank_score': 2.0, 'evidence': []},
        {'official_id': 2, 'rerank_score': -7.0, 'evidence': [{'text': 'keep'}]},
        {'official_id': 3, 'rerank_score': -2.0, 'evidence': []},
        {'official_id': 4, 'rerank_score': -3.0, 'evidence': []},
    ]
    before = deepcopy(documents)
    assert [d['official_id'] for d in filter_documents(documents, -2)] == [1, 2, 3]
    assert documents == before
    assert filter_documents(documents, None) == before
    with pytest.raises(ValueError):
        filter_documents(documents, float('nan'))
    with pytest.raises(ValueError):
        filter_documents([{'rerank_score': float('inf'), 'evidence': []}], -2)


def test_fixed_document_cap_preserves_chunk_parents_and_order():
    documents = [
        {'official_id': 1, 'rerank_score': 4.0, 'evidence': []},
        {'official_id': 2, 'rerank_score': 3.0, 'evidence': []},
        {'official_id': 3, 'rerank_score': 2.0, 'evidence': [{'text': 'protected'}]},
        {'official_id': 4, 'rerank_score': 1.0, 'evidence': []},
    ]
    before = deepcopy(documents)
    assert [d['official_id'] for d in cap_documents_preserving_chunks(documents, 2)] == [1, 2, 3]
    assert documents == before
    with pytest.raises(ValueError):
        cap_documents_preserving_chunks(documents, 0)


def test_adaptive_document_cap_uses_score_count_and_protects_chunks():
    documents = [
        {'official_id': 1, 'rerank_score': 4.0, 'evidence': []},
        {'official_id': 2, 'rerank_score': 2.0, 'evidence': []},
        {'official_id': 3, 'rerank_score': -1.0, 'evidence': []},
        {'official_id': 4, 'rerank_score': -3.0, 'evidence': []},
        {'official_id': 5, 'rerank_score': -5.0, 'evidence': [{'text': 'protected'}]},
    ]
    selected, target = select_documents_by_confidence(
        documents, threshold=0.0, min_documents=2, max_documents=4)
    assert target == 2
    assert [d['official_id'] for d in selected] == [1, 2, 5]
    selected, target = select_documents_by_confidence(
        documents, threshold=-4.0, min_documents=2, max_documents=3)
    assert target == 3
    assert [d['official_id'] for d in selected] == [1, 2, 3, 5]


@pytest.mark.parametrize('kwargs', [
    {'threshold': float('nan'), 'min_documents': 1, 'max_documents': 2},
    {'threshold': 0.0, 'min_documents': 0, 'max_documents': 2},
    {'threshold': 0.0, 'min_documents': 3, 'max_documents': 2},
])
def test_adaptive_document_cap_rejects_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        select_documents_by_confidence([], **kwargs)


def test_adaptive_document_cap_rejects_unsorted_scores():
    documents = [
        {'official_id': 1, 'rerank_score': 1.0, 'evidence': []},
        {'official_id': 2, 'rerank_score': 2.0, 'evidence': []},
    ]
    with pytest.raises(ValueError, match='descending'):
        select_documents_by_confidence(
            documents, threshold=0.0, min_documents=1, max_documents=2)


def test_r3_blend_changes_passage_and_parent_ranking():
    rows = [
        {'official_id': 1, 'evidence': [
            {'passage_id': 'a', 'rerank_score': 4.0, 'r3_score': -2.0},
            {'passage_id': 'b', 'rerank_score': 1.0, 'r3_score': 5.0}]},
        {'official_id': 2, 'evidence': [
            {'passage_id': 'c', 'rerank_score': 3.0, 'r3_score': 0.0}]},
    ]
    assert choose(rows, 0.25)[1]['passage_id'] == 'a'
    assert choose(rows, 1.0)[1]['passage_id'] == 'b'


def test_r3_alpha_requires_strict_development_gain_and_guardrails():
    base = {'historical_matched_chunks/exact_anchor': 10,
            'historical_matched_chunks/chunk_parent': 12,
            'manual_107/exact_anchor': 3, 'manual_107/chunk_parent': 4}
    development = {'0.0': base}
    for alpha in (0.25, 0.5, 0.75, 1.0):
        development[str(alpha)] = dict(base)
    development['0.25']['historical_matched_chunks/exact_anchor'] = 11
    development['0.5']['historical_matched_chunks/exact_anchor'] = 12
    development['0.5']['manual_107/exact_anchor'] = 2
    assert select_alpha(development) == 0.25
    development['0.25']['historical_matched_chunks/chunk_parent'] = 11
    assert select_alpha(development) == 0.0


def context_fixture():
    store, seed = fixture()
    original = derive_adjacent_evidence(seed, store.source(17), store.passages_for_parent(17),
                                       SelectionPolicy(chunk_context='adjacent', context_window=0))
    return store, original


def test_expansion_preserves_full_original_and_registered_source():
    store, original = context_fixture()
    before = deepcopy(original)
    expanded = expand_existing_context(original, store.source(17), store.passages_for_parent(17))
    assert expanded['text'] == store.source(17).text
    assert original['text'] in expanded['text']
    assert expanded['seed_evidence'] == original['seed_evidence']
    assert expanded['rerank_score'] == original['rerank_score']
    assert original == before
    assert expand_existing_context(original, store.source(17), store.passages_for_parent(17)) == expanded


def test_cap_cannot_trim_or_shift_existing_context():
    store, original = context_fixture()
    unchanged = expand_existing_context(original, store.source(17), store.passages_for_parent(17),
                                        max_characters=len(original['text']))
    assert unchanged == original
    expanded = expand_existing_context(original, store.source(17), store.passages_for_parent(17),
                                       max_characters=len('alpha evidence\nbeta treatment'))
    assert expanded['text'] == 'alpha evidence\nbeta treatment'
    assert original['text'] in expanded['text']
    with pytest.raises(ValueError, match='cannot remove'):
        expand_existing_context(original, store.source(17), store.passages_for_parent(17),
                                max_characters=len(original['text']) - 1)


@pytest.mark.parametrize('corruption', ['text', 'source_hash', 'context_ids'])
def test_expansion_rejects_corrupted_provenance(corruption):
    store, original = context_fixture()
    if corruption == 'text':
        original['text'] += ' invented'
    elif corruption == 'source_hash':
        original['source_sha256'] = 'wrong'
    else:
        original['context_passage_ids'] = ['missing']
    with pytest.raises(ValueError):
        expand_existing_context(original, store.source(17), store.passages_for_parent(17))


def candidate_fixture():
    return [
        {'official_id': 1, 'score': .1, 'rerank_score': 1., 'evidence': [
            {'passage_id': 'a', 'rerank_score': 1., 'score_status': 'reranked'},
            {'passage_id': 'b', 'score_status': 'not_reranked_budget'}]},
        {'official_id': 2, 'score': .2, 'evidence': [
            {'passage_id': 'c', 'score_status': 'not_reranked_budget'}]},
    ]


def test_supplemental_scores_rescue_unscored_parent_and_better_passage():
    candidates = candidate_fixture()
    before = deepcopy(candidates)
    result = merge_supplemental_scores(candidates, [
        {'official_id': 1, 'passage_id': 'b', 'score': 3.},
        {'official_id': 2, 'passage_id': 'c', 'score': 4.},
    ])
    assert candidates == before
    assert [d['official_id'] for d in result] == [2, 1]
    assert [e['passage_id'] for e in result[1]['evidence']] == ['b', 'a']
    assert result[1]['evidence'][1]['rerank_score'] == 1.
    assert result[1]['rerank_score'] == 3.
    assert all(e['score_status'] == 'reranked' for d in result for e in d['evidence'])


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'overwrite', 'nan'])
def test_supplemental_scores_reject_incomplete_or_corrupt_coverage(corruption):
    scores = [{'official_id': 1, 'passage_id': 'b', 'score': 3.},
              {'official_id': 2, 'passage_id': 'c', 'score': 4.}]
    if corruption == 'missing':
        scores.pop()
    elif corruption == 'duplicate':
        scores.append(dict(scores[0]))
    elif corruption == 'overwrite':
        scores.append({'official_id': 1, 'passage_id': 'a', 'score': 10.})
    else:
        scores[0]['score'] = float('nan')
    with pytest.raises(ValueError):
        merge_supplemental_scores(candidate_fixture(), scores)


def r5_context(doc, ident, original, r3):
    return {'official_id': doc, 'context_id': ident, 'text': ident,
            'original_score': original, 'r3_score': r3}


def test_r5_parent_replacement_compares_exact_context_scores_within_model():
    old = r5_context(1, 'old', 1.0, 1.0)
    weak = r5_context(2, 'weak', 1.9, 10.0)
    strong = r5_context(3, 'strong', 3.5, -1.0)
    policy = {'mode': 'parent_replace', 'model': 'original', 'cap': 1, 'margin': 2.0}
    assert decisions({1: old}, [old, weak, strong], policy) == [(1, strong)]
    policy['margin'] = 3.0
    assert decisions({1: old}, [old, weak, strong], policy) == []


def test_r5_consensus_requires_both_models_to_improve():
    old = r5_context(1, 'old', 1.0, 1.0)
    disagrees = r5_context(2, 'disagrees', 5.0, 0.0)
    agrees = r5_context(3, 'agrees', 4.0, 4.0)
    policy = {'mode': 'consensus_replace', 'cap': 1, 'percentile_margin': 0.05}
    assert decisions({1: old}, [old, disagrees, agrees], policy) == [(1, agrees)]

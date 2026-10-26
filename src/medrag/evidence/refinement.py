"""Refine completed submissions while preserving existing chunk parents and spans."""
from __future__ import annotations

import hashlib
import json
import math
from copy import deepcopy

from src.medrag.evidence.verifier import verify


def merge_supplemental_scores(candidates: list[dict], scores: list[dict]) -> list[dict]:
    """Fill every previously unscored pair and preserve all original model scores."""
    values = {}
    for row in scores:
        key = (row['official_id'], row['passage_id'])
        value = row['score']
        if key in values:
            raise ValueError('Duplicate supplemental score')
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Supplemental score must be finite')
        values[key] = value
    expected = {(d['official_id'], e['passage_id']) for d in candidates
                for e in d['evidence'] if 'rerank_score' not in e}
    if set(values) != expected:
        raise ValueError('Supplemental scores must cover exactly the unscored candidate pairs')
    result = deepcopy(candidates)
    for doc in result:
        for evidence in doc['evidence']:
            key = (doc['official_id'], evidence['passage_id'])
            if key in values:
                evidence['rerank_score'] = values[key]
                evidence['score_status'] = 'reranked'
        doc['evidence'].sort(key=lambda e: (-e['rerank_score'], e['passage_id']))
        if doc['evidence']:
            doc['rerank_score'] = max(e['rerank_score'] for e in doc['evidence'])
    result.sort(key=lambda d: (-d.get('rerank_score', float('-inf')), -d['score'], d['official_id']))
    return result


def filter_documents(documents: list[dict], threshold: float | None) -> list[dict]:
    """Apply a score gate only to documents without exported evidence.

    A document supporting an existing chunk is always retained, so this
    experiment changes document membership without changing chunk predictions.
    """
    if threshold is not None and not math.isfinite(threshold):
        raise ValueError('Document threshold must be finite')
    kept = []
    for doc in documents:
        score = doc.get('rerank_score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Document requires a finite reranker score')
        if threshold is None or doc['evidence'] or score >= threshold:
            kept.append(doc)
    return kept


def cap_documents_preserving_chunks(documents: list[dict], cap: int) -> list[dict]:
    """Keep a ranked document prefix without removing any chunk parent."""
    if type(cap) is not int or cap <= 0:
        raise ValueError('Document cap must be a positive integer')
    return [doc for rank, doc in enumerate(documents)
            if rank < cap or doc['evidence']]


def select_documents_by_confidence(documents: list[dict], *, threshold: float,
                                   min_documents: int, max_documents: int) -> tuple[list[dict], int]:
    """Choose a query-specific prefix from the reranker score distribution.

    The policy uses no query labels: it counts documents above ``threshold``,
    clips that count to the configured interval, and protects every parent of
    an already selected chunk.
    """
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) \
            or not math.isfinite(threshold):
        raise ValueError('Adaptive document threshold must be finite')
    if (type(min_documents) is not int or type(max_documents) is not int
            or min_documents <= 0 or max_documents < min_documents):
        raise ValueError('Adaptive document bounds are invalid')
    previous = float('inf')
    passing = 0
    for doc in documents:
        score = doc.get('rerank_score')
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError('Document requires a finite reranker score')
        if score > previous:
            raise ValueError('Adaptive document selection requires descending reranker scores')
        previous = score
        passing += score >= threshold
    target = max(min_documents, min(max_documents, passing))
    target = min(target, len(documents))
    return cap_documents_preserving_chunks(documents, target), target


def expand_existing_context(evidence: dict, source, passages: list, *,
                            max_characters: int = 5500, extra_neighbors: int = 1) -> dict:
    """Expand to registered adjacent boundaries, always retaining the old span.

    Expansion is independent of labels and scores. Each candidate contains all
    previously exported text; a character cap cannot shift the old span away.
    """
    if type(max_characters) is not int or max_characters <= 0:
        raise ValueError('Context cap must be a positive integer')
    if type(extra_neighbors) is not int or extra_neighbors < 0:
        raise ValueError('Extra neighbors must be a non-negative integer')
    start, end = evidence['start'], evidence['end']
    if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source.text)
            or source.text[start:end] != evidence['text']
            or evidence.get('source_sha256') != source.text_sha256):
        raise ValueError('Existing evidence differs from registered source')
    if end - start > max_characters:
        raise ValueError('Context cap cannot remove existing evidence')
    ordered = sorted(passages, key=lambda p: (p.source_start, p.source_end, p.passage_id))
    identities = [p.passage_id for p in ordered]
    if len(identities) != len(set(identities)):
        raise ValueError('Duplicate registered passage identity')
    old_ids = evidence['context_passage_ids']
    if not old_ids or len(old_ids) != len(set(old_ids)) or not set(old_ids) <= set(identities):
        raise ValueError('Existing context passages are missing or duplicated')
    positions = [identities.index(p) for p in old_ids]
    first, last = min(positions), max(positions)
    if positions != list(range(first, last + 1)):
        raise ValueError('Existing context passages must be contiguous')
    old_members = ordered[first:last + 1]
    if (min(p.source_start for p in old_members) != start
            or max(p.source_end for p in old_members) != end):
        raise ValueError('Existing context boundaries differ from registered passages')
    options = []
    for left in range(max(0, first - extra_neighbors), first + 1):
        for right in range(last, min(len(ordered), last + extra_neighbors + 1)):
            members = ordered[left:right + 1]
            if any(p.official_id != source.official_id or not verify(p, source) for p in members):
                raise ValueError('Context contains ungrounded or foreign passage')
            lo = min(p.source_start for p in members)
            hi = max(p.source_end for p in members)
            if lo <= start and hi >= end and hi - lo <= max_characters:
                options.append((hi - lo, -abs((start - lo) - (hi - end)), -lo, lo, hi, members))
    if not options:
        raise ValueError('No source-grounded span preserves the existing context')
    _, _, _, lo, hi, members = max(options, key=lambda row: row[:3])
    if (lo, hi) == (start, end):
        return dict(evidence)
    provenance = {'version': 'expanded_existing_source_span_v1',
                  'official_id': source.official_id, 'source_sha256': source.text_sha256,
                  'start': lo, 'end': hi, 'original_context_id': evidence['passage_id'],
                  'context_passage_ids': [p.passage_id for p in members],
                  'max_characters': max_characters, 'extra_neighbors': extra_neighbors}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True,
                                            separators=(',', ':')).encode()).hexdigest()
    return {**evidence, 'evidence_kind': provenance['version'],
            'passage_id': f'expanded_context:{fingerprint}',
            'context_passage_ids': provenance['context_passage_ids'],
            'start': lo, 'end': hi, 'text': source.text[lo:hi],
            'expansion_provenance': provenance}

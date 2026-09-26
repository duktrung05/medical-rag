"""Macro source-context/document recall and first-positive MRR from scored rankings."""

import math
import statistics


KS = (1, 3, 5, 10, 20, 50, 100)


def validate_ranking(ranking, allowed_ids):
    ids = [cid for cid, _ in ranking]
    if len(ids) != len(set(ids)) or any(cid not in allowed_ids for cid in ids):
        raise ValueError('Ranking contains duplicate or unknown IDs')
    if any(not math.isfinite(score) for _, score in ranking):
        raise ValueError('Ranking contains non-finite scores')


def summarize_rankings(samples, rows):
    if not samples or len(samples) != len(rows):
        raise ValueError('Metrics require all query rows')
    if [s['id'] for s in samples] != [r['id'] for r in rows]:
        raise ValueError('Ranking query order/IDs differ from samples')

    def summarize(pairs):
        recalls = {str(k): [] for k in KS}
        doc_recalls = {str(k): [] for k in KS}
        reciprocal_ranks = []
        misses = {str(k): 0 for k in KS}
        for sample, row in pairs:
            chunk_ids = [cid for cid, _ in row['results']]
            doc_ids = [did for did, _ in row['documents']]
            positive = set(sample['relevant_chunks'])
            positive_docs = set(sample['relevant_docs'])
            if not positive or not positive_docs:
                raise ValueError('Ground truth must not be empty')
            rank = next((i for i, cid in enumerate(chunk_ids[:100], 1) if cid in positive), None)
            reciprocal_ranks.append(1 / rank if rank else 0)
            for k in KS:
                recall = len(set(chunk_ids[:k]) & positive) / len(positive)
                recalls[str(k)].append(recall)
                doc_recalls[str(k)].append(len(set(doc_ids[:k]) & positive_docs) / len(positive_docs))
                misses[str(k)] += int(recall == 0)
        return dict(num_queries=len(pairs), recall_at_k={k: statistics.mean(v) for k, v in recalls.items()},
                    document_recall_at_k={k: statistics.mean(v) for k, v in doc_recalls.items()},
                    mrr_at_100=statistics.mean(reciprocal_ranks), zero_hit_queries_at_k=misses)

    pairs = list(zip(samples, rows, strict=True))
    metrics = summarize(pairs)
    metrics['by_topic'] = {str(topic): summarize([(s, r) for s, r in pairs if s['topic'] == topic])
                           for topic in sorted({s['topic'] for s in samples})}
    latencies = sorted(r['total_latency_ms'] for r in rows)
    metrics['latency_ms'] = dict(mean=statistics.mean(latencies),
                                p50=latencies[(len(latencies)-1)//2],
                                p95=latencies[math.ceil(.95*len(latencies))-1])
    return metrics

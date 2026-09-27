"""Evaluate inferred source-context retrieval on the full ViMedQA test split."""

import argparse
import json
import time
from pathlib import Path

from src.adapters import adapt_query
from src.data.schema import QueryRecord
from src.config import load_config
from src.service import build_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vimed_bm25.yaml')
    parser.add_argument('--data', type=Path, default=Path('data/vimed'))
    parser.add_argument('--output', type=Path, default=Path('outputs/vimed_bm25'))
    args = parser.parse_args()
    pipeline = build_pipeline(load_config(args.config))
    samples = [json.loads(line) for line in (args.data / 'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    args.output.mkdir(parents=True, exist_ok=True)
    ranks, latencies, topic_ranks = [], [], {}
    with (args.output / 'rankings.jsonl').open('w', encoding='utf-8') as handle:
        for i, sample in enumerate(samples):
            query = adapt_query(QueryRecord(id=sample['id'], query=sample['query']))
            start = time.perf_counter()
            pipeline.run_query(query.query_id, query.text)
            results = sorted(
                [(cid, scores['selection_score']) for cid, scores in pipeline.last_candidate_scores.items()
                 if scores['selection_eligible']], key=lambda item: (-item[1], item[0]),
            )[:100]
            latencies.append((time.perf_counter() - start) * 1000)
            positives = set(sample['relevant_chunks'])
            rank = next((i for i, (cid, _) in enumerate(results, 1) if cid in positives), None)
            ranks.append(rank)
            topic_ranks.setdefault(str(sample['topic']), []).append(rank)
            handle.write(json.dumps(dict(id=sample['id'], first_positive_rank=rank,
                                         latency_ms=latencies[-1], results=results)) + '\n')
            if (i + 1) % 250 == 0:
                print(f'Evaluated {i + 1}/{len(samples)}', flush=True)
    def summarize(values):
        return dict(num_queries=len(values), recall_at_k={str(k): sum(r is not None and r <= k for r in values) / len(values)
                                                         for k in (1, 3, 5, 10, 20, 50, 100)},
                    mrr_at_100=sum(1 / r if r else 0 for r in values) / len(values))
    metrics = summarize(ranks)
    metrics.update(backend=load_config(args.config).backend, mean_latency_ms=sum(latencies) / len(latencies),
                   by_topic={topic: summarize(values) for topic, values in topic_ranks.items()},
                   corpus=json.loads((args.data / 'manifest.json').read_text(encoding='utf-8')),
                   note='Recall uses one inferred source-context positive per query; no test tuning.')
    (args.output / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(metrics, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

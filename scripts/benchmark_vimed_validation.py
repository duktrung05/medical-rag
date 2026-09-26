"""Run/cache each validation retrieval stage with timeout, integrity checks and resume."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from src.adapters import adapt_query
from src.config import load_pipeline_config
from src.data.loader import DataLoader
from src.data.schema import QueryRecord
from src.evaluation.ranking_metrics import summarize_rankings, validate_ranking
from src.indexing.sparse_index import corpus_sha256
from src.retrieval.fusion import reciprocal_rank_fusion
from src.scoring.document_score import aggregate_doc_scores
from src.service import build_pipeline


CONFIGS = dict(bm25='configs/vimed_bm25.yaml', dense='configs/vimed_dense_bge_m3.yaml',
               hybrid='configs/vimed_hybrid.yaml', rerank='configs/vimed_hybrid_rerank.yaml')


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CachedRetriever:
    def __init__(self, rankings):
        self.rankings = rankings

    def search(self, query, top_k):
        return self.rankings[query][:top_k]


def load_dependency(root, stage, samples, corpus_hash, samples_hash):
    directory = root / stage
    manifest = json.loads((directory / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['status'] != 'complete' or manifest['corpus_sha256'] != corpus_hash or manifest['samples_sha256'] != samples_hash:
        raise ValueError(f'Incompatible or incomplete {stage} cache')
    if manifest['rankings_sha256'] != file_hash(directory / 'rankings.jsonl'):
        raise ValueError(f'{stage} ranking cache checksum mismatch')
    if manifest['config_sha256'] != file_hash(Path(CONFIGS[stage])):
        raise ValueError(f'{stage} configuration changed since measurement')
    rows = read_rows(directory / 'rankings.jsonl')
    if [r['id'] for r in rows] != [s['id'] for s in samples]:
        raise ValueError(f'{stage} cache query IDs differ')
    if len(rows) != len(samples):
        raise ValueError(f'{stage} cache query count differs')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=CONFIGS, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/vimed/validation'))
    parser.add_argument('--output', type=Path, default=Path('outputs/vimed_validation'))
    parser.add_argument('--timeout-seconds', type=int, default=3600)
    parser.add_argument('--limit', type=int, help='Resource probe only; writes to a separate --output directory')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.timeout_seconds <= 0 or (args.limit is not None and args.limit <= 0):
        parser.error('Timeout/limit must be positive')
    if args.limit and args.output == Path('outputs/vimed_validation'):
        parser.error('Resource probes require a separate --output directory')
    if not args.worker:
        command = [sys.executable, '-m', 'scripts.benchmark_vimed_validation', *sys.argv[1:], '--worker']
        print(f'Starting {args.stage}; timeout={args.timeout_seconds}s', flush=True)
        try:
            completed = subprocess.run(command, timeout=args.timeout_seconds)
        except subprocess.TimeoutExpired:
            print(f'{args.stage} timed out; metrics not complete; use the same command to resume', file=sys.stderr)
            raise SystemExit(124)
        raise SystemExit(completed.returncode)

    config_path = Path(CONFIGS[args.stage])
    config = load_pipeline_config(config_path)
    samples_path = args.data / 'samples.jsonl'
    samples = read_rows(samples_path)
    if args.limit:
        samples = samples[:args.limit]
    samples_hash = hashlib.sha256(json.dumps(samples, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    chunks = DataLoader.load_chunks(config.corpus)
    corpus_hash = corpus_sha256(chunks)
    validation_manifest = json.loads((args.data / 'manifest.json').read_text(encoding='utf-8'))
    if validation_manifest['split'] != 'validation' or validation_manifest['corpus_sha256'] != corpus_hash:
        raise ValueError('Validation split/corpus manifest mismatch')
    lookup = {c.chunk_id: c for c in chunks}
    allowed = set(lookup)
    for s in samples:
        if not set(s['relevant_chunks']) <= allowed:
            raise ValueError('Ground truth references unknown chunks')
    normalized = [adapt_query(QueryRecord(id=s['id'], query=s['query'])).text for s in samples]
    if len(set(normalized)) != len(normalized):
        # Cache uses query text; identical texts are valid only if their rankings agree.
        print('Validation includes identical normalized query texts', flush=True)

    directory = args.output / args.stage
    directory.mkdir(parents=True, exist_ok=True)
    identity = dict(stage=args.stage, corpus_sha256=corpus_hash, samples_sha256=samples_hash,
                    config_sha256=file_hash(config_path), split='validation', num_queries=len(samples),
                    probe=bool(args.limit))
    state_path = directory / 'run_identity.json'
    if state_path.exists() and json.loads(state_path.read_text()) != identity:
        raise ValueError('Existing run config/data differ; choose a new output directory')
    state_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    if (directory / 'manifest.json').exists():
        load_dependency(args.output, args.stage, samples, corpus_hash, samples_hash)
        print(f'{args.stage}: full compatible result already exists', flush=True)
        return

    dependency_rows, candidate_cache, base_latencies = {}, {}, {}
    if args.stage == 'hybrid':
        for name in ('bm25', 'dense'):
            dependency_rows[name] = load_dependency(args.output, name, samples, corpus_hash, samples_hash)
        for i, text in enumerate(normalized):
            start = time.perf_counter()
            candidate_cache[text] = reciprocal_rank_fusion([dependency_rows[n][i]['results'] for n in ('bm25', 'dense')],
                                                           k=config.fusion.rrf_k, top_k=config.fusion.top_k)
            base_latencies[text] = sum(dependency_rows[n][i]['total_latency_ms'] for n in ('bm25', 'dense')) + (time.perf_counter()-start)*1000
    elif args.stage == 'rerank':
        dependency_rows['hybrid'] = load_dependency(args.output, 'hybrid', samples, corpus_hash, samples_hash)
        for i, text in enumerate(normalized):
            candidate_cache[text] = dependency_rows['hybrid'][i]['results']
            base_latencies[text] = dependency_rows['hybrid'][i]['total_latency_ms']

    override = (CachedRetriever(candidate_cache), config.fusion.top_k) if candidate_cache else None
    pipeline = build_pipeline(config, retriever_override=override)
    import torch
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    # Load reranker / warm up without including initialization in query latency.
    warmup_start = time.perf_counter()
    pipeline.run_query(samples[0]['id'], normalized[0])
    warmup_seconds = time.perf_counter()-warmup_start
    row_path = directory / 'rankings.jsonl'
    rows = read_rows(row_path) if row_path.exists() else []
    if len(rows) > len(samples):
        raise ValueError('Partial cache has too many queries')
    for row in rows:
        validate_ranking(row['results'], allowed)
    if [r['id'] for r in rows] != [s['id'] for s in samples[:len(rows)]]:
        raise ValueError('Partial cache IDs/order differ; do not resume')
    started = time.perf_counter()
    with row_path.open('a', encoding='utf-8') as handle:
        for i in range(len(rows), len(samples)):
            sample, text = samples[i], normalized[i]
            start = time.perf_counter()
            prediction = pipeline.run_query(sample['id'], text)
            stage_ms = (time.perf_counter()-start)*1000
            eligible = [(cid, item['selection_score']) for cid, item in pipeline.last_candidate_scores.items()
                        if item['selection_eligible']]
            ranking = sorted(eligible, key=lambda item: (-item[1], item[0]))
            validate_ranking(ranking, allowed)
            if len(ranking) > 100:
                raise ValueError('Comparison protocol caps all stages at 100')
            if args.stage == 'rerank' and set(cid for cid, _ in ranking) != set(cid for cid, _ in candidate_cache[text]):
                raise ValueError('Reranker dropped or introduced candidates')
            doc_scores = aggregate_doc_scores(dict(ranking), pipeline.doc_map.chunk_to_doc,
                                             method=config.scoring.document.aggregation,
                                             top_n=config.scoring.document.top_n_mean)
            documents = sorted(doc_scores.items(), key=lambda item: (-item[1], item[0]))
            positive = set(sample['relevant_chunks'])
            first_rank = next((r for r, (cid, _) in enumerate(ranking, 1) if cid in positive), None)
            row = dict(id=sample['id'], query=text, topic=sample['topic'], first_positive_rank=first_rank,
                       stage_latency_ms=stage_ms, total_latency_ms=base_latencies.get(text, 0)+stage_ms,
                       results=ranking, documents=documents, prediction=prediction.model_dump())
            if args.stage == 'hybrid':
                pools = [{cid for cid, _ in dependency_rows[n][i]['results']} for n in ('bm25', 'dense')]
                row['union_positive_coverage'] = len(positive & (pools[0] | pools[1])) / len(positive)
            handle.write(json.dumps(row, ensure_ascii=False)+'\n')
            handle.flush()
            rows.append(row)
            if (i+1) % (1 if args.limit else 50) == 0 or i+1 == len(samples):
                elapsed = time.perf_counter()-started
                print(f'{args.stage}: {i+1}/{len(samples)}; elapsed_this_run={elapsed:.1f}s', flush=True)

    metrics = summarize_rankings(samples, rows)
    metrics.update(stage=args.stage, split='validation', candidate_depth=100,
                   latency_method='measured_pipeline_query' if args.stage in ('bm25', 'dense') else 'sum_of_cached_stage_latencies_plus_current_stage',
                   warmup_seconds=warmup_seconds,
                   peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                   peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                   note='Source-context positives inferred from QA; unjudged passages may still be relevant.')
    if args.stage == 'hybrid':
        metrics['union_candidate_recall'] = sum(r['union_positive_coverage'] for r in rows)/len(rows)
    (directory / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    manifest = dict(**identity, status='complete', rankings_sha256=file_hash(row_path),
                    config=str(config_path), config_snapshot=config.model_dump(mode='json'),
                    warmup_seconds=warmup_seconds)
    (directory / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()

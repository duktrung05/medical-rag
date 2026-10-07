"""Build two controlled submission probes from authentic cached reranker scores.

Run from the repository root with the evidence environment. Both ZIPs contain
one predictions.json at the archive root; neither requires new GPU inference.
"""
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from r2ai.evidence_io import fingerprint, read_ranking_cache, selection_summary
from r2ai.validate import validate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--max-docs', type=int, default=80)
    parser.add_argument('--max-chunks', type=int, default=60)
    parser.add_argument('--threshold', type=float, default=-6.0)
    parser.add_argument('--window', type=int, default=1)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'outputs/submissions/vibiomir_probes_20261007')
    args = parser.parse_args()
    if not 0 < args.max_chunks <= args.max_docs or args.window < 0:
        parser.error('Require 0 < max-chunks <= max-docs and window >= 0')
    if args.output.exists():
        raise FileExistsError(f'Choose a new experiment directory: {args.output}')

    cache = ROOT / 'outputs/evidence/test_20261006/rankings.jsonl.gz'
    rankings, metadata = read_ranking_cache(cache)
    chunks_path = ROOT / 'data/vibiomir/chunks_v4.parquet'
    query_path = ROOT / 'data/raw/vibiomir/query.parquet'
    corpus_path = ROOT / 'data/raw/vibiomir/links_corpus.parquet'
    for key, path in [('chunks', chunks_path), ('queries', query_path)]:
        current = fingerprint(path)
        if current['sha256'] != metadata['inputs'][key]['sha256']:
            raise ValueError(f'Ranking cache no longer matches {key}')
    expected = set(pq.read_table(query_path, columns=['id']).column(0).to_pylist())
    if len(rankings) != len(expected) or {r.query_id for r in rankings} != expected:
        raise ValueError('Ranking cache does not cover all official queries')

    selections = []
    wanted = defaultdict(set)
    centers = {}
    for row in rankings:
        if any(c.rerank_score is None for c in row.candidates):
            raise ValueError('Every candidate in these probes must have a real score')
        candidates = sorted(row.candidates,
                            key=lambda c: (-c.rerank_score, c.doc_id, c.chunk_index))
        doc_scores = {}
        best_by_doc = {}
        for candidate in candidates:
            doc_scores.setdefault(candidate.doc_id, candidate.rerank_score)
            best_by_doc.setdefault(candidate.doc_id, candidate)
        docs = sorted(doc_scores, key=lambda d: (-doc_scores[d], d))[:args.max_docs]
        chosen = [best_by_doc[d] for d in docs
                  if best_by_doc[d].rerank_score >= args.threshold][:args.max_chunks]
        # Preserve separate documents with identical evidence: doc_id is part
        # of the submission identity. Do not globally deduplicate text.
        selections.append((row.query_id, docs, chosen))
        for candidate in chosen:
            key = candidate.doc_id, candidate.chunk_index
            if key in centers and centers[key] != candidate.chunk_text:
                raise ValueError(f'Conflicting cache text for {key}')
            centers[key] = candidate.chunk_text
            wanted[candidate.doc_id].update(range(max(0, candidate.chunk_index - args.window),
                                                  candidate.chunk_index + args.window + 1))

    needed_ids = np.array(list(wanted), dtype=np.int64)
    texts = {}
    scanned = 0
    print(f'Loading adjacent source chunks for {len(wanted):,} documents', flush=True)
    for batch in pq.ParquetFile(chunks_path).iter_batches(
            batch_size=100_000, columns=['doc_id', 'chunk_index', 'text']):
        ids = batch.column(0).to_numpy(zero_copy_only=False)
        for pos in np.flatnonzero(np.isin(ids, needed_ids)):
            pos = int(pos)
            doc_id = int(ids[pos])
            index = int(batch.column(1)[pos].as_py())
            if index in wanted[doc_id]:
                key = doc_id, index
                if key in texts:
                    raise ValueError(f'Duplicate source chunk identity: {key}')
                texts[key] = batch.column(2)[pos].as_py()
        scanned += batch.num_rows
        if scanned % 2_000_000 == 0:
            print(f'Scanned {scanned:,} rows', flush=True)
    for key, text in centers.items():
        if texts.get(key) != text:
            raise ValueError(f'Cached evidence does not match source chunk {key}')

    args.output.mkdir(parents=True)
    report = {
        'inputs': {'rankings': fingerprint(cache), 'chunks': metadata['inputs']['chunks'],
                   'queries': metadata['inputs']['queries']},
        'reranker': metadata['reranker'],
        'config': {'max_docs': args.max_docs, 'max_chunks': args.max_chunks,
                   'chunk_threshold': args.threshold, 'window': args.window,
                   'document_order': 'descending maximum authentic chunk reranker score',
                   'chunks_per_doc': 1, 'cross_document_text_deduplication': False},
        'limitations': [
            'Threshold and cutoffs are experimental, not calibrated on ViBioMIR labels.',
            'Only the 120 documents per query with evidence in the original ZIP are scored.',
            'The other 80 baseline documents and other chunks in each document are not reranked.',
            'Widening uses source text from adjacent chunk indices in the same document.',
            'Official chunk-matching implementation was not independently accessible.',
            'No official score for either generated ZIP is available yet.',
        ],
        'source_chunk_rows_scanned': scanned,
        'exact_centers_verified': len(centers),
        'variants': {},
    }
    parent_sets = {}
    for variant in ['exact', 'window1']:
        records = []
        widened = 0
        for qid, docs, chosen in selections:
            chunks = []
            for candidate in chosen:
                text = candidate.chunk_text
                if variant == 'window1':
                    neighbors = [texts[candidate.doc_id, index]
                                 for index in range(max(0, candidate.chunk_index - args.window),
                                                    candidate.chunk_index + args.window + 1)
                                 if (candidate.doc_id, index) in texts]
                    text = '\n\n'.join(neighbors)
                    if candidate.chunk_text not in text:
                        raise ValueError('Expansion lost the exact source evidence')
                    widened += text != candidate.chunk_text
                chunks.append({'doc_id': candidate.doc_id, 'chunk_text': text})
            records.append({'id': qid, 'relevant_docs': docs, 'relevant_chunks': chunks})
        stem = f'vibiomir_rerank{args.max_docs}_{variant}_20261007'
        json_path = args.output / f'{stem}.json'
        zip_path = args.output / f'{stem}.zip'
        payload = ('[\n' + ',\n'.join(json.dumps(r, ensure_ascii=False, allow_nan=False)
                                     for r in records) + '\n]\n').encode('utf-8')
        json_path.write_bytes(payload)
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
            zf.write(json_path, arcname='predictions.json')
        with zipfile.ZipFile(zip_path) as zf:
            if zf.namelist() != ['predictions.json'] or zf.testzip() is not None:
                raise ValueError('Invalid archive layout or CRC')
            if zf.read('predictions.json') != payload:
                raise ValueError('ZIP and standalone JSON differ')
        errors = validate(zip_path, query_path=query_path, corpus_path=corpus_path)
        if errors or zip_path.stat().st_size > 48_000_000:
            raise ValueError(f'Invalid or oversized submission: {errors}; lower max-chunks')
        identities = [(r['id'], r['relevant_docs'], [c['doc_id'] for c in r['relevant_chunks']])
                      for r in records]
        if parent_sets and identities != parent_sets['exact']:
            raise ValueError('Controlled probes changed document/chunk parent selection')
        parent_sets[variant] = identities
        chars = [len(c['chunk_text']) for r in records for c in r['relevant_chunks']]
        report['variants'][variant] = {
            **selection_summary(records), 'json': fingerprint(json_path),
            'zip': fingerprint(zip_path), 'validation_errors': errors,
            'average_chunk_characters': sum(chars) / len(chars) if chars else 0,
            'chunks_with_added_context': widened,
        }
        print(f'VALID: {zip_path.name} ({zip_path.stat().st_size / 1e6:.2f} MB)', flush=True)
    (args.output / 'experiment_report.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()

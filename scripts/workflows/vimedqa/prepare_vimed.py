"""Convert local ViMedQA contexts to a retrieval corpus and test benchmark."""

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import pandas as pd

from src.config import SparseRetrievalConfig
from src.data.schema import ChunkRecord
from src.retrieval.bm25 import BM25Retriever


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:24]


def write_rows(path, rows):
    with path.open('w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw', type=Path, default=Path('data/raw/vimedaqa/all'))
    parser.add_argument('--output', type=Path, default=Path('data/vimed'))
    parser.add_argument('--index', type=Path, default=Path('artifacts/vimed_bm25'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    contexts, metadata, samples, skipped = {}, {}, [], {}
    for split in ('train', 'validation', 'test'):
        files = sorted(args.raw.glob(f'{split}-*.parquet'))
        if not files:
            raise FileNotFoundError(f'Missing {split} parquet in {args.raw}')
        frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
        skipped[split] = 0
        for row in frame.to_dict('records'):
            value = row.get('context')
            context = '' if pd.isna(value) else unicodedata.normalize('NFC', str(value)).strip()
            if not context:
                skipped[split] += 1
                continue
            # Deduplicate identical contexts, including those shared across splits.
            cid = 'ctx_' + digest(context)
            url = str(row.get('article_url') or '').strip()
            doc_id = 'doc_' + digest(url or context)
            if cid not in contexts:
                contexts[cid] = dict(chunk_id=cid, doc_id=doc_id, chunk_index=0,
                                     language='vi', text=context, title=str(row.get('title') or ''))
                metadata[cid] = dict(article_url=url, keyword=str(row.get('keyword') or ''),
                                     topic=int(row['topic']))
            if split == 'test':
                samples.append(dict(id=str(row['question_idx']), query=str(row['question']),
                                    answer=str(row['answer']), topic=int(row['topic']),
                                    relevant_chunks=[cid], relevant_docs=[contexts[cid]['doc_id']]))
    counters = {}
    for chunk in contexts.values():
        doc = chunk['doc_id']
        chunk['chunk_index'] = counters.get(doc, 0)
        counters[doc] = chunk['chunk_index'] + 1
    if len({s['id'] for s in samples}) != len(samples):
        raise ValueError('Duplicate test query IDs')
    write_rows(args.output / 'chunks.jsonl', contexts.values())
    write_rows(args.output / 'queries.jsonl', [dict(id=s['id'], query=s['query']) for s in samples])
    write_rows(args.output / 'ground_truth.jsonl', [dict(id=s['id'], relevant_chunks=s['relevant_chunks'],
                                                       relevant_docs=s['relevant_docs']) for s in samples])
    write_rows(args.output / 'samples.jsonl', samples)
    (args.output / 'metadata.json').write_text(json.dumps(metadata, ensure_ascii=False), encoding='utf-8')
    chunks = [ChunkRecord(**row) for row in contexts.values()]
    corpus_hash = BM25Retriever.build_index(chunks, args.index, SparseRetrievalConfig())
    manifest = dict(dataset='local ViMedQA all', corpus_splits=['train', 'validation', 'test'],
                    query_split='test', num_chunks=len(chunks), num_docs=len(counters),
                    num_test_queries=len(samples), skipped_empty_context=skipped, corpus_sha256=corpus_hash,
                    protocol='Closed corpus context retrieval; questions and answers are not indexed. '
                    'One exact deduplicated source context is the inferred positive per question. '
                    'This is not an exhaustive relevance judgment or an official retrieval benchmark.')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

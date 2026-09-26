"""Map validation questions to the existing, immutable ViMed context corpus."""

import argparse
import json
import unicodedata
from pathlib import Path

import pandas as pd

from scripts.prepare_vimed import digest, write_rows
from src.data.loader import DataLoader
from src.indexing.sparse_index import corpus_sha256


def prepare(raw: Path, corpus: Path, output: Path):
    chunks = DataLoader.load_chunks(corpus)
    lookup = {c.chunk_id: c for c in chunks}
    samples, skipped = [], []
    files = sorted(raw.glob('validation-*.parquet'))
    if not files:
        raise FileNotFoundError(f'Missing validation parquet in {raw}')
    rows = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True).to_dict('records')
    for row in rows:
        value = row.get('context')
        context = '' if pd.isna(value) else unicodedata.normalize('NFC', str(value)).strip()
        if not context:
            skipped.append(str(row['question_idx']))
            continue
        cid = 'ctx_' + digest(context)
        if cid not in lookup or lookup[cid].text != context:
            raise ValueError(f'Validation context is absent from the existing corpus: {cid}')
        samples.append(dict(id=str(row['question_idx']), query=str(row['question']),
                            answer=str(row['answer']), topic=int(row['topic']),
                            relevant_chunks=[cid], relevant_docs=[lookup[cid].doc_id]))
    ids = {s['id'] for s in samples}
    if len(ids) != len(samples) or not samples:
        raise ValueError('Validation IDs must be unique and nonempty')
    test_path = corpus.parent / 'queries.jsonl'
    test_ids = {q.id for q in DataLoader.load_queries(test_path)} if test_path.exists() else set()
    if ids & test_ids:
        raise ValueError('Validation query IDs overlap the test split')
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / 'samples.jsonl', samples)
    write_rows(output / 'queries.jsonl', [dict(id=s['id'], query=s['query']) for s in samples])
    write_rows(output / 'ground_truth.jsonl', [dict(id=s['id'], relevant_docs=s['relevant_docs'],
                                                 relevant_chunks=s['relevant_chunks']) for s in samples])
    manifest = dict(split='validation', raw_rows=len(rows), num_queries=len(samples),
                    skipped_empty_context_ids=skipped, corpus_sha256=corpus_sha256(chunks),
                    num_chunks=len(chunks), test_id_overlap=0,
                    protocol='Closed corpus; one inferred source-context positive per question. '
                    'Validation queries only; corpus and existing indexes unchanged.')
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw', type=Path, default=Path('data/raw/vimedaqa/all'))
    p.add_argument('--corpus', type=Path, default=Path('data/vimed/chunks.jsonl'))
    p.add_argument('--output', type=Path, default=Path('data/vimed/validation'))
    args = p.parse_args()
    print(json.dumps(prepare(args.raw, args.corpus, args.output), indent=2))


if __name__ == '__main__':
    main()

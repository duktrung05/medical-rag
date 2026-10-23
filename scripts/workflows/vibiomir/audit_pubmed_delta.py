"""Reconcile complete PubMed raw/text/vector outputs against the pinned input."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from scripts.workflows.vibiomir.collect_pubmed import (
    atomic_json, DOC_SCHEMA, file_sha, now, parse_records, ResourceGate,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wait-for-embedding',action='store_true')
    args=parser.parse_args()
    os.nice(10)
    cores=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,cores[:4])
    output=args.output.resolve()
    if args.wait_for_embedding:
        deadline=time.monotonic()+3600
        while not (output/'vectors/embedding_report.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Embedding did not finish within audit wait window')
            print('Audit waiting for complete embedding report',flush=True)
            time.sleep(30)
    queue=pq.read_table(args.queue).to_pylist()
    expected={x['id']:(x['pmid'],x['official_url']) for x in queue}
    con=sqlite3.connect(f'file:{output / "state.sqlite"}?mode=ro',uri=True)
    con.execute('PRAGMA cache_size=-8192')
    if con.execute("SELECT value FROM meta WHERE key='queue_sha256'").fetchone()[0]!=file_sha(args.queue):
        raise ValueError('Audit input queue fingerprint changed')
    states=Counter();languages=Counter();raw_files={};seen=set();metadata=[];missing=[]
    sample=[];empty_titles=0;text_hashes={};usable_ids=set()
    for values in con.execute('SELECT '+','.join(DOC_SCHEMA.names)+' FROM documents ORDER BY id'):
        row=dict(zip(DOC_SCHEMA.names,values));ident=row['id']
        if ident not in expected or expected[ident]!=(row['pmid'],row['official_url']) or ident in seen:
            raise ValueError('Official ID–PMID–URL mismatch or duplicate')
        seen.add(ident);states[row['status']]+=1
        text_hashes[ident]=row['text_sha256']
        if row['status'] in ('usable_abstract','metadata_only'):
            if hashlib.sha256(row['text'].encode()).hexdigest()!=row['text_sha256']:
                raise ValueError('Text checksum mismatch')
            if not row['raw_file'] or not row['raw_sha256']:
                raise ValueError('Missing raw provenance')
            if row['raw_file'] in raw_files and raw_files[row['raw_file']]!=row['raw_sha256']:
                raise ValueError('Inconsistent raw provenance')
            raw_files[row['raw_file']]=row['raw_sha256']
            content=json.loads(row['metadata_json'])
            abstract='\n\n'.join((f"{x['label']}: {x['text']}" if x['label'] else x['text'])
                                for x in content['sections'] if x['text'])
            reproduced=f"{row['title']}\n\n{abstract}".strip()
            if reproduced!=row['text'] or (row['status']=='usable_abstract' and not abstract):
                raise ValueError('Text does not reproduce original ordered abstract sections')
            if not row['title']:empty_titles+=1
            languages[row['language']]+=1
            if row['status']=='metadata_only':metadata.append(row)
            else:
                usable_ids.add(ident)
                if len(sample)<30:sample.append({k:row[k] for k in ['id','pmid','title','language','text_sha256']})
        else:
            missing.append(row)
    if seen!=set(expected) or sum(states.values())!=240000:
        raise ValueError('Incomplete official ID ledger')
    if set(states)-{'usable_abstract','metadata_only','unavailable'}:
        raise ValueError('Pending or failed collection remains')
    gate=ResourceGate();raw_bytes=0;raw_records=0
    for i,(name,digest) in enumerate(raw_files.items()):
        gate.wait();path=Path(name)
        if file_sha(path)!=digest:raise ValueError('Raw checksum mismatch')
        body=gzip.decompress(path.read_bytes());parsed=parse_records(body)
        sidecar=json.loads(path.with_suffix('.json').read_text())
        if digest!=sidecar['gzip_sha256'] or hashlib.sha256(body).hexdigest()!=sidecar['body_sha256']:
            raise ValueError('Raw sidecar checksum mismatch')
        if set(parsed)-set(sidecar['requested_pmids']):raise ValueError('Unexpected raw PMIDs')
        raw_bytes+=path.stat().st_size;raw_records+=len(parsed)
        if (i+1)%200==0:print('Verified raw batch files',i+1,'/',len(raw_files),flush=True)
    parts=[];exported=Counter();parquet_rows=0;exported_ids=set()
    for path in sorted((output/'documents').glob('part_*.parquet')):
        file=pq.ParquetFile(path);parquet_rows+=file.metadata.num_rows
        for batch in file.iter_batches(columns=['id','status','pmid','text_sha256'],batch_size=5000):
            columns=batch.to_pydict()
            for ident,status,pmid,digest in zip(columns['id'],columns['status'],columns['pmid'],columns['text_sha256']):
                if ident not in expected or ident in exported_ids or expected[ident][0]!=pmid or digest!=text_hashes[ident]:
                    raise ValueError('Duplicate or mismatched exported ID/PMID/text checksum')
                exported_ids.add(ident)
                exported[status]+=1
        parts.append({'path':str(path),'rows':file.metadata.num_rows,'sha256':file_sha(path)})
    if parquet_rows!=240000 or exported!=states or exported_ids!=set(expected):raise ValueError('Parquet export/state mismatch')
    for name,rows in [('metadata_only.parquet',metadata),('missing_pmids.parquet',missing)]:
        temp=output/(name+'.tmp');pq.write_table(pa.Table.from_pylist(rows,schema=DOC_SCHEMA),temp,compression='zstd')
        os.replace(temp,output/name)
    embedding=json.loads((output/'vectors/embedding_report.json').read_text())
    if not embedding['complete'] or embedding['rows']!=states['usable_abstract']:
        raise ValueError('Embedding/usable-document coverage mismatch')
    vector_ids=set()
    for part in embedding['parts']:
        path=Path(part['path'])
        if file_sha(path)!=part['sha256']:raise ValueError('Embedding part checksum mismatch')
        with np.load(path,allow_pickle=False) as arrays:
            for ident,digest in zip(arrays['ids'],arrays['text_sha256']):
                ident=int(ident)
                if ident not in usable_ids or ident in vector_ids or str(digest)!=text_hashes[ident]:
                    raise ValueError('Embedding text/ID mismatch')
                vector_ids.add(ident)
    if vector_ids!=usable_ids:raise ValueError('Embedding IDs do not equal usable IDs')
    for name,digest in embedding['files'].items():
        if file_sha(output/'vectors'/name)!=digest:raise ValueError('Merged embedding checksum mismatch')
    revision=con.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
    report={'finished_at':now(),'complete':True,'corpus_revision':revision,
            'input_queue_sha256':file_sha(args.queue),'official_ids':240000,'states':dict(states),
            'raw_batch_files_verified':len(raw_files),'raw_bytes_referenced':raw_bytes,
            'raw_records_in_referenced_batches':raw_records,'documents_exported':parquet_rows,
            'metadata_languages':dict(languages),'metadata_language_is_not_detected_abstract_language':True,
            'empty_titles':empty_titles,'embedding_rows':embedding['rows'],
            'embedding_dimensions':embedding['dimensions'],'physical_gpu_devices':embedding['physical_gpu_devices'],
            'preparation_workers_per_gpu':4,'documents_truncated_in_encoder':embedding['documents_truncated_in_encoder'],
            'full_abstract_retained':True,'sample':sample,'documents_parts':parts,
            'index_runtime_switched':False,
            'limitation':'PubMed citation/abstract delta; unavailable/metadata-only separate from usable content. Encoder input limited to 384 tokens; full text remains in documents.'}
    atomic_json(output/'release_manifest.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('sample','documents_parts','metadata_languages')},ensure_ascii=False,indent=2),flush=True)
    con.close()


if __name__=='__main__':main()

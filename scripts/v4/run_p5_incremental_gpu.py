"""Incremental A6000 GPU scoring of immutable A0 lineage; CPU producer is read-only."""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import pyarrow.parquet as pq

from scripts.v4.run_p5_2_split import (CONFIG, ROOT, FROZEN_CONFIG_SHA256, ScoreCache,
    encoded, finalize, read, validate_pair_shard, validate_score_shard)
from src.medrag.core.config import load_config
from src.medrag.data.pair_lineage import (PairLineageResolver, canonical_bytes,
    write_once, write_atomic_replace)
from src.medrag.data.phase2_batch import sha256_file
from src.medrag.data.disk_store import DiskCorpusStore
from src.medrag.evidence.verifier import verify

FORMAT = 'medrag-v4-p5-incremental-a6000-v1'
SOURCE = ROOT / 'outputs/v4/a0_fast_parent_first_v1_retry_20261023'
OUTPUT = ROOT / 'outputs/v4/p5_gpu_incremental_a6000_20261023'
VERSIONS = {'torch':'2.6.0+cu124', 'torchvision':'0.21.0+cu124',
            'sentence-transformers':'4.1.0', 'transformers':'4.51.3',
            'huggingface-hub':'0.36.2', 'tokenizers':'0.21.4'}
CODE = ('scripts/v4/run_p5_incremental_gpu.py', 'scripts/v4/run_p5_2_split.py',
        'src/medrag/ranking/reranker.py', 'src/medrag/ranking/score_cache.py',
        'src/medrag/core/model_revision.py', 'src/medrag/data/pair_lineage.py',
        'src/medrag/core/pipeline.py', 'src/medrag/evidence/selector.py',
        'src/medrag/evidence/context.py', 'src/medrag/adapters/validation.py')

def digest(value): return hashlib.sha256(canonical_bytes(value)).hexdigest()
def utc(): return dt.datetime.now(dt.timezone.utc).isoformat()


def transport_lock(resolver):
    lock = copy.deepcopy(resolver.lock)
    lock.update(format='medrag-v4-p5-2-split-lock-v1', mode='full')
    lock['transport_derivation'] = {
        'format': FORMAT, 'source_lock_sha256': sha256_file(resolver.lock_path),
        'lineage_sha256': sha256_file(resolver.lineage_path),
        'only_rewritten_pair_field': 'lock_sha256'}
    return lock


class Handoff:
    def __init__(self, source, output):
        self.source, self.output = source.resolve(), output.resolve()
        if self.output == self.source or self.source in self.output.parents:
            raise ValueError('GPU output must be separate from CPU producer')
        self.resolver = PairLineageResolver(self.source)
        self.lock = transport_lock(self.resolver)
        if self.lock['query_ids'] != list(range(1,1201)):
            raise ValueError('Expected full 1200-query identity')
        if sha256_file(CONFIG) != FROZEN_CONFIG_SHA256:
            raise ValueError('Frozen config changed')
        migration = read(ROOT/'configs/vibiomir/phase2_batch_migration_v1.json')
        query_path = ROOT/migration['official_queries']
        if sha256_file(query_path) != self.lock['official_queries_sha256']:
            raise ValueError('Official query identity differs')
        self.queries = {int(r['id']):str(r['query']) for r in pq.read_table(query_path,columns=['id','query']).to_pylist()}
        self.staging = self.output/'staging'
        write_once(self.staging/'lock.json', self.lock)
        cache_path = self.staging/'key_contract.sqlite'
        self.cache = ScoreCache(cache_path,self.lock['score_contract'],create=not cache_path.exists())
        index = Path(self.resolver.lock['index_namespace'])
        manifest = read(index/'manifest.json')
        if sha256_file(index/'manifest.json') != self.lock['index_manifest_sha256']:
            raise ValueError('Index manifest differs')
        self.index = index
        self.store = DiskCorpusStore(index/'corpus.sqlite',manifest['corpus_version'],manifest['fingerprint'])
        self.lock_sha = sha256_file(self.staging/'lock.json')

    def available(self):
        local = {int(p.stem) for p in (self.source/'pairs').glob('*.json') if p.stem.isdigit()}
        return sorted((set(self.resolver.inherited)|local)&set(self.lock['query_ids']))

    def package(self, qid):
        original, origin, source_sha = self.resolver.validate(qid,self.queries[qid])
        pair = copy.deepcopy(original)
        pair['lock_sha256'] = self.lock_sha
        validate_pair_shard(pair,self.lock,qid,self.queries[qid],self.cache)
        # Read-only indexed source grounding; no retrieval or model work.
        candidates = pair['candidates']
        sources = self.store.sources_batch({d['official_id'] for d in candidates})
        passages = self.store.passages_batch([e['passage_id'] for d in candidates for e in d['evidence']])
        for doc in candidates:
            for evidence in doc['evidence']:
                p = passages[evidence['passage_id']]
                if not verify(p,sources[doc['official_id']]) or (evidence['text'],evidence['start'],evidence['end']) != (p.text,p.source_start,p.source_end):
                    raise ValueError(f'Source grounding failed: {qid}/{p.passage_id}')
        for p in pair['pairs']:
            source = sources[p['official_id']]; passage = passages[p['passage_id']]
            expected = f'{source.title}\n{passage.text}' if source.title else passage.text
            if p['payload'] != expected or passage.official_id != p['official_id']:
                raise ValueError(f'Payload/source mismatch: {qid}')
        write_once(self.staging/'pairs'/f'{qid}.json',pair)
        path, _ = self.resolver.resolve(qid)
        mapping = {'id':qid,'origin':origin,'source_path':str(path),'source_sha256':source_sha,
                   'transport_sha256':sha256_file(self.staging/'pairs'/f'{qid}.json'),
                   'pair_count':200,'input_hashes_sha256':digest([p['input_sha256'] for p in pair['pairs']])}
        write_once(self.staging/'lineage'/f'{qid}.json',mapping)
        return pair,mapping

    def batch(self, ids):
        if not ids or len(ids)!=len(set(ids)) or any(q not in self.available() for q in ids):
            raise ValueError('Invalid/duplicate/unpublished batch IDs')
        mappings = {}
        for qid in ids:
            _, mappings[str(qid)] = self.package(qid)
        manifest = {'format':FORMAT+'-partial-batch','transport_lock_sha256':self.lock_sha,
                    'query_ids':ids,'pair_count':200*len(ids),'shards':mappings}
        write_once(self.staging/'batches'/f'{digest(manifest)}.json',manifest)
        return manifest


def validate_checkpoint(checkpoint, pair, mapping, fingerprint, runtime_sha):
    if (checkpoint.get('format') != FORMAT+'-checkpoint'
            or checkpoint.get('runtime_sha256') != runtime_sha
            or checkpoint.get('lineage') != mapping):
        raise ValueError('Checkpoint runtime/lineage mismatch')
    score = checkpoint['score']
    validate_score_shard(score,pair,mapping['transport_sha256'],fingerprint)
    proof = checkpoint['proof']
    if (proof.get('query_id') != pair['id'] or proof.get('score_sha256') != digest(score)
            or proof.get('fresh_model_pairs') != 200 or proof.get('device') != 'cuda:0'
            or proof.get('runtime_sha256') != runtime_sha):
        raise ValueError('GPU proof mismatch')
    return score


def commit(output, checkpoint):
    qid = checkpoint['score']['id']
    # Atomic envelope is the authoritative resume checkpoint. Derived files can be repaired.
    write_once(output/'checkpoints'/f'{qid}.json', checkpoint)
    write_once(output/'scores'/f'{qid}.json', checkpoint['score'])
    write_once(output/'provenance'/f'{qid}.json', checkpoint['proof'])


def recover(handoff, runtime_sha):
    entries = {}
    for path in sorted((handoff.output/'checkpoints').glob('*.json'),key=lambda p:int(p.stem)):
        qid = int(path.stem)
        pair,mapping = handoff.package(qid)
        checkpoint = read(path)
        if checkpoint['score']['id'] != qid:
            raise ValueError('Checkpoint filename/query mismatch')
        validate_checkpoint(checkpoint,pair,mapping,handoff.cache.fingerprint,runtime_sha)
        commit(handoff.output,checkpoint)
        entries[str(qid)] = {'checkpoint_sha256':sha256_file(path),
                            'score_sha256':digest(checkpoint['score']),
                            'source_sha256':mapping['source_sha256'],
                            'seconds':checkpoint['proof']['seconds']}
    # Reject detached scores, rather than silently count or overwrite them.
    for path in (handoff.output/'scores').glob('*.json'):
        if path.stem not in entries: raise ValueError('Detached score without committed checkpoint')
    return entries


def ledger(handoff, entries, runtime_sha):
    complete = set(entries)=={str(q) for q in range(1,1201)}
    seconds = sum(e['seconds'] for e in entries.values())
    result = {'format':FORMAT+'-ledger','status':'COMPLETE' if complete else 'RUNNING',
              'updated_utc':utc(),'runtime_sha256':runtime_sha,'scored_queries':len(entries),
              'scored_pairs':200*len(entries),'remaining_pairs':240000-200*len(entries),
              'model_scoring_seconds':seconds,'pairs_per_second':200*len(entries)/seconds if seconds else None,
              'queries':entries}
    write_atomic_replace(handoff.output/'ledger.json',result)
    return result


def runtime_contract(handoff):
    import torch
    from src.medrag.core.model_revision import require_cached_revision
    actual = {name:importlib.metadata.version(name) for name in VERSIONS}
    if actual != VERSIONS: raise ValueError(f'Pinned GPU dependency versions differ: {actual}')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.version.cuda!='12.4':
        raise ValueError('Exactly one CUDA 12.4 GPU required; CPU fallback prohibited')
    if 'A6000' not in torch.cuda.get_device_name(0): raise ValueError('A6000 runtime required')
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    settings = handoff.lock['score_contract']
    snapshot = require_cached_revision(settings['model'],settings['revision'])
    model_files = {str(p.relative_to(snapshot)):sha256_file(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    return torch, {'format':FORMAT+'-runtime','score_contract':settings,
        'score_contract_fingerprint':handoff.cache.fingerprint,'transport_lock_sha256':handoff.lock_sha,
        'versions':actual,'cuda':'12.4','gpu_name':torch.cuda.get_device_name(0),
        'gpu_uuid':os.environ.get('CUDA_VISIBLE_DEVICES'),
        'capability':list(torch.cuda.get_device_capability(0)),
        'dtype':'float16','activation':'Identity/raw logits','batch_size':32,
        'max_length':512,'num_threads':2,'num_interop_threads':1,
        'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32,
        'scorer':'unchanged BGEReranker.score/CrossEncoder.predict',
        'model_snapshot_files_sha256':model_files,
        'implementation_sha256':{name:sha256_file(ROOT/name) for name in CODE}}


def run(source,output,*,bounded=False,poll_seconds=30):
    output.mkdir(parents=True,exist_ok=True)
    guard=(output/'worker.lock').open('a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    handoff=Handoff(source,output)
    torch,runtime=runtime_contract(handoff)
    write_once(output/'runtime_lock.json',runtime)
    runtime_sha=sha256_file(output/'runtime_lock.json')
    entries=recover(handoff,runtime_sha)
    if not bounded:
        gate=read(output/'bounded_report.json')
        if gate.get('status')!='PASS' or gate.get('runtime_sha256')!=runtime_sha:
            raise ValueError('Continuous scoring requires matching bounded GPU PASS')
    ledger(handoff,entries,runtime_sha)
    model=None; newly=[]; wall_start=time.monotonic()
    while len(entries)<1200:
        available=handoff.available()
        pending=[q for q in available if str(q) not in entries]
        if bounded:
            boundary=max(handoff.resolver.inherited)+1
            chosen=[1,2,boundary,boundary+1,boundary+2]
            if any(q not in available for q in chosen): raise ValueError('Bounded mixed-lineage shards not ready')
            pending=[q for q in chosen if str(q) not in entries]
        if not pending:
            if bounded: break
            print(f'Waiting for CPU shards: scored={len(entries)}, ready={len(available)}',flush=True)
            time.sleep(poll_seconds);continue
        batch=pending[:5 if bounded else 50]
        handoff.batch(batch)
        if model is None:
            if torch.cuda.mem_get_info(0)[0]<12*1024**3: raise RuntimeError('Less than 12 GiB free GPU memory')
            from src.medrag.ranking.reranker import BGEReranker
            settings=handoff.lock['score_contract']
            model=BGEReranker(settings['model'],revision=settings['revision'],max_length=512,device='cuda:0')
            if next(model.model.model.parameters()).device.type!='cuda' or next(model.model.model.parameters()).dtype!=torch.float16:
                raise ValueError('Reranker did not load on GPU in fp16')
        for qid in batch:
            pair=read(handoff.staging/'pairs'/f'{qid}.json');mapping=read(handoff.staging/'lineage'/f'{qid}.json')
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            started=time.monotonic();cpu_start=time.process_time()
            values=model.score(pair['query'],[p['payload'] for p in pair['pairs']])
            torch.cuda.synchronize();seconds=time.monotonic()-started
            if len(values)!=200: raise ValueError('GPU score count differs')
            score={'format':'medrag-v4-p5-2-scores-v1','id':qid,
                   'pair_shard_sha256':mapping['transport_sha256'],
                   'contract_fingerprint':handoff.cache.fingerprint,
                   'scores':[{k:p[k] for k in ('cache_key','official_id','passage_id','input_sha256')}|{'score':float(v)}
                             for p,v in zip(pair['pairs'],values,strict=True)]}
            validate_score_shard(score,pair,mapping['transport_sha256'],handoff.cache.fingerprint)
            proof={'query_id':qid,'runtime_sha256':runtime_sha,'score_sha256':digest(score),
                   'fresh_model_pairs':200,'cache_hits':0,'device':'cuda:0','completed_utc':utc(),
                   'seconds':seconds,'cpu_seconds':time.process_time()-cpu_start,
                   'pairs_per_second':200/seconds,
                   'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                   'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
            checkpoint={'format':FORMAT+'-checkpoint','runtime_sha256':runtime_sha,
                        'lineage':mapping,'score':score,'proof':proof}
            validate_checkpoint(checkpoint,pair,mapping,handoff.cache.fingerprint,runtime_sha)
            commit(output,checkpoint)
            entries[str(qid)]={'checkpoint_sha256':sha256_file(output/'checkpoints'/f'{qid}.json'),
                              'score_sha256':digest(score),'source_sha256':mapping['source_sha256'],'seconds':seconds}
            newly.append(qid);status=ledger(handoff,entries,runtime_sha)
            print(f'scored {qid}: 200 real GPU pairs; {seconds:.3f}s; total={len(entries)}/1200',flush=True)
        if bounded: break
    if bounded:
        chosen=[1,2,max(handoff.resolver.inherited)+1,max(handoff.resolver.inherited)+2,max(handoff.resolver.inherited)+3]
        if any(str(q) not in entries for q in chosen): raise ValueError('Bounded coverage incomplete')
        seconds=sum(entries[str(q)]['seconds'] for q in chosen)
        report={'status':'PASS','runtime_sha256':runtime_sha,'query_ids':chosen,'pairs':1000,
                'scoring_seconds':seconds,'pairs_per_second':1000/seconds,
                'wall_seconds_this_invocation':time.monotonic()-wall_start,
                'first_real_gpu_test':True,'freshly_scored_this_invocation':newly}
        write_atomic_replace(output/'bounded_report.json',report)
        print(json.dumps(report),flush=True)
    return ledger(handoff,entries,runtime_sha)


def finalize_complete(source, output, destination):
    """Strict full coverage gate; reuse frozen CPU selection and BTC ZIP validation."""
    guard=(output/'worker.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    h=Handoff(source,output);runtime=read(output/'runtime_lock.json');runtime_sha=sha256_file(output/'runtime_lock.json')
    if runtime['score_contract']!=h.lock['score_contract']: raise ValueError('Scoring contract differs')
    for name,expected in runtime['implementation_sha256'].items():
        if sha256_file(ROOT/name)!=expected: raise ValueError(f'Implementation changed: {name}')
    entries=recover(h,runtime_sha)
    if set(entries)!={str(q) for q in range(1,1201)}:
        raise ValueError('Refusing incomplete production finalization: 1200/240000 required')
    if destination.resolve() in (source.resolve(),output.resolve()) or source.resolve() in destination.resolve().parents:
        raise ValueError('Finalization requires a separate namespace')
    destination.mkdir(parents=True,exist_ok=True)
    write_once(destination/'lock.json',h.lock)
    cache_path=destination/'key_contract.sqlite'
    ScoreCache(cache_path,h.lock['score_contract'],create=not cache_path.exists())
    hashes={}
    for qid in range(1,1201):
        pair=read(h.staging/'pairs'/f'{qid}.json');score=read(output/'scores'/f'{qid}.json')
        write_once(destination/'pairs'/f'{qid}.json',pair);write_once(destination/'scores'/f'{qid}.json',score)
        hashes[str(qid)]=sha256_file(destination/'pairs'/f'{qid}.json')
    write_once(destination/'pair_manifest.json',{'format':'medrag-v4-p5-2-pair-manifest-v1',
        'lock_sha256':sha256_file(destination/'lock.json'),'pair_shards_sha256':hashes})
    write_once(destination/'gpu_import_provenance.json',{'format':FORMAT+'-full-import',
        'runtime_sha256':runtime_sha,'query_count':1200,'pair_count':240000,'shards':entries})
    migration=read(ROOT/'configs/vibiomir/phase2_batch_migration_v1.json')
    return finalize(destination,load_config(CONFIG),h.lock,
        [{'id':q,'query':h.queries[q]} for q in range(1,1201)],h.index,ROOT/migration['official_corpus'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['bounded','worker','finalize'])
    p.add_argument('--source',type=Path,default=SOURCE);p.add_argument('--output',type=Path,default=OUTPUT)
    p.add_argument('--destination',type=Path)
    a=p.parse_args()
    if a.command=='finalize':
        if a.destination is None: p.error('--destination required for finalize')
        result=finalize_complete(a.source,a.output,a.destination)
    else: result=run(a.source,a.output,bounded=a.command=='bounded')
    print(json.dumps({k:v for k,v in result.items() if k!='queries'},indent=2),flush=True)
if __name__=='__main__': main()

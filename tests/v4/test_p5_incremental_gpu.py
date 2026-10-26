"""Focused contracts for the incremental boundary, without GPU/model inference."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.v4 import run_p5_incremental_gpu as gpu
from src.medrag.core.contracts import Source, Passage
from src.medrag.data.pair_lineage import PairLineageResolver, snapshot_source, write_once
from tests.v4.test_p5_fast_a0 import _lock, _contract, _shard


@pytest.fixture
def handoff(tmp_path):
    source=tmp_path/'original';source.mkdir()
    write_once(source/'lock.json',_lock())
    cache=gpu.ScoreCache(source/'key_contract.sqlite',_contract(),create=True)
    source_by_id={};passage_by_id={}
    def grounded_shard(qid, lock, cache):
        shard=_shard(qid,f'q{qid}',lock,cache)
        for doc in shard['candidates']:
            ident=doc['official_id'];e=doc['evidence'][0];text=e['text']
            s=Source('v',ident,f'u{ident}',f'Title {ident}',text,'fixture','raw')
            p=Passage(e['passage_id'],'v',ident,s.url,text,0,len(text),s.text_sha256,'en','fixture')
            e['end']=len(text);source_by_id[ident]=s;passage_by_id[p.passage_id]=p
        return shard
    write_once(source/'pairs/1.json',grounded_shard(1,source/'lock.json',cache))
    target=tmp_path/'fast';target.mkdir();lineage=snapshot_source(source,{1:'q1'})
    write_once(target/'lineage.json',lineage)
    lock=_lock();lock.update(format='medrag-v4-p5-fast-a0-lock-v1',lineage_sha256=gpu.sha256_file(target/'lineage.json'))
    write_once(target/'lock.json',lock)
    local=gpu.ScoreCache(target/'key_contract.sqlite',_contract(),create=True)
    write_once(target/'pairs/2.json',grounded_shard(2,target/'lock.json',local))
    (target/'pairs/.3.json.incomplete').write_text('incomplete')
    h=gpu.Handoff.__new__(gpu.Handoff);h.source=target;h.output=tmp_path/'gpu'
    h.resolver=PairLineageResolver(target);h.lock=gpu.transport_lock(h.resolver)
    h.queries={1:'q1',2:'q2'};h.staging=h.output/'staging'
    write_once(h.staging/'lock.json',h.lock);h.lock_sha=gpu.sha256_file(h.staging/'lock.json')
    h.cache=gpu.ScoreCache(h.staging/'key_contract.sqlite',_contract(),create=True)
    h.store=SimpleNamespace(sources_batch=lambda ids:{i:source_by_id[i] for i in ids},
        passages_batch=lambda ids:{i:passage_by_id[i] for i in ids})
    return h


def checkpoint(h,qid=1):
    pair,mapping=h.package(qid)
    score={'format':'medrag-v4-p5-2-scores-v1','id':qid,'pair_shard_sha256':mapping['transport_sha256'],
           'contract_fingerprint':h.cache.fingerprint,'scores':[{k:p[k] for k in ('cache_key','official_id','passage_id','input_sha256')}|{'score':1.0} for p in pair['pairs']]}
    c={'format':gpu.FORMAT+'-checkpoint','runtime_sha256':'runtime','lineage':mapping,'score':score,
       'proof':{'query_id':qid,'score_sha256':gpu.digest(score),'fresh_model_pairs':200,'device':'cuda:0','runtime_sha256':'runtime','seconds':1.0}}
    return pair,mapping,c


def test_mixed_lineage_transport_preserves_pairs_and_ignores_temporary(handoff):
    h=handoff;before={q:h.resolver.resolve(q)[0].read_bytes() for q in (1,2)}
    assert h.available()==[1,2]
    batch=h.batch([1,2]);assert batch['pair_count']==400
    for q in (1,2):
        original,origin,_=h.resolver.validate(q,f'q{q}')
        transported=gpu.read(h.staging/'pairs'/f'{q}.json')
        assert transported['pairs']==original['pairs']
        assert transported['candidates']==original['candidates']
        assert h.resolver.resolve(q)[0].read_bytes()==before[q]
        assert batch['shards'][str(q)]['origin']==('inherited' if q==1 else 'local')
    with pytest.raises(ValueError):h.batch([1,1])
    with pytest.raises(ValueError):h.batch([3])


def test_checkpoint_rejects_nonfinite_reordered_wrong_runtime_or_lineage(handoff):
    h=handoff;p,m,c=checkpoint(h)
    gpu.validate_checkpoint(c,p,m,h.cache.fingerprint,'runtime')
    changes=[]
    bad=copy.deepcopy(c);bad['score']['scores'][0]['score']=float('nan');changes.append(bad)
    bad=copy.deepcopy(c);bad['score']['scores'].reverse();changes.append(bad)
    bad=copy.deepcopy(c);bad['runtime_sha256']='other';changes.append(bad)
    bad=copy.deepcopy(c);bad['lineage']['source_sha256']='other';changes.append(bad)
    for bad in changes:
        with pytest.raises(ValueError):gpu.validate_checkpoint(bad,p,m,h.cache.fingerprint,'runtime')


def test_atomic_checkpoint_recovers_derived_files_and_rejects_conflicts(handoff):
    h=handoff;_,_,c=checkpoint(h)
    write_once(h.output/'checkpoints/1.json',c) # crash before score/proof publication
    entries=gpu.recover(h,'runtime');assert set(entries)=={'1'}
    before=(h.output/'scores/1.json').read_bytes()
    assert gpu.recover(h,'runtime')==entries
    assert (h.output/'scores/1.json').read_bytes()==before
    bad=copy.deepcopy(c);bad['score']['scores'][0]['score']=2.0
    with pytest.raises(ValueError):gpu.commit(h.output,bad)


def test_incomplete_finalization_refused_before_output_creation(handoff,monkeypatch):
    h=handoff
    write_once(h.output/'runtime_lock.json',{'score_contract':h.lock['score_contract'],'implementation_sha256':{}})
    monkeypatch.setattr(gpu,'Handoff',lambda *a:h)
    destination=h.output/'final_submission'
    with pytest.raises(ValueError,match='Refusing incomplete'):
        gpu.finalize_complete(h.source,h.output,destination)
    assert not destination.exists()

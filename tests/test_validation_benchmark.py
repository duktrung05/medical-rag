from pathlib import Path
import math
import pytest
from src.evaluation.ranking_metrics import summarize_rankings, validate_ranking
from scripts.benchmark_vimed_validation import CachedRetriever
from src.reranking.bge_reranker import BGEReranker


def test_multi_positive_metrics():
    samples=[dict(id='a',topic=0,relevant_chunks=['x','y'],relevant_docs=['d']),dict(id='b',topic=1,relevant_chunks=['z'],relevant_docs=['e'])]
    rows=[dict(id='a',results=[('x',2),('w',1),('y',0)],documents=[('d',2)],total_latency_ms=10),dict(id='b',results=[('w',2)],documents=[('d',2)],total_latency_ms=20)]
    m=summarize_rankings(samples,rows)
    assert m['recall_at_k']['1']==.25
    assert m['recall_at_k']['3']==.5
    assert m['mrr_at_100']==.5
    assert m['document_recall_at_k']['1']==.5
    assert m['zero_hit_queries_at_k']['100']==1
    assert m['by_topic']['0']['recall_at_k']['3']==1
    with pytest.raises(ValueError): summarize_rankings(samples,rows[::-1])


@pytest.mark.parametrize('ranking',[[('x',1),('x',0)],[('unknown',1)],[('x',math.nan)],[('x',math.inf)]])
def test_invalid_rankings(ranking):
    with pytest.raises(ValueError): validate_ranking(ranking,{'x'})


def test_cached_retrieval_and_reranker_change_rank():
    class Model:
        def predict(self,pairs,**kwargs):
            assert len(pairs)==2
            return [-1,3]
    cached=CachedRetriever({'query':[('a',2),('b',1)]})
    assert cached.search('query',1)==[('a',2)]
    reranker=BGEReranker(model=Model())
    result=reranker.rerank('query',[('a','wrong'),('b','positive')],top_k=2)
    assert result==[('b',3),('a',-1)]
    assert {x for x,_ in result}=={'a','b'}


def test_reranker_loader_pins_revision_and_raw_logits(monkeypatch):
    import sys, types, torch
    seen={}
    def fake(*args,**kwargs):
        seen.update(kwargs)
        return object()
    monkeypatch.setitem(sys.modules,'sentence_transformers',types.SimpleNamespace(CrossEncoder=fake))
    r=BGEReranker(revision='immutable',device='cpu')
    r._load_model()
    assert seen['revision']=='immutable'
    assert seen['model_kwargs']['torch_dtype']==torch.float32
    assert isinstance(seen['activation_fn'],torch.nn.Identity)
    with pytest.raises(RuntimeError): BGEReranker(device='cpu',precision='float16')._load_model()


def test_validation_maps_existing_corpus_and_rejects_test_overlap(tmp_path):
    import json,pandas as pd
    from scripts.prepare_vimed import digest
    from scripts.prepare_vimed_validation import prepare
    raw=tmp_path/'raw';raw.mkdir()
    corpus=tmp_path/'chunks.jsonl'
    cid='ctx_'+digest('context')
    corpus.write_text(json.dumps(dict(chunk_id=cid,doc_id='doc',text='context',chunk_index=0,language='vi'))+'\n')
    pd.DataFrame([dict(question_idx='val',question='query',answer='answer',context='context',topic=0),dict(question_idx='empty',question='q',answer='a',context=None,topic=0)]).to_parquet(raw/'validation-0.parquet')
    original=corpus.read_bytes()
    m=prepare(raw,corpus,tmp_path/'out')
    assert m['num_queries']==1
    assert m['skipped_empty_context_ids']==['empty']
    assert corpus.read_bytes()==original
    (tmp_path/'queries.jsonl').write_text(json.dumps(dict(id='val',query='test'))+'\n')
    with pytest.raises(ValueError,match='overlap'):prepare(raw,corpus,tmp_path/'out')


def test_dependency_rejects_modified_rankings(tmp_path):
    import json,hashlib
    from scripts.benchmark_vimed_validation import load_dependency, file_hash, CONFIGS
    directory=tmp_path/'bm25';directory.mkdir()
    path=directory/'rankings.jsonl'
    path.write_text(json.dumps(dict(id='a',results=[]))+'\n')
    manifest=dict(status='complete',corpus_sha256='corpus',samples_sha256='samples',rankings_sha256=file_hash(path),config_sha256=file_hash(Path(CONFIGS['bm25'])))
    (directory/'manifest.json').write_text(json.dumps(manifest))
    assert load_dependency(tmp_path,'bm25',[{'id':'a'}],'corpus','samples')[0]['id']=='a'
    path.write_text('{}\n')
    with pytest.raises(ValueError,match='checksum'):load_dependency(tmp_path,'bm25',[{'id':'a'}],'corpus','samples')
    with pytest.raises(ValueError,match='Incompatible'):load_dependency(tmp_path,'bm25',[{'id':'a'}],'other','samples')

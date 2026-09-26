import hashlib,json
from pathlib import Path
import pytest,yaml
from fastapi.testclient import TestClient
from src.inspector import create_app,CONFIGS

class Engine:
    def __init__(self,root):pass
    def search(self,stage,qid,text):
        return dict(results=[('b',2),('a',1)],prediction=dict(id=qid,relevant_chunks=['b'],relevant_docs=['doc']),latency_ms=3,setup_ms=0,scores={})

@pytest.fixture
def client(tmp_path):
    data=tmp_path/'data/vimed/validation';data.mkdir(parents=True)
    samples=[dict(id='q',query='câu hỏi',topic=1,answer='answer',relevant_chunks=['a'],relevant_docs=['doc'])]
    (data/'samples.jsonl').write_text(json.dumps(samples[0],ensure_ascii=False)+'\n',encoding='utf-8')
    (data/'manifest.json').write_text(json.dumps(dict(split='validation',num_chunks=2)))
    corpus=data.parent/'chunks.jsonl'
    corpus.write_text(''.join(json.dumps(dict(chunk_id=c,doc_id='doc',text=c,language='vi',chunk_index=i))+'\n' for i,c in enumerate(['a','b'])))
    configs=tmp_path/'configs';configs.mkdir()
    for name,file in CONFIGS.items():
        config=yaml.safe_load(Path('configs',file).read_text(encoding='utf-8'))
        config['corpus']=str(corpus)
        path=configs/file;path.write_text(yaml.safe_dump(config))
        directory=tmp_path/'outputs/vimed_validation'/name;directory.mkdir(parents=True)
        row=dict(id='q',query='câu hỏi',results=[['a',2],['b',1]],first_positive_rank=1,prediction=dict(id='q',relevant_chunks=['a'],relevant_docs=['doc']),total_latency_ms=12)
        rankings=directory/'rankings.jsonl';rankings.write_text(json.dumps(row)+'\n')
        (directory/'manifest.json').write_text(json.dumps(dict(status='complete',rankings_sha256=hashlib.sha256(rankings.read_bytes()).hexdigest(),config_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
        (directory/'metrics.json').write_text(json.dumps(dict(num_queries=1,recall_at_k={'10':1})))
    with TestClient(create_app(tmp_path,Engine)) as c:yield c


def test_comparison_and_gold(client):
    assert client.get('/health').json()['default_stage']=='rerank'
    assert client.get('/').status_code==200
    benchmark=client.get('/inspector/benchmark').json()
    assert benchmark['split']=='validation'
    assert benchmark['samples'][0]['ranks']['rerank']==1
    d=client.post('/inspector/search',json=dict(query='câu hỏi',sample_id='q',mode='benchmark',stages=list(CONFIGS),top_k=1)).json()
    assert len(d['comparisons'])==4
    assert d['gold'][0]['chunk_id']=='a'
    assert all(r['recall_at_k']==1 and r['results'][0]['is_positive'] for r in d['comparisons'])
    d=client.post('/inspector/search',json=dict(query='câu hỏi',sample_id='q',mode='live',top_k=1)).json()
    assert d['comparisons'][0]['recall_at_k']==0
    assert d['comparisons'][0]['first_positive_rank']==2


def test_edited_query_has_no_gold(client):
    d=client.post('/inspector/search',json=dict(query='edited',sample_id='q',mode='live')).json()
    assert not d['labeled'] and d['gold']==[] and d['comparisons'][0]['recall_at_k'] is None
    assert client.post('/inspector/search',json=dict(query='edited',sample_id='q',mode='benchmark')).status_code==422
    assert client.post('/inspector/search',json=dict(query='  ')).status_code==422
    assert client.post('/inspector/search',json=dict(query='q',stages=[])).status_code==422
    assert client.post('/inspector/search',json=dict(query='q',stages=['bad'])).status_code==422


def test_labels_persist_latest_and_do_not_alter_gold(client):
    body=dict(query='câu hỏi',sample_id='q',chunk_id='b',label='relevant',note='review')
    assert client.post('/inspector/labels',json=body).status_code==200
    body['label']='irrelevant'
    assert client.post('/inspector/labels',json=body).status_code==200
    rows=client.get('/inspector/labels',params={'query':'câu hỏi'}).json()
    assert len(rows)==1 and rows[0]['label']=='irrelevant'
    body['chunk_id']='unknown'
    assert client.post('/inspector/labels',json=body).status_code==404
    assert client.get('/inspector/benchmark').json()['samples'][0]['ranks']['rerank']==1

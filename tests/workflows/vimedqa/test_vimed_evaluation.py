"""Regression coverage for the full ViMed evaluation pipeline."""
import json
from scripts.workflows.vimedqa import evaluate_vimed

def test_vimed_evaluator_uses_reranked_candidates(tmp_path,monkeypatch):
    data=tmp_path/'data';data.mkdir()
    (data/'samples.jsonl').write_text(json.dumps(dict(id='q',query='QUESTION??',topic=0,relevant_chunks=['positive']))+'\n')
    (data/'manifest.json').write_text('{}')
    class Pipeline:
        def run_query(self,qid,text):
            assert text=='question?'
            self.last_candidate_scores={'wrong':{'selection_score':-1,'selection_eligible':True},'positive':{'selection_score':3,'selection_eligible':True},'outside':{'selection_score':None,'selection_eligible':False}}
        @property
        def retriever(self):raise AssertionError('Evaluator bypassed reranker')
    monkeypatch.setattr(evaluate_vimed,'build_pipeline',lambda c:Pipeline())
    monkeypatch.setattr(evaluate_vimed,'load_config',lambda c:type('Config',(),{'backend':'hybrid'})())
    out=tmp_path/'out'
    monkeypatch.setattr('sys.argv',['evaluate_vimed','--data',str(data),'--output',str(out)])
    evaluate_vimed.main()
    metrics=json.loads((out/'metrics.json').read_text())
    assert metrics['recall_at_k']['1']==1
    assert metrics['mrr_at_100']==1
def test_prepare_vimed_skips_nan_contexts(tmp_path,monkeypatch):
    import pandas as pd
    from scripts.workflows.vimedqa import prepare_vimed
    raw=tmp_path/'raw';raw.mkdir()
    for split in ['train','validation','test']:
        pd.DataFrame([dict(question_idx=split,question='query',answer='answer',topic=1,context='context',article_url='url',title='title'),dict(question_idx=split+'_empty',question='query',answer='answer',topic=1,context=float('nan'),article_url='',title='')]).to_parquet(raw/(split+'-0.parquet'))
    monkeypatch.setattr(prepare_vimed.BM25Retriever,'build_index',lambda *a,**k:'hash')
    out=tmp_path/'out'
    monkeypatch.setattr('sys.argv',['prepare_vimed','--raw',str(raw),'--output',str(out),'--index',str(tmp_path/'index')])
    prepare_vimed.main()
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['num_chunks']==1 and manifest['num_test_queries']==1
    assert manifest['skipped_empty_context']==dict(train=1,validation=1,test=1)

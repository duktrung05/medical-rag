"""Local comparison UI for ViMed retrieval, benchmark replay and human labels."""
import hashlib
import json
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from src.adapters import adapt_query
from src.config import load_pipeline_config
from src.data.loader import DataLoader
from src.data.schema import QueryRecord
from src.service import build_pipeline
from src.retrieval.hybrid import HybridRetriever

STAGES = ('bm25', 'dense', 'hybrid', 'rerank')
CONFIGS = dict(zip(STAGES, ('vimed_bm25.yaml','vimed_dense_bge_m3.yaml','vimed_hybrid.yaml','vimed_hybrid_rerank.yaml')))

class InspectRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    sample_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=100)
    stages: list[Literal['bm25','dense','hybrid','rerank']] = Field(default_factory=lambda:['rerank'],min_length=1,max_length=4)
    mode: Literal['live','benchmark'] = 'live'

class LabelRequest(BaseModel):
    query: str = Field(min_length=1,max_length=4000)
    sample_id: str | None = None
    chunk_id: str
    label: Literal['relevant','irrelevant','unsure']
    note: str = Field(default='',max_length=2000)

class LiveEngine:
    """Lazy pipelines sharing one dense encoder and one sparse index."""
    def __init__(self, root):
        self.root, self.pipelines = root, {}
    def pipeline(self, stage):
        if stage not in self.pipelines:
            config=load_pipeline_config(self.root/'configs'/CONFIGS[stage])
            override=None
            if stage in ('hybrid','rerank'):
                sparse=self.pipeline('bm25').retriever
                dense=self.pipeline('dense').retriever
                override=(HybridRetriever(sparse,dense,rrf_k=config.fusion.rrf_k,
                                          sparse_top_k=config.retrieval.sparse.top_k,
                                          dense_top_k=config.retrieval.dense.top_k),config.fusion.top_k)
            self.pipelines[stage]=build_pipeline(config,retriever_override=override)
        return self.pipelines[stage]
    def search(self, stage, qid, text):
        start=time.perf_counter(); pipeline=self.pipeline(stage)
        load_ms=(time.perf_counter()-start)*1000
        start=time.perf_counter(); prediction=pipeline.run_query(qid,text)
        elapsed=(time.perf_counter()-start)*1000
        ranking=sorted([(cid,s['selection_score']) for cid,s in pipeline.last_candidate_scores.items() if s['selection_eligible']],key=lambda x:(-x[1],x[0]))
        return dict(results=ranking,prediction=prediction.model_dump(),latency_ms=elapsed,setup_ms=load_ms,
                    scores={cid:dict(s) for cid,s in pipeline.last_candidate_scores.items()})

def read_rows(path):
    return [json.loads(x) for x in path.read_text(encoding='utf-8').splitlines() if x.strip()]

def create_app(root=None, engine_factory=LiveEngine):
    root=Path(root or Path(__file__).resolve().parents[1])
    data=root/'data/vimed/validation'; output=root/'outputs/vimed_validation'
    lock=threading.Lock()
    @asynccontextmanager
    async def lifespan(app):
        config=load_pipeline_config(root/'configs/vimed_hybrid_rerank.yaml')
        app.state.chunks={c.chunk_id:c.model_dump() for c in DataLoader.load_chunks(config.corpus)}
        app.state.samples={s['id']:s for s in read_rows(data/'samples.jsonl')}
        metadata=root/'data/vimed/metadata.json'
        app.state.metadata=json.loads(metadata.read_text(encoding='utf-8')) if metadata.exists() else {}
        app.state.cache={};app.state.metrics={}
        for stage in STAGES:
            directory=output/stage
            if not (directory/'manifest.json').exists():continue
            manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
            path=directory/'rankings.jsonl'
            if manifest['status']!='complete' or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['rankings_sha256']:
                raise ValueError(f'Invalid benchmark cache: {stage}')
            if hashlib.sha256((root/'configs'/CONFIGS[stage]).read_bytes()).hexdigest()!=manifest['config_sha256']:
                raise ValueError(f'Benchmark config changed: {stage}')
            rows=read_rows(path)
            if [r['id'] for r in rows]!=list(app.state.samples):raise ValueError('Benchmark sample IDs differ')
            app.state.cache[stage]={r['id']:r for r in rows}
            app.state.metrics[stage]=json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        app.state.engine=engine_factory(root)
        yield
    app=FastAPI(title='ViMed Retrieval Lab',lifespan=lifespan)
    @app.get('/')
    def home():return FileResponse(Path(__file__).parent/'web/inspector.html')
    @app.get('/health')
    def health():return dict(status='ok',default_stage='rerank',split='validation',chunks=len(app.state.chunks),queries=len(app.state.samples))
    @app.get('/inspector/benchmark')
    def benchmark():
        return dict(split='validation',metrics=app.state.metrics,default_stage='rerank',
                    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8')),
                    samples=[dict(id=s['id'],query=s['query'],topic=s['topic'],ranks={stage:app.state.cache.get(stage,{}).get(s['id'],{}).get('first_positive_rank') for stage in STAGES}) for s in app.state.samples.values()])
    @app.post('/inspector/search')
    def search(request:InspectRequest):
        if not request.query.strip():raise HTTPException(422,'Query must not be blank')
        sample=app.state.samples.get(request.sample_id)
        labeled=sample is not None and sample['query'].strip()==request.query.strip()
        if request.mode=='benchmark' and not labeled:raise HTTPException(422,'Benchmark mode requires an unchanged validation question')
        query=adapt_query(QueryRecord(id=request.sample_id or 'manual',query=request.query))
        positives=set(sample['relevant_chunks']) if labeled else set()
        comparisons=[]
        with lock:
            for stage in dict.fromkeys(request.stages):
                if request.mode=='benchmark':
                    cached=app.state.cache.get(stage,{}).get(request.sample_id)
                    if cached is None:raise HTTPException(409,f'No completed benchmark for {stage}')
                    result=dict(results=cached['results'],prediction=cached['prediction'],latency_ms=cached['total_latency_ms'],setup_ms=0,scores={})
                else:
                    try:result=app.state.engine.search(stage,query.query_id,query.text)
                    except Exception as exc:raise HTTPException(503,f'{stage} could not run: {type(exc).__name__}: {exc}') from exc
                ranking=result['results']; selected=set(result['prediction']['relevant_chunks'])
                rows=[]
                for rank,(cid,score) in enumerate(ranking[:request.top_k],1):
                    rows.append(dict(**app.state.chunks[cid],**app.state.metadata.get(cid,{}),rank=rank,score=score,
                                     scores=result['scores'].get(cid,{'selection_score':score}),is_positive=cid in positives,selected=cid in selected))
                positive_rank=next((i for i,(cid,_) in enumerate(ranking,1) if cid in positives),None)
                recall=len({cid for cid,_ in ranking[:request.top_k]}&positives)/len(positives) if positives else None
                comparisons.append(dict(stage=stage,results=rows,first_positive_rank=positive_rank,recall_at_k=recall,
                                        latency_ms=result['latency_ms'],setup_ms=result['setup_ms'],prediction=result['prediction']))
        return dict(query=query.text,mode=request.mode,labeled=labeled,comparisons=comparisons,
                    gold=[dict(**app.state.chunks[cid],**app.state.metadata.get(cid,{})) for cid in sorted(positives)],answer=sample['answer'] if labeled else None)
    labels_path=output/'human_labels.jsonl'
    @app.get('/inspector/labels')
    def labels(query:str):
        with lock:rows=read_rows(labels_path) if labels_path.exists() else []
        normalized=adapt_query(QueryRecord(id='manual',query=query)).text
        latest={r['chunk_id']:r for r in rows if r['normalized_query']==normalized}
        return list(latest.values())
    @app.post('/inspector/labels')
    def label(request:LabelRequest):
        if not request.query.strip():raise HTTPException(422,'Query must not be blank')
        if request.chunk_id not in app.state.chunks:raise HTTPException(404,'Unknown chunk')
        sample=app.state.samples.get(request.sample_id)
        row=request.model_dump()
        row['sample_id']=request.sample_id if sample and sample['query'].strip()==request.query.strip() else None
        row.update(normalized_query=adapt_query(QueryRecord(id='label',query=request.query)).text,created_at=time.time())
        with lock:
            labels_path.parent.mkdir(parents=True,exist_ok=True)
            with labels_path.open('a',encoding='utf-8') as f:f.write(json.dumps(row,ensure_ascii=False)+'\n')
        return dict(status='saved',label=row)
    return app

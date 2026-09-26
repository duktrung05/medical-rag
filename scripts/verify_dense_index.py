"""Verify full dense cache integrity and real model CLS/mean encoding parity."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.config import load_pipeline_config
from src.data.loader import DataLoader
from src.indexing.sparse_index import corpus_sha256
from src.retrieval.factory import create_dense_retriever


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/vimed_dense_bge_m3.yaml')
    parser.add_argument('--output', default='outputs/vimed_dense_build/verification.json')
    args = parser.parse_args()
    started = time.monotonic()
    config = load_pipeline_config(args.config)
    chunks = DataLoader.load_chunks(config.corpus)
    retriever = create_dense_retriever(config.retrieval.dense, expected_corpus_hash=corpus_sha256(chunks))
    parameter_dtype = next(retriever.model.parameters()).dtype
    if config.retrieval.dense.precision == 'float16':
        assert retriever.device.type == 'cuda' and parameter_dtype == torch.float16
    assert len(retriever.chunk_ids) == len(chunks)
    assert set(retriever.chunk_ids) == {c.chunk_id for c in chunks}
    norms = np.linalg.norm(retriever.passage_embeddings, axis=1)
    assert np.isfinite(retriever.passage_embeddings).all()
    if config.retrieval.dense.normalize_embeddings:
        np.testing.assert_allclose(norms, 1., atol=0.002)
    lookup = {c.chunk_id: c for c in chunks}
    positions = [0, len(chunks)//2, len(chunks)-1]
    texts = [lookup[retriever.chunk_ids[i]].text for i in positions]
    actual = retriever._encode([retriever.passage_prefix + text for text in texts])
    cached = retriever.passage_embeddings[positions]
    np.testing.assert_allclose(actual, cached, atol=0.002, rtol=0.01)

    # Independent raw-transformer reference to the published BGE CLS convention.
    queries = ['đau đầu kéo dài', 'tác dụng phụ của thuốc', 'heart disease']
    query_texts = [retriever.query_prefix + text for text in queries]
    tokens = retriever.tokenizer(query_texts, padding=True, truncation=True,
                                 max_length=retriever.max_length, return_tensors='pt')
    tokens = {k: v.to(retriever.device) for k, v in tokens.items()}
    with torch.inference_mode():
        hidden = retriever.model(**tokens).last_hidden_state
        if retriever.pooling == 'cls':
            reference = hidden[:, 0]
        else:
            mask = tokens['attention_mask'].unsqueeze(-1).to(hidden.dtype)
            reference = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if config.retrieval.dense.normalize_embeddings:
            reference = torch.nn.functional.normalize(reference, p=2, dim=1)
    encoded = retriever._encode(query_texts)
    np.testing.assert_allclose(encoded, reference.float().cpu().numpy(), atol=0.002, rtol=0.01)
    batch = retriever.search_batch(queries, top_k=5)
    for query, results in zip(queries, batch, strict=True):
        single = retriever.search(query, top_k=5)
        # FP16 batch kernels can introduce tiny differences; compare identities and scores.
        assert [cid for cid, _ in results] == [cid for cid, _ in single]
        np.testing.assert_allclose([s for _, s in results], [s for _, s in single], atol=0.002)
    report = dict(status='passed', passages=len(chunks), dimension=retriever.passage_embeddings.shape[1],
                  encoder_profile=retriever.encoder_profile, pooling=retriever.pooling,
                  query_prefix=retriever.query_prefix, passage_prefix=retriever.passage_prefix,
                  precision=retriever.precision, device=str(retriever.device),
                  model_parameter_dtype=str(parameter_dtype),
                  finite=True, normalized=bool(np.allclose(norms, 1., atol=0.002)),
                  cache_reencode_max_abs_error=float(np.max(np.abs(actual-cached))),
                  reference_max_abs_error=float(np.max(np.abs(encoded-reference.float().cpu().numpy()))),
                  batch_single_consistent=True, verification_seconds=time.monotonic()-started,
                  queries=[dict(query=q, results=r) for q, r in zip(queries, batch, strict=True)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()

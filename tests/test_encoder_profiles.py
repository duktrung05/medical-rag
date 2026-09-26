"""Regression tests for model conventions, pooling and stale index rejection."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

from src.config import DenseRetrievalConfig
from src.retrieval.dense import DenseRetriever
from src.retrieval.dense_encoding import encode_texts
from tests.test_dense_retriever import _fake_encoder, _fixture_chunks


@pytest.mark.parametrize('model,pooling,query,passage', [
    ('BAAI/bge-m3', 'cls', '', ''),
    ('intfloat/multilingual-e5-small', 'mean', 'query: ', 'passage: '),
])
def test_model_specific_defaults(model, pooling, query, passage):
    config = DenseRetrievalConfig(model_name=model)
    assert (config.pooling, config.query_prefix, config.passage_prefix) == (pooling, query, passage)


@pytest.mark.parametrize('override', [dict(pooling='mean'), dict(query_prefix='query: '), dict(passage_prefix='passage: ')])
def test_bge_rejects_e5_conventions(override):
    with pytest.raises(ValidationError, match='bge_m3 requires'):
        DenseRetrievalConfig(model_name='BAAI/bge-m3', **override)


def test_cls_and_masked_mean_are_distinct_and_ignore_padding():
    def tokenizer(texts, **kwargs):
        return dict(input_ids=torch.zeros((len(texts), 3), dtype=torch.long),
                    attention_mask=torch.tensor([[1, 1, 0]] * len(texts)))

    class Model:
        def eval(self):
            return self

        def __call__(self, **kwargs):
            return SimpleNamespace(last_hidden_state=torch.tensor([[[2., 0.], [0., 4.], [99., 99.]]]))

    common = dict(texts=['example'], tokenizer=tokenizer, model=Model(), batch_size=1,
                  max_length=8, normalize=False, device=torch.device('cpu'))
    np.testing.assert_array_equal(encode_texts(**common, pooling='cls'), [[2., 0.]])
    np.testing.assert_array_equal(encode_texts(**common, pooling='mean'), [[1., 2.]])
    normalized = encode_texts(**{**common, 'normalize': True}, pooling='cls')
    np.testing.assert_array_equal(normalized, [[1., 0.]])


def test_bge_build_search_and_manifest_share_profile(tmp_path, monkeypatch):
    monkeypatch.setattr('src.retrieval.dense.load_encoder', _fake_encoder)
    calls = []

    def record(texts, *args, **kwargs):
        calls.append((list(texts), kwargs['pooling']))
        return encode_texts(texts, *args, **kwargs)

    monkeypatch.setattr('src.retrieval.dense.encode_texts', record)
    config = DenseRetrievalConfig(model_name='BAAI/bge-m3', index_path=tmp_path / 'index')
    manifest = DenseRetriever.build_index(_fixture_chunks(), config)
    assert manifest['pooling'] == 'cls'
    assert manifest['encoder_profile'] == 'bge_m3'
    assert manifest['query_prefix'] == manifest['passage_prefix'] == ''
    retriever = DenseRetriever(model_name=config.model_name, index_path=config.index_path)
    assert retriever.search('heart', 1) == [('c-z', 1.)]
    assert calls == [(['lung health', 'heart disease'], 'cls'), (['heart'], 'cls')]

    manifest['pooling'] = 'attention_mask_mean'
    manifest['format_version'] = 1
    (config.index_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr('src.retrieval.dense.load_encoder', lambda *a, **kw: pytest.fail('must reject before model load'))
    with pytest.raises(ValueError, match='pooling mismatch.*rebuild'):
        DenseRetriever(model_name=config.model_name, index_path=config.index_path)


def test_e5_build_and_query_keep_required_prefixes(tmp_path, monkeypatch):
    monkeypatch.setattr('src.retrieval.dense.load_encoder', _fake_encoder)
    original = encode_texts
    calls = []

    def record(texts, *args, **kwargs):
        calls.append((texts, kwargs['pooling']))
        return original(texts, *args, **kwargs)

    monkeypatch.setattr('src.retrieval.dense.encode_texts', record)
    config = DenseRetrievalConfig(model_name='intfloat/multilingual-e5-small', index_path=tmp_path / 'index')
    DenseRetriever.build_index(_fixture_chunks(), config)
    retriever = DenseRetriever(model_name=config.model_name, index_path=config.index_path)
    retriever.search('heart', 1)
    assert all(text.startswith('passage: ') for text in calls[0][0])
    assert calls[1] == (['query: heart'], 'mean')


def test_profile_and_precision_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr('src.retrieval.dense.load_encoder', _fake_encoder)
    config = DenseRetrievalConfig(model_name='BAAI/bge-m3', index_path=tmp_path / 'index')
    manifest = DenseRetriever.build_index(_fixture_chunks(), config)
    manifest['precision'] = 'float16'
    (config.index_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='precision'):
        DenseRetriever(model_name=config.model_name, index_path=config.index_path)


def test_legacy_float32_cache_cannot_silently_mix_with_fp16_query(tmp_path, monkeypatch):
    monkeypatch.setattr('src.retrieval.dense.load_encoder', _fake_encoder)
    config = DenseRetrievalConfig(model_name='intfloat/multilingual-e5-small', index_path=tmp_path / 'index')
    manifest = DenseRetriever.build_index(_fixture_chunks(), config)
    manifest['format_version'] = 1
    manifest.pop('precision')
    manifest.pop('encoder_profile')
    (config.index_path / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    monkeypatch.setattr('src.retrieval.dense.load_encoder', lambda *a, **kw: pytest.fail('must reject before model load'))
    with pytest.raises(ValueError, match='precision mismatch.*rebuild'):
        DenseRetriever(model_name=config.model_name, index_path=config.index_path, precision='float16')

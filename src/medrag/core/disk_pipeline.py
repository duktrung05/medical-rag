"""V4 orchestration over verified disk-backed stores and lexical indexes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.medrag.core.pipeline import Pipeline
from src.medrag.data.disk_store import DiskCorpusStore, PassageLookup, SourceLookup
from src.medrag.evidence.selector import SelectionPolicy
from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.disk_dense import DiskDense


class DiskPipeline(Pipeline):
    def __init__(self, index: Path, config: dict, *, reranker=None, dense=None, score_cache=None):
        manifest_path = index / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != "medrag-v4-disk-index-v1" or manifest.get("status") != "complete":
            raise ValueError("Incomplete or unsupported V4 disk index")
        fingerprint = str(manifest["fingerprint"])
        self.corpus_version = str(manifest["corpus_version"])
        for name, expected in manifest["files"].items():
            with (index / name).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            if digest != expected:
                raise ValueError(f"V4 disk index checksum mismatch: {name}")
        store = DiskCorpusStore(index / "corpus.sqlite", self.corpus_version, fingerprint)
        if store.counts() != (manifest["source_count"], manifest["passage_count"]):
            raise ValueError("V4 disk store count mismatch")
        self.sources = SourceLookup(store)
        self.passages = PassageLookup(store)
        self.doc_bm25 = DiskBM25(index / "documents.sqlite", "document_bm25", fingerprint)
        self.passage_bm25 = DiskBM25(index / "passages.sqlite", "passage_bm25", fingerprint)
        if int(self.doc_bm25._check(require_complete=True)["rows"]) != len(self.sources):
            raise ValueError("V4 document BM25 count mismatch")
        if int(self.passage_bm25._check(require_complete=True)["rows"]) != len(self.passages):
            raise ValueError("V4 passage BM25 count mismatch")
        dense_config = config.get("dense", {})
        reranker_config = config.get("reranker", {})
        if dense_config.get("enabled", False) and dense is None:
            dense_contract = manifest.get("dense")
            if not isinstance(dense_contract, dict):
                raise ValueError("Dense is enabled but this disk index has no dense search backend")
            expected = {key: dense_config.get(key) for key in
                        ("model", "revision", "document_max_length", "query_max_length")}
            if any(dense_contract.get(key) != value for key, value in expected.items()):
                raise ValueError("Dense index model/input configuration mismatch")
            from src.medrag.retrieval.bge_encoder import BGEM3Encoder

            encoder = BGEM3Encoder(str(dense_config["model"]),
                                   revision=str(dense_config["revision"]),
                                   document_max_length=int(dense_config["document_max_length"]),
                                   query_max_length=int(dense_config["query_max_length"]),
                                   device=dense_config.get("device"))
            dense = DiskDense(index / "dense", encoder, str(dense_contract["fingerprint"]))
            if dense.count() != len(self.passages):
                raise ValueError("Dense row count differs from eligible passage count")
            dense._check(require_complete=True)
        if reranker_config.get("enabled", False) and reranker is None:
            reranker_contract = manifest.get("reranker")
            if not isinstance(reranker_contract, dict):
                raise ValueError("Reranker is enabled but no reranker was supplied")
            expected = {key: reranker_config.get(key) for key in ("model", "revision", "max_length")}
            if any(reranker_contract.get(key) != value for key, value in expected.items()):
                raise ValueError("Reranker index/run configuration mismatch")
            from src.medrag.ranking.reranker import BGEReranker

            reranker = BGEReranker(str(reranker_config["model"]),
                                   revision=str(reranker_config["revision"]),
                                   max_length=int(reranker_config["max_length"]),
                                   device=reranker_config.get("device"))
        self.dense = dense
        self.reranker = reranker
        self.score_cache = score_cache
        retrieval = config.get("retrieval", {})
        selection = config.get("selection", {})
        self.document_candidate_limit = int(retrieval.get("document_candidate_limit", 400))
        self.passage_candidate_limit = int(retrieval.get("passage_candidate_limit", 200))
        self.fused_candidate_limit = int(retrieval.get("fused_candidate_limit", 400))
        parent_limit = retrieval.get("fused_parent_limit")
        self.fused_parent_limit = int(parent_limit) if parent_limit is not None else None
        self.rrf_k = int(retrieval.get("rrf_k", 60))
        self.rerank_candidate_limit = int(retrieval.get("rerank_candidate_limit", 200))
        self.rerank_passages_per_document = int(retrieval.get("rerank_passages_per_document", 3))
        self.rerank_allocation = str(retrieval.get("rerank_allocation", "document_order"))
        self.selection_policy = SelectionPolicy(
            max_docs=int(selection.get("max_docs", 200)),
            max_chunks=int(selection.get("max_chunks", 60)),
            chunks_per_doc=int(selection.get("chunks_per_doc", 1)),
            document_selection=str(selection.get("document_selection", "with_chunks")),
            chunk_context=str(selection.get("chunk_context", "passage")),
            context_window=int(selection.get("context_window", 1)),
            context_max_characters=int(selection.get("context_max_characters", 4096)),
        )
        if min(self.document_candidate_limit, self.passage_candidate_limit,
               self.fused_candidate_limit, self.rrf_k, self.rerank_candidate_limit,
               self.rerank_passages_per_document) <= 0:
            raise ValueError("V4 retrieval budgets must be positive")
        if self.fused_parent_limit is not None and self.fused_parent_limit <= 0:
            raise ValueError("V4 fused parent limit must be positive")
        if self.rerank_allocation not in {"document_order", "parent_first"}:
            raise ValueError("Unsupported rerank allocation")

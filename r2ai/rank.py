"""Dense retrieval + cross-encoder reranking, then emit the competition JSON.

Scoring is F2-macro (recall weighted 4x precision) averaged over document level
and chunk level, so the selection sizes K_docs / K_chunks are the main tunable
levers. They are flags rather than constants because the public phase allows 10
submissions a day, which is the only labelled signal available.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from r2ai.evidence_io import fingerprint, write_ranking_cache
from r2ai.evidence_selector import (
    EvidenceCandidate,
    QueryRanking,
    SelectionConfig,
)
from r2ai.select_evidence import (
    add_selection_arguments,
    config_from_args,
    replay,
)
from r2ai.taxonomy import DEFAULT_CONFIG, Taxonomy, metadata_overlap, rrf_rerank

RAW = ROOT / "data/raw/vibiomir"
OUT = ROOT / "data/vibiomir"
SUB = ROOT / "outputs/submissions"

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
EMBED_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
RERANK_REVISION = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"


def _embedder(device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(
        EMBED_MODEL, revision=EMBED_REVISION, device=device, model_kwargs={"torch_dtype": torch.float16}
    )


def embed_chunks(texts: list[str], titles: list[str], batch: int, device: str) -> np.ndarray:
    import torch

    model = _embedder(device)
    model.max_seq_length = 512
    # Prepending the title gives short chunks the topical anchor they otherwise lack.
    payload = [f"{t}\n{c}" if t else c for t, c in zip(titles, texts)]
    vecs = model.encode(
        payload, batch_size=batch, normalize_embeddings=True,
        convert_to_numpy=True, show_progress_bar=True,
    )
    del model
    torch.cuda.empty_cache()
    return vecs.astype(np.float32)


def embed_queries(queries: list[str], device: str) -> np.ndarray:
    import torch

    model = _embedder(device)
    model.max_seq_length = 512
    vecs = model.encode(queries, batch_size=32, normalize_embeddings=True,
                        convert_to_numpy=True, show_progress_bar=True)
    del model
    torch.cuda.empty_cache()
    return vecs.astype(np.float32)


Shortlist = list[tuple[np.ndarray, np.ndarray]]


def dense_shortlist(
    chunk_vecs: np.ndarray, query_vecs: np.ndarray, qids: list[int],
    doc_of: np.ndarray, pool: int, allowed_per_query: dict[int, set[int]] | None,
    device: str,
) -> Shortlist:
    """Top-`pool` chunks per query, as ragged (indices, scores) pairs.

    With a stage-1 candidate set each query is scored only against its own
    documents, which keeps this exact while touching a tiny slice of the corpus.
    Without one, documents are scored in blocks against a running top-k so peak
    memory follows the block size, not the corpus size.
    """
    if pool <= 0:
        raise ValueError("pool must be positive")
    if len(chunk_vecs) == 0:
        return [(np.empty(0, np.int64), np.empty(0, np.float32)) for _ in qids]
    import torch

    query_t = torch.from_numpy(query_vecs.astype(np.float16)).to(device)
    out: Shortlist = []

    if allowed_per_query is not None:
        chunks_by_doc: dict[int, list[int]] = defaultdict(list)
        for i, d in enumerate(doc_of.tolist()):
            chunks_by_doc[d].append(i)
        for qi, qid in enumerate(qids):
            idx = np.fromiter(
                (i for d in allowed_per_query.get(qid, ()) for i in chunks_by_doc.get(d, ())),
                dtype=np.int64,
            )
            if idx.size == 0:
                out.append((np.empty(0, np.int64), np.empty(0, np.float32)))
                continue
            sub = torch.from_numpy(chunk_vecs[idx].astype(np.float16)).to(device)
            sims = (sub @ query_t[qi]).float()
            k = min(pool, idx.size)
            scores, order = torch.topk(sims, k=k)
            out.append((idx[order.cpu().numpy()], scores.cpu().numpy()))
            del sub, sims
        torch.cuda.empty_cache()
        return out

    block = 2_000_000
    pool_n = min(pool, len(chunk_vecs))
    best_s = torch.full((len(qids), pool_n), -1e4, dtype=torch.float16, device=device)
    best_i = torch.zeros((len(qids), pool_n), dtype=torch.int64, device=device)
    for start in range(0, len(chunk_vecs), block):
        stop = min(start + block, len(chunk_vecs))
        part = torch.from_numpy(chunk_vecs[start:stop].astype(np.float16)).to(device)
        sims = query_t @ part.T
        k = min(pool_n, stop - start)
        scores, idx = torch.topk(sims, k=k, dim=1)
        best_s, order = torch.topk(torch.cat([best_s, scores], 1), k=pool_n, dim=1)
        best_i = torch.gather(torch.cat([best_i, idx + start], 1), 1, order)
        del part, sims
    torch.cuda.empty_cache()
    idx_np, score_np = best_i.cpu().numpy(), best_s.float().cpu().numpy()
    return [(idx_np[qi], score_np[qi]) for qi in range(len(qids))]


@dataclass
class RankedPool:
    indices: np.ndarray
    retrieval_scores: np.ndarray
    rerank_scores: list[float | None]


def cross_encoder_rerank(
    shortlist: Shortlist, qtexts: list[str], texts: list[str], titles: list[str],
    device: str, batch: int, rerank_top: int = 0, revision: str = RERANK_REVISION,
    model_metadata: dict | None = None,
) -> list[RankedPool]:
    """Rescore shortlisted pairs in one flat predict() call.

    Per-call model overhead dominates when this is done query by query.

    `rerank_top` caps how deep the cross-encoder goes. The pool has to be large
    to support K in the hundreds, but F2 at that depth is driven by recall, so
    the tail ordering barely matters; reranking only the head keeps the cost
    proportional to what actually affects the score. Reranked items are kept
    ahead of the dense-ordered tail.
    """
    if rerank_top < 0 or batch <= 0:
        raise ValueError("rerank_top must be nonnegative and batch positive")
    if len(shortlist) != len(qtexts):
        raise ValueError("Shortlist and query counts differ")
    heads = [idx[:rerank_top] if rerank_top else idx for idx, _ in shortlist]
    pairs = [
        [qtexts[qi], f"{titles[c]}\n{texts[c]}" if titles[c] else texts[c]]
        for qi, head in enumerate(heads) for c in head
    ]
    print(f"reranking {len(pairs):,} pairs", flush=True)
    if not pairs:
        return [RankedPool(idx, dense_scores, [None] * len(idx)) for idx, dense_scores in shortlist]
    import torch
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(RERANK_MODEL, device=device, max_length=512, revision=revision,
                            model_kwargs={"torch_dtype": torch.float16}, activation_fn=torch.nn.Identity())
    if model_metadata is not None:
        model_metadata["resolved_revision"] = getattr(reranker.model.config, "_commit_hash", None)
    try:
        flat = np.asarray(reranker.predict(pairs, batch_size=batch, show_progress_bar=True)).reshape(-1)
    finally:
        # CrossEncoder model-card references can keep a model alive after del.
        # Collect cycles before releasing the allocator cache for checkpointed runs.
        del reranker
        gc.collect()
        torch.cuda.empty_cache()
    if len(flat) != len(pairs) or not np.isfinite(flat).all():
        raise ValueError("Reranker returned invalid scores or the wrong score count")

    out: list[RankedPool] = []
    cursor = 0
    for (idx, dense_scores), head in zip(shortlist, heads):
        scores = flat[cursor:cursor + len(head)]
        cursor += len(head)
        order = np.argsort(-scores, kind="stable")
        head_idx, head_scores = head[order], scores[order]
        # Tail retains dense ordering and has no cross-encoder score.
        tail_idx = idx[len(head):]
        out.append(RankedPool(np.concatenate([head_idx, tail_idx]),
                              np.concatenate([dense_scores[:len(head)][order], dense_scores[len(head):]]),
                              [float(s) for s in head_scores] + [None] * len(tail_idx)))
    return out


def main(
    chunks_file: str, k_docs_sweep: list[int], k_chunks: int, chunks_per_doc: int,
    pool: int, rerank: bool, device: str, out_name: str, batch: int,
    doc_id_as_string: bool = False, candidates: str | None = None,
    rerank_top: int = 0, taxonomy_docs: str | None = None,
    taxonomy_chunks: str | None = None, taxonomy_config: str | None = None,
    taxonomy_weight: float = 0.0,
    selection_configs: list[SelectionConfig] | None = None,
    ranking_cache: Path | None = None, reranker_revision: str = RERANK_REVISION,
) -> None:
    if not k_docs_sweep or any(k < 0 for k in k_docs_sweep) or k_chunks < 0 or chunks_per_doc < 0:
        raise ValueError("Cutoffs must be nonnegative and k_docs_sweep nonempty")
    if pool <= 0 or rerank_top < 0 or batch <= 0:
        raise ValueError("pool/batch must be positive and rerank_top nonnegative")
    configs = selection_configs or [SelectionConfig(max_docs=k, max_chunks=k_chunks or k,
                                                    chunks_per_doc=chunks_per_doc) for k in k_docs_sweep]
    if not rerank and any(config.mode == "threshold" for config in configs):
        raise ValueError("Threshold selection requires reranking; remove --no-rerank")
    if doc_id_as_string and any(config.mode == "threshold" for config in configs):
        raise ValueError("Threshold submissions require integer document IDs")
    names = [out_name if len(configs) == 1 else f"{out_name}_k{config.max_docs}" for config in configs]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate output names in selection sweep")
    for config, name in zip(configs, names, strict=True):
        if config.mode == "threshold" and (SUB / name).exists():
            raise FileExistsError(f"Choose a new output directory: {SUB / name}")
    if ranking_cache and ranking_cache.exists():
        raise FileExistsError(f"Ranking cache already exists: {ranking_cache}")
    chunks = pq.read_table(OUT / chunks_file).to_pydict()
    doc_of = np.asarray(chunks["doc_id"], dtype=np.int64)
    chunk_index_of = np.asarray(chunks["chunk_index"], dtype=np.int32)
    texts, titles = chunks["text"], chunks["title"]
    print(f"chunks: {len(texts):,} over {len(set(doc_of.tolist())):,} docs", flush=True)

    queries = pq.read_table(RAW / "query.parquet").to_pydict()
    qids, qtexts = queries["id"], queries["query"]
    if not qids or len(set(qids)) != len(qids):
        raise ValueError("Queries must be nonempty with unique IDs")

    allowed_per_query: dict[int, set[int]] | None = None
    if candidates:
        # Keep only chunks whose document survived stage 1. At corpus scale this is
        # the difference between embedding ~24M chunks and ~1M.
        cand = pq.read_table(OUT / candidates).to_pydict()
        allowed_per_query = defaultdict(set)
        for q, d in zip(cand["query_id"], cand["doc_id"]):
            allowed_per_query[q].add(d)
        keep_docs = set(cand["doc_id"])
        mask = np.fromiter((d in keep_docs for d in doc_of.tolist()), bool, len(doc_of))
        doc_of = doc_of[mask]
        chunk_index_of = chunk_index_of[mask]
        texts = [t for t, m in zip(texts, mask) if m]
        titles = [t for t, m in zip(titles, mask) if m]
        print(f"after stage-1 filter: {len(texts):,} chunks over "
              f"{len(keep_docs):,} candidate docs", flush=True)

    taxonomy = None
    docs_metadata: dict[int, dict] = {}
    chunk_metadata: dict[int, dict] = {}
    if taxonomy_weight > 0:
        if not taxonomy_docs and not taxonomy_chunks:
            raise ValueError("taxonomy weight requires --taxonomy-docs and/or --taxonomy-chunks")
        taxonomy = Taxonomy.load(taxonomy_config) if taxonomy_config else Taxonomy.load()
        if taxonomy_docs:
            meta = pq.read_table(taxonomy_docs, columns=["doc_id", "concepts", "intents", "specialties"])
            for row in meta.to_pylist():
                docs_metadata[int(row["doc_id"])] = row
        if taxonomy_chunks:
            meta = pq.read_table(taxonomy_chunks, columns=["doc_id", "chunk_index", "concepts", "intents", "specialties"])
            by_chunk = {
                (int(row["doc_id"]), int(row["chunk_index"])): row
                for row in meta.to_pylist()
            }
            for index, (doc_id, chunk_index) in enumerate(zip(doc_of.tolist(), chunk_index_of.tolist())):
                row = by_chunk.get((int(doc_id), int(chunk_index)))
                if row is not None:
                    chunk_metadata[index] = row
        print(f"taxonomy metadata loaded: docs={len(docs_metadata):,}, "
              f"candidate chunks={len(chunk_metadata):,}, weight={taxonomy_weight:g}", flush=True)

    # Bind embeddings to exact inputs and encoder settings, including query order.
    # Older name-only caches remain untouched because their provenance is unknown.
    inputs = {"chunks": fingerprint(OUT / chunks_file), "queries": fingerprint(RAW / "query.parquet")}
    if candidates:
        inputs["candidates"] = fingerprint(OUT / candidates)
    encoder = {"model": EMBED_MODEL, "revision": EMBED_REVISION, "max_length": 512,
               "normalize": True, "title_prefix": True,
               "precision": "float16"}
    cache_key = hashlib.sha256(json.dumps({"inputs": inputs, "encoder": encoder},
                                         sort_keys=True).encode()).hexdigest()[:16]
    tag = f"{Path(chunks_file).stem}" + (f"_{Path(candidates).stem}" if candidates else "")
    cache = OUT / f"emb_{tag}_{cache_key}.npz"
    if cache.exists():
        with np.load(cache) as blob:
            chunk_vecs, query_vecs = blob["chunks"], blob["queries"]
        if len(chunk_vecs) != len(texts) or len(query_vecs) != len(qids):
            raise SystemExit(f"{cache} has {len(chunk_vecs)} rows but the current\n"
                             f"selection has {len(texts)} chunks; delete the cache and rerun")
        print(f"loaded embeddings from {cache.name}", flush=True)
    else:
        if texts:
            chunk_vecs = embed_chunks(texts, titles, batch, device)
            query_vecs = embed_queries(qtexts, device)
        else:
            chunk_vecs = np.empty((0, 0), dtype=np.float32)
            query_vecs = np.empty((len(qids), 0), dtype=np.float32)
        np.savez(cache, chunks=chunk_vecs, queries=query_vecs)
        print(f"cached embeddings to {cache.name}", flush=True)

    shortlist = dense_shortlist(chunk_vecs, query_vecs, qids, doc_of, pool,
                                allowed_per_query, device)
    model_metadata = {"model": RERANK_MODEL, "revision": reranker_revision, "activation": "identity",
                      "precision": "float16", "max_length": 512, "enabled": rerank}
    if rerank:
        pools = cross_encoder_rerank(shortlist, qtexts, texts, titles,
                                    device, batch, rerank_top, reranker_revision, model_metadata)
    else:
        pools = [RankedPool(idx, scores, [None] * len(idx)) for idx, scores in shortlist]

    # Ranking is the expensive part; emitting a different cutoff from it is almost
    # free. One GPU run therefore yields a whole sweep of leaderboard probes, which
    # is what pins down |G| and the F2-optimal K (see r2ai/calibrate.py).
    ranked_per_query: list[tuple[list[int], dict[int, list[int]]]] = []
    evidence_rankings: list[QueryRanking] = []
    metadata_doc_hits = metadata_chunk_hits = 0
    for qi in range(len(qids)):
        doc_best: dict[int, int] = {}
        doc_chunks: dict[int, list[int]] = defaultdict(list)
        pool_result = pools[qi]
        evidence_by_index = {}
        for rank, (c, dense_score, rerank_score) in enumerate(zip(
                pool_result.indices, pool_result.retrieval_scores, pool_result.rerank_scores, strict=True)):
            d = int(doc_of[c])
            doc_best.setdefault(d, rank)
            doc_chunks[d].append(int(c))
            evidence_by_index[int(c)] = EvidenceCandidate(d, int(chunk_index_of[c]), texts[c],
                                                         rerank_score, float(dense_score), titles[c] or "")
        ranked_docs = sorted(doc_best, key=doc_best.get)
        if taxonomy is not None:
            profile = taxonomy.profile(qtexts[qi])
            metadata_doc_hits += sum(
                1 for doc_id in ranked_docs
                if metadata_overlap(profile, docs_metadata.get(doc_id, {})) > 0
            )
            ranked_docs = rrf_rerank(
                ranked_docs, profile, docs_metadata, weight=taxonomy_weight
            )
            for doc_id, indices in doc_chunks.items():
                if chunk_metadata:
                    metadata_chunk_hits += sum(1 for idx in indices if idx in chunk_metadata)
                    doc_chunks[doc_id] = rrf_rerank(
                        indices, profile, chunk_metadata, weight=taxonomy_weight
                    )
        ranked_per_query.append((ranked_docs, doc_chunks))
        evidence_rankings.append(QueryRanking(int(qids[qi]), ranked_docs,
                                             [evidence_by_index[c] for d in ranked_docs for c in doc_chunks[d]],
                                             qtexts[qi]))

    if taxonomy is not None:
        print(f"taxonomy candidate coverage: doc matches={metadata_doc_hits:,}, "
              f"chunks with metadata={metadata_chunk_hits:,}", flush=True)

    metadata = {"inputs": inputs, "encoder": encoder, "reranker": model_metadata,
                "pool": pool, "rerank_top": rerank_top, "taxonomy_weight": taxonomy_weight,
                "priority": "document rank, then within-document chunk rank",
                "score_semantics": "raw cross-encoder logits; null means not reranked"}
    for name, path in (("taxonomy_docs", taxonomy_docs), ("taxonomy_chunks", taxonomy_chunks),
                       ("taxonomy_config", taxonomy_config)):
        if path and taxonomy is not None:
            metadata[name] = fingerprint(Path(path))
    if taxonomy is not None:
        metadata["taxonomy_version"] = taxonomy.version
        metadata["taxonomy_config"] = fingerprint(Path(taxonomy_config) if taxonomy_config else DEFAULT_CONFIG)
    if ranking_cache:
        write_ranking_cache(ranking_cache, evidence_rankings, metadata)
        print(f"cached evidence rankings: {ranking_cache}", flush=True)
    SUB.mkdir(parents=True, exist_ok=True)
    for config in configs:
        name = out_name if len(configs) == 1 else f"{out_name}_k{config.max_docs}"
        if config.mode == "topk":
            emit(qids, ranked_per_query, texts, config.max_docs, config.max_chunks, config.chunks_per_doc,
                 doc_id_as_string, name)
        else:
            summary = replay(evidence_rankings, metadata, config, SUB / name)
            print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"max docs available per query: "
          f"{min(len(d) for d, _ in ranked_per_query)}-"
          f"{max(len(d) for d, _ in ranked_per_query)}", flush=True)


COMPRESSION = {
    "deflate": zipfile.ZIP_DEFLATED,
    "bzip2": zipfile.ZIP_BZIP2,
    "lzma": zipfile.ZIP_LZMA,
}


def write_submission(records: list[dict], out_name: str, keep_json: bool = False,
                     compression: str = "deflate") -> Path:
    """Write the upload zip, and by default not the loose JSON beside it.

    Only the zip is ever uploaded, so keeping the uncompressed copy duplicates
    every submission on disk — that reached 4.3 GB across one week of runs.

    The platform caps uploads at ~50 MB *after* compression, and chunk count is
    what drives chunk recall, so the codec decides how much evidence fits.
    Measured on a 205 MB payload: deflate 44.7 MB, bzip2 31.8 MB, lzma 27.3 MB.
    Both alternatives are standard ZIP methods, but a scorer that only handles
    deflate will reject them, so deflate stays the default.
    """
    SUB.mkdir(parents=True, exist_ok=True)
    json_path = SUB / f"{out_name}.json"
    json_path.write_text(dumps_one_record_per_line(records), encoding="utf-8")
    zip_path = SUB / f"{out_name}.zip"
    with zipfile.ZipFile(zip_path, "w", COMPRESSION[compression]) as zf:
        zf.write(json_path, arcname="predictions.json")
    if not keep_json:
        json_path.unlink()
    return zip_path


def dumps_one_record_per_line(records: list[dict]) -> str:
    """Serialise the array with one query per line.

    Still a single JSON array, so the scorer reads it unchanged, but a human (or
    `wc -l`, or `grep '"id"'`) can see all 1200 queries. Dumping it minified puts
    ~15 MB on one line, which makes a complete file look like a single record in
    an editor.
    """
    body = ",\n ".join(json.dumps(r, ensure_ascii=False) for r in records)
    return f"[{body}]\n"


def emit(
    qids: list[int],
    ranked_per_query: list[tuple[list[int], dict[int, list[int]]]],
    texts: list[str], k_docs: int, k_chunks: int, chunks_per_doc: int,
    doc_id_as_string: bool, out_name: str,
) -> None:
    # The spec says doc ids are integers but its worked example shows strings;
    # the flag lets us settle it with one cheap public-phase submission.
    cast = str if doc_id_as_string else int

    records = []
    for qid, (docs, doc_chunks) in zip(qids, ranked_per_query):
        ranked_docs = docs[:k_docs]
        selected: list[dict] = []
        for d in ranked_docs:
            for c in doc_chunks[d][:chunks_per_doc]:
                selected.append({"doc_id": cast(d), "chunk_text": texts[c]})
        records.append({
            "id": int(qid),
            "relevant_docs": [cast(d) for d in ranked_docs],
            "relevant_chunks": selected[:k_chunks],
        })

    zip_path = write_submission(records, out_name)
    empty = sum(1 for r in records if not r["relevant_docs"])
    avg_chunks = sum(len(r["relevant_chunks"]) for r in records) / max(len(records), 1)
    print(f"wrote {zip_path}  k_docs={k_docs}  avg_chunks={avg_chunks:.1f}  "
          f"queries={len(records)}  empty={empty}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks-file", default="chunks.parquet")
    ap.add_argument("--k-docs", default="10",
                    help="cutoff, or a comma-separated sweep such as 5,10,20,30")
    ap.add_argument("--k-chunks", type=int, default=0,
                    help="chunks per query; 0 means match --k-docs")
    ap.add_argument("--chunks-per-doc", type=int, default=1)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--doc-id-as-string", action="store_true",
                    help="emit doc ids as JSON strings instead of integers (A/B test)")
    ap.add_argument("--rerank-top", type=int, default=0,
                    help="rerank only the top N of the pool (0 = all)")
    ap.add_argument("--candidates", default=None,
                    help="stage-1 candidates parquet (e.g. docidx_candidates.parquet)")
    ap.add_argument("--taxonomy-docs", default=None,
                    help="optional docs_taxonomy_v1.parquet metadata for late fusion")
    ap.add_argument("--taxonomy-chunks", default=None,
                    help="optional chunks_taxonomy_v1.parquet metadata for chunk ordering")
    ap.add_argument("--taxonomy-config", default=None,
                    help="taxonomy JSON config; defaults to configs/vibiomir/taxonomy_v1.json")
    ap.add_argument("--taxonomy-weight", type=float, default=0.0,
                    help="RRF contribution from taxonomy metadata; 0 disables it")
    ap.add_argument("--out", default="submission")
    ap.add_argument("--ranking-cache", type=Path, help="Write self-contained JSONL or .jsonl.gz for CPU replay")
    ap.add_argument("--reranker-revision", default=RERANK_REVISION, help="Pin a Hugging Face revision for score calibration")
    add_selection_arguments(ap)
    a = ap.parse_args()
    sweep = [int(x) for x in str(a.k_docs).split(",") if x.strip()]
    configs = ([config_from_args(a, max_docs=k, max_chunks=a.k_chunks or k,
                                chunks_per_doc=a.chunks_per_doc) for k in sweep]
               if not a.selection_config else [config_from_args(a, max_docs=0, max_chunks=0, chunks_per_doc=0)])
    if any(c.mode == "threshold" for c in configs) and a.doc_id_as_string:
        ap.error("Threshold submissions use integer document IDs; remove --doc-id-as-string")
    main(a.chunks_file, sweep, a.k_chunks, a.chunks_per_doc, a.pool,
         not a.no_rerank, a.device, a.out, a.batch, a.doc_id_as_string,
         a.candidates, a.rerank_top, a.taxonomy_docs, a.taxonomy_chunks,
         a.taxonomy_config, a.taxonomy_weight, configs, a.ranking_cache, a.reranker_revision)

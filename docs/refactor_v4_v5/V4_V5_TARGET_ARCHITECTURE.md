# V4/V5 target architecture and implemented slice

V4 has one package, `src/medrag`, and the main CLI `python3 -m src.cli` (`medrag` after installation). The default `src.api` import and Docker service also forward to V4. `Source` carries corpus version, official ID, URL, title, extracted text, origin and raw hash. `Passage` carries exact offsets into that extracted text. `Pipeline` rejects mixed corpus versions and verifies every passage against its parent before retrieval.

Flow: pinned corpus registry → verified ID/URL mapping → immutable raw HTML → extraction → token-span passages → full-body document BM25 and passage BM25 → optional BGE-M3 dense adapter → RRF with rank provenance → within-document evidence → optional BGE reranker → ViBioMIR record adapter.

The current BM25 implementation is an in-memory backend capped at 1,000 prepared sources. It indexes the entire text of every document supplied to it, but is **not** suitable for 1.22 million official URLs or 20 million passages. Preparation streams pinned official and historical URL Parquet, verifies ID/URL pairs, and writes a checksummed snapshot without changing source data. The BGE-M3 and reranker adapters are implemented, but their packages and model weights were unavailable in this shell, so no real model score is claimed. A corpus-scale backend, checkpointed full preparation, persisted index manifest, and full submission ZIP workflow remain release work.

`configs/v4_core.yaml` pins official revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`. `configs/v5_extensions.yaml` declares optional extension points with every feature disabled. Taxonomy, expansion, multihop, and late interaction never form a hard dependency for V4.

The historical CLI/API/Docker source is archived at `archive/legacy_runtime/`; `r2ai/` and the older runtime modules remain for replay. The default V4 entrypoints import none of them.

## Phase 2 adapter update — 2026-10-16

The V4 prepared loader now also accepts a checksummed Phase 2 snapshot directory. It maps the release's normalized source text and original passage IDs/offsets into V4 `Source`/`Passage` contracts and verifies each passage against its parent. `prepare-phase2`, CLI `search`, and the HTTP API all load the resulting snapshot through the same `Pipeline`; no Phase 2 rechunking is performed.

This path remains bounded at 1,000 sources and currently uses V4's in-memory BM25 implementation. BGE-M3 and the reranker remain disabled in `configs/v4_core.yaml`. Task 2's real-model retrieval/reranking results therefore are not claimed as V4 parity. Full-release indexing, dense/reranker integration in the V4 backend, and a full submission remain follow-up work.

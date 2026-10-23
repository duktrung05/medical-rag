# Dependency audit

Static `import` / `from` edges below were extracted with Python AST before the mainline cutover. They document the historical call graph and are not a current import inventory. Dynamic CLI imports and config references were reviewed separately; static absence alone never marks a file unused.

| Module | Internal imports |
|---|---|
| `src/adapters/__init__.py` | `src.adapters.input` |
| `src/adapters/input.py` | `src.data.preprocess`, `src.data.schema` |
| `src/api.py` | `src.adapters`, `src.config`, `src.data.schema`, `src.service` |
| `src/cli.py` | `scripts.runtime.evaluate`, `scripts.runtime.retrieve`, `src.config`, `src.submission.validate` |
| `src/config.py` | `src.encoder_profile` |
| `src/data/__init__.py` | `src.data.loader`, `src.data.preprocess`, `src.data.schema` |
| `src/data/loader.py` | `src.data.schema` |
| `src/data/validator.py` | `src.data.schema` |
| `src/evaluation/__init__.py` | `src.evaluation.evaluator`, `src.evaluation.fbeta`, `src.evaluation.precision`, `src.evaluation.recall` |
| `src/evaluation/evaluator.py` | `src.data.loader`, `src.data.schema`, `src.evaluation.fbeta` |
| `src/evaluation/fbeta.py` | `src.evaluation.precision`, `src.evaluation.recall` |
| `src/indexing/__init__.py` | `src.indexing.sparse_index` |
| `src/indexing/sparse_index.py` | `src.data.schema` |
| `src/medrag/chunking/passage_builder.py` | `src.medrag.core.contracts` |
| `src/medrag/cli.py` | `src.medrag.core.config`, `src.medrag.data.corpus_registry`, `src.medrag.pilot` |
| `src/medrag/core/pipeline.py` | `src.medrag.core.contracts`, `src.medrag.evidence.verifier`, `src.medrag.retrieval.core` |
| `src/medrag/data/loader.py` | `src.medrag.core.contracts`, `src.medrag.data.corpus_registry`, `src.medrag.data.extraction` |
| `src/medrag/evidence/verifier.py` | `src.medrag.core.contracts` |
| `src/medrag/pilot.py` | `src.medrag.chunking.passage_builder`, `src.medrag.core.pipeline`, `src.medrag.data.loader` |
| `src/medrag/retrieval/core.py` | `src.medrag.core.contracts` |
| `src/pipeline.py` | `src.data.loader`, `src.data.schema`, `src.reranking.base`, `src.retrieval.bm25`, `src.selection.prediction` |
| `src/reranking/__init__.py` | `src.reranking.base`, `src.reranking.bge_reranker`, `src.reranking.factory` |
| `src/reranking/bge_reranker.py` | `src.reranking.base` |
| `src/reranking/factory.py` | `src.config`, `src.reranking.base`, `src.reranking.bge_reranker` |
| `src/retrieval/__init__.py` | `src.retrieval.bm25`, `src.retrieval.dense`, `src.retrieval.factory`, `src.retrieval.fusion`, `src.retrieval.hybrid` |
| `src/retrieval/bm25.py` | `src.config`, `src.data.schema`, `src.indexing.sparse_index` |
| `src/retrieval/demo.py` | `src.data.preprocess`, `src.data.schema`, `src.retrieval.bm25` |
| `src/retrieval/dense.py` | `src.config`, `src.data.schema`, `src.encoder_profile`, `src.indexing.sparse_index`, `src.retrieval.bm25`, `src.retrieval.dense_encoding`, `src.retrieval.exact_dense` |
| `src/retrieval/factory.py` | `src.config`, `src.retrieval.bm25`, `src.retrieval.dense`, `src.retrieval.hybrid` |
| `src/retrieval/hybrid.py` | `src.retrieval.bm25`, `src.retrieval.fusion` |
| `src/scoring/__init__.py` | `src.scoring.document_score` |
| `src/selection/__init__.py` | `src.selection.calibrate`, `src.selection.threshold` |
| `src/selection/calibrate.py` | `src.data.schema`, `src.evaluation.evaluator`, `src.selection.threshold` |
| `src/selection/prediction.py` | `src.data.loader`, `src.data.schema`, `src.scoring.document_score`, `src.selection.threshold` |
| `src/service.py` | `src.config`, `src.data.loader`, `src.data.validator`, `src.indexing.sparse_index`, `src.pipeline`, `src.reranking.factory`, `src.retrieval.demo`, `src.retrieval.factory` |
| `src/submission/__init__.py` | `src.submission.build`, `src.submission.validate` |
| `src/submission/build.py` | `src.data.loader`, `src.data.schema` |
| `src/submission/validate.py` | `src.data.loader`, `src.data.validator` |
| `r2ai/bm25index.py` | `r2ai.taxonomy` |
| `r2ai/calibrate_evidence.py` | `r2ai.evidence_io`, `r2ai.evidence_selector` |
| `r2ai/candidates.py` | `r2ai.urltext` |
| `r2ai/crawl.py` | `r2ai.deadhosts` |
| `r2ai/deadhosts.py` | `r2ai.crawl` |
| `r2ai/evidence_io.py` | `r2ai.evidence_selector` |
| `r2ai/expand.py` | `r2ai.rank` |
| `r2ai/index.py` | `r2ai.langbalance` |
| `r2ai/probe_domains.py` | `r2ai.crawl` |
| `r2ai/purge.py` | `r2ai.crawl` |
| `r2ai/rank.py` | `r2ai.evidence_io`, `r2ai.evidence_selector`, `r2ai.select_evidence`, `r2ai.taxonomy` |
| `r2ai/reslice.py` | `r2ai.rank` |
| `r2ai/select_evidence.py` | `r2ai.evidence_io`, `r2ai.evidence_selector` |
| `r2ai/taxonomy_rescore_submission.py` | `r2ai.taxonomy` |
| `r2ai/test_evidence_submission.py` | `r2ai.evidence_io`, `r2ai.evidence_selector`, `r2ai.rank`, `r2ai.select_evidence`, `r2ai.validate` |
| `scripts/experiments/vimedqa/ablate_fusion.py` | `scripts.tooling.run_metadata`, `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.retrieval.fusion` |
| `scripts/experiments/vimedqa/ablate_reranker.py` | `src.adapters`, `src.config`, `src.data.loader`, `src.evaluation.evaluator`, `src.reranking.factory`, `src.service` |
| `scripts/experiments/vimedqa/analyze_failures.py` | `scripts.tooling.run_metadata`, `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.data.loader` |
| `scripts/experiments/vimedqa/calibrate_selection.py` | `scripts.tooling.run_metadata`, `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.data.loader`, `src.evaluation.evaluator`, `src.selection.prediction` |
| `scripts/experiments/vimedqa/freeze_baseline.py` | `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.data.loader`, `src.evaluation.ranking_metrics`, `src.indexing.sparse_index` |
| `scripts/experiments/vimedqa/replay_baseline.py` | `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.evaluation.ranking_metrics` |
| `scripts/experiments/vimedqa/replay_selection.py` | `scripts.tooling.run_metadata`, `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.config`, `src.data.loader`, `src.selection.prediction` |
| `scripts/runtime/build_dense_index.py` | `src.config`, `src.data.loader`, `src.retrieval.dense` |
| `scripts/runtime/build_sparse_index.py` | `src.config`, `src.data.loader`, `src.retrieval.bm25` |
| `scripts/runtime/evaluate.py` | `src.data.loader`, `src.evaluation.evaluator` |
| `scripts/runtime/retrieve.py` | `src.adapters`, `src.config`, `src.data.loader`, `src.data.validator`, `src.service` |
| `scripts/runtime/run_dense_baseline.py` | `scripts.runtime.run_dense_smoke`, `src.adapters`, `src.config`, `src.data.loader`, `src.data.validator`, `src.evaluation.evaluator`, `src.indexing.sparse_index`, `src.retrieval.dense`, `src.retrieval.exact_dense`, `src.submission.validate` |
| `scripts/runtime/run_dense_smoke.py` | `src.adapters`, `src.data.loader`, `src.data.schema`, `src.data.validator`, `src.encoder_profile`, `src.evaluation.evaluator`, `src.retrieval.dense_encoding`, `src.retrieval.exact_dense`, `src.submission.validate` |
| `scripts/runtime/run_sparse_smoke.py` | `src.adapters`, `src.config`, `src.data.loader`, `src.data.schema`, `src.evaluation.evaluator`, `src.evaluation.recall`, `src.indexing.sparse_index`, `src.retrieval.bm25`, `src.submission.validate` |
| `scripts/runtime/verify_dense_index.py` | `src.config`, `src.data.loader`, `src.indexing.sparse_index`, `src.retrieval.factory` |
| `scripts/tooling/make_submission.py` | `src.data.loader`, `src.submission.build`, `src.submission.validate` |
| `scripts/tooling/prepare_data.py` | `src.data.loader`, `src.data.preprocess`, `src.data.validator` |
| `scripts/tooling/validate_submission.py` | `src.submission.validate` |
| `scripts/v4/pilot.py` | `src.medrag.pilot` |
| `scripts/workflows/medquad/run_medquad_hybrid_smoke.py` | `src.adapters`, `src.config`, `src.data.loader`, `src.data.validator`, `src.evaluation.evaluator`, `src.indexing.sparse_index`, `src.pipeline`, `src.retrieval.bm25`, `src.retrieval.dense`, `src.retrieval.hybrid` |
| `scripts/workflows/vibiomir/audit_task0.py` | `r2ai.evidence_io` |
| `scripts/workflows/vibiomir/chunk_vibiomir_probe.py` | `src.chunking` |
| `scripts/workflows/vibiomir/crawl_vibiomir_batch.py` | `scripts.workflows.vibiomir.prepare_vibiomir`, `src.collection.fetcher` |
| `scripts/workflows/vibiomir/extract_vibiomir_probe.py` | `src.extraction.html_extractor` |
| `scripts/workflows/vibiomir/probe_vibiomir_urls.py` | `scripts.workflows.vibiomir.prepare_vibiomir`, `src.collection.fetcher` |
| `scripts/workflows/vibiomir/sample_vibiomir_urls.py` | `scripts.workflows.vibiomir.prepare_vibiomir` |
| `scripts/workflows/vimedqa/benchmark_vimed_validation.py` | `src.adapters`, `src.config`, `src.data.loader`, `src.data.schema`, `src.evaluation.ranking_metrics`, `src.indexing.sparse_index`, `src.retrieval.fusion`, `src.scoring.document_score`, `src.service` |
| `scripts/workflows/vimedqa/evaluate_vimed.py` | `src.adapters`, `src.config`, `src.data.schema`, `src.service` |
| `scripts/workflows/vimedqa/prepare_vimed.py` | `src.config`, `src.data.schema`, `src.retrieval.bm25` |
| `scripts/workflows/vimedqa/prepare_vimed_validation.py` | `scripts.workflows.vimedqa.prepare_vimed`, `src.data.loader`, `src.indexing.sparse_index` |
| `scripts/workflows/vimedqa/report_vimed_validation.py` | `scripts.tooling.run_metadata`, `scripts.workflows.vimedqa.benchmark_vimed_validation`, `src.data.loader`, `src.data.schema`, `src.evaluation.evaluator`, `src.evaluation.ranking_metrics`, `src.indexing.sparse_index` |
| `tests/integration/runtime/test_api.py` | `src.api` |
| `tests/integration/runtime/test_inspector.py` | `src.data.loader`, `src.indexing.sparse_index`, `src.inspector` |
| `tests/integration/runtime/test_retrieve_backends.py` | `scripts.runtime.retrieve`, `src.config`, `src.data.loader`, `src.reranking.bge_reranker`, `src.retrieval.bm25`, `src.retrieval.demo`, `src.retrieval.dense`, `src.retrieval.hybrid`, `src.service` |
| `tests/integration/runtime/test_runtime.py` | `src.adapters`, `src.config`, `src.data.schema`, `src.service` |
| `tests/integration/tooling/test_build_dense_timeout.py` | `scripts.runtime` |
| `tests/integration/tooling/test_config.py` | `src.config`, `src.retrieval.factory` |
| `tests/integration/tooling/test_hf_vibiomir_snapshot.py` | `scripts.tooling.hf_vibiomir_snapshot` |
| `tests/integration/tooling/test_submission.py` | `src.data.loader`, `src.data.schema`, `src.submission.build`, `src.submission.validate` |
| `tests/repository/test_workflow_integrity.py` | `scripts.runtime.run_dense_smoke`, `src.config` |
| `tests/unit/data/test_doc_chunk_relation.py` | `src.data.loader`, `src.data.schema`, `src.submission.build` |
| `tests/unit/data/test_schema.py` | `src.data.schema` |
| `tests/unit/ranking/test_metrics.py` | `src.data.loader`, `src.data.schema`, `src.evaluation.evaluator`, `src.evaluation.fbeta`, `src.evaluation.precision`, `src.evaluation.recall` |
| `tests/unit/ranking/test_scoring_and_selection.py` | `src.retrieval.fusion`, `src.scoring.document_score`, `src.selection.threshold` |
| `tests/unit/ranking/test_selection_prediction.py` | `src.data.loader`, `src.selection.prediction`, `src.selection.threshold` |
| `tests/unit/retrieval/test_bge_reranker.py` | `src.data.loader`, `src.pipeline`, `src.reranking.bge_reranker`, `src.retrieval.fusion` |
| `tests/unit/retrieval/test_bm25.py` | `src.config`, `src.data.schema`, `src.retrieval.bm25` |
| `tests/unit/retrieval/test_dense_retriever.py` | `src.config`, `src.data.loader`, `src.data.schema`, `src.retrieval.dense`, `src.retrieval.dense_encoding`, `src.service` |
| `tests/unit/retrieval/test_encoder_profiles.py` | `src.config`, `src.retrieval.dense`, `src.retrieval.dense_encoding` |
| `tests/unit/retrieval/test_exact_dense.py` | `scripts.runtime.run_dense_smoke`, `src.data.loader`, `src.retrieval.exact_dense` |
| `tests/unit/retrieval/test_factories.py` | `src.config`, `src.data.schema`, `src.reranking.bge_reranker`, `src.reranking.factory`, `src.retrieval.bm25`, `src.retrieval.dense`, `src.retrieval.factory`, `src.retrieval.hybrid` |
| `tests/unit/retrieval/test_hybrid_retriever.py` | `src.retrieval.hybrid` |
| `tests/v4/test_core.py` | `src.medrag.adapters.vibiomir_submission`, `src.medrag.chunking.passage_builder`, `src.medrag.core.config`, `src.medrag.core.contracts`, `src.medrag.core.pipeline`, `src.medrag.data.corpus_registry`, `src.medrag.data.extraction`, `src.medrag.evidence.verifier`, `src.medrag.retrieval.core` |
| `tests/workflows/medquad/test_convert_medquad.py` | `scripts.workflows.medquad.convert_medquad` |
| `tests/workflows/vibiomir/test_audit_task0.py` | `r2ai.evidence_selector`, `scripts.workflows.vibiomir.audit_task0` |
| `tests/workflows/vibiomir/test_chunk_vibiomir_probe.py` | `scripts.workflows.vibiomir.chunk_vibiomir_probe`, `src.chunking` |
| `tests/workflows/vibiomir/test_crawl_vibiomir_batch.py` | `scripts.workflows.vibiomir.crawl_vibiomir_batch` |
| `tests/workflows/vibiomir/test_evidence_selector.py` | `r2ai.calibrate_evidence`, `r2ai.evidence_io`, `r2ai.evidence_selector`, `r2ai.rank`, `r2ai.select_evidence`, `r2ai.test_evidence_submission`, `r2ai.validate` |
| `tests/workflows/vibiomir/test_extract_vibiomir_probe.py` | `scripts.workflows.vibiomir.extract_vibiomir_probe` |
| `tests/workflows/vibiomir/test_prepare_vibiomir.py` | `scripts.workflows.vibiomir.prepare_vibiomir` |
| `tests/workflows/vibiomir/test_probe_vibiomir_urls.py` | `scripts.workflows.vibiomir.prepare_vibiomir`, `scripts.workflows.vibiomir.probe_vibiomir_urls`, `scripts.workflows.vibiomir.sample_vibiomir_urls` |
| `tests/workflows/vibiomir/test_sample_vibiomir_urls.py` | `scripts.workflows.vibiomir.prepare_vibiomir`, `scripts.workflows.vibiomir.sample_vibiomir_urls` |
| `tests/workflows/vibiomir/test_taxonomy.py` | `r2ai.taxonomy` |
| `tests/workflows/vibiomir/test_vibiomir_chunker.py` | `src.chunking` |
| `tests/workflows/vibiomir/test_vibiomir_fetcher.py` | `src.collection.fetcher` |
| `tests/workflows/vibiomir/test_vibiomir_html_extractor.py` | `src.extraction.html_extractor` |
| `tests/workflows/vimedqa/test_dense_baseline_preprocessing.py` | `scripts.runtime` |
| `tests/workflows/vimedqa/test_dense_language_metrics.py` | `scripts.runtime.run_dense_smoke`, `src.data.schema`, `src.retrieval.exact_dense` |
| `tests/workflows/vimedqa/test_offline_analysis.py` | `scripts.experiments.vimedqa.ablate_fusion`, `scripts.experiments.vimedqa.analyze_failures` |
| `tests/workflows/vimedqa/test_validation_benchmark.py` | `scripts.workflows.vimedqa.benchmark_vimed_validation`, `scripts.workflows.vimedqa.prepare_vimed`, `scripts.workflows.vimedqa.prepare_vimed_validation`, `src.evaluation.ranking_metrics`, `src.reranking.bge_reranker` |
| `tests/workflows/vimedqa/test_vimed_evaluation.py` | `scripts.workflows.vimedqa` |

## Entrypoints and call paths

- Historical generic CLI: its original source, now `archive/legacy_runtime/src_cli_v1.py`, used dynamic imports into `src/submission/validate.py`, `scripts/runtime/evaluate.py`, `src/config.py`, and `scripts/runtime/retrieve.py`.
- Historical API: its original source, now `archive/legacy_runtime/src_api_v1.py`, called `src/service.py` and the old pipeline.
- Current default CLI: `src/cli.py` → `src/medrag/cli.py`; current default API: `src/api.py` → `src/medrag/api.py`. The Docker service runs the V4 API. Official snapshots pass through `src/medrag/data/prepared.py` and verify both official and historical ID/URL pairs before using existing raw HTML.
- ViBioMIR competition scripts: `r2ai/crawl.py` → raw HTML; `r2ai/extract.py` → historical chunks; `r2ai/bm25index.py` and `r2ai/index.py` → indexes; `r2ai/fuse.py`, `r2ai/rank.py`, `r2ai/evidence_selector.py` → ranking/selection; `r2ai/validate.py` → ZIP schema.
- V4 entrypoint: `src/medrag/cli.py` → V4 config/registry → `src/medrag/pilot.py` → loader/extraction → passage builder → V4 pipeline → document/passage BM25, optional dense, fusion, optional reranker, exact evidence. V4 imports no `r2ai` or old `src` retrieval modules.
- `src/retrieval_v2/` does not exist in this checkout. Historical V2 plans in outputs/docs remain archived.

## Duplication and reuse decisions

- `src/extraction/html_extractor.py` and `r2ai/extract.py` both decode/extract; the former has better charset handling, the latter has competition crawling assumptions. V4 uses an independent structure-preserving extractor with tests; charset and fallback behavior still need broader regression.
- `src/chunking/paragraph_chunker.py` and `r2ai/extract.py` both chunk; V4 uses token spans and exact extracted-text offsets. Neither old implementation is imported.
- `src/retrieval/fusion.py` and `r2ai/fuse.py` overlap; V4 has one RRF implementation preserving channel ranks.
- `src/submission/validate.py` expects JSONL chunk IDs, while `r2ai/validate.py` expects ViBioMIR JSON in ZIP with `{doc_id,chunk_text}`. They are different schemas. V4 adapter targets the latter; final ZIP validation is not yet implemented.
- `r2ai/bm25index.py` uses `max_chunks` and imports `r2ai.taxonomy`; V4 full-body retrieval deliberately avoids both constraints.

## Caveats

- This AST map does not resolve function-level calls, reflection, shell invocations, or config-selected classes. It is evidence for migration, not proof that old files may be deleted.
- Existing tests and ViBioMIR outputs were preserved. Old entrypoint source was copied to `archive/legacy_runtime/` before replacement; other legacy modules remain available for historical replay.

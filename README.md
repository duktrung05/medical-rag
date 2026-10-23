# Medical-RAG

Medical-RAG V4 is the project's **main architecture** on `master`. V5 features extend the same pipeline through disabled-by-default interfaces. The default CLI, HTTP API and Docker service all enter `src/medrag/`; they never fall back to V1/V2.

The corpus registry pins the current locally verified ViBioMIR revision in [v4_corpus_registry.json](configs/v4_corpus_registry.json). Existing raw HTML, Parquet, indexes, scoring feedback and unique submissions remain preserved. The V4 runtime is currently validated on bounded CPU snapshots. A corpus-scale sparse backend, BGE-M3/BGE-reranker runs and official submission ZIP validation are still required before a full competition run. See the [status report](docs/refactor_v4_v5/FINAL_REFACTOR_REPORT.md).

## Run V4 locally

Use Python 3.11 or newer with the project dependencies:

```bash
python3.11 -m pip install -e '.[api]'
python3.11 -m src.cli check-config
python3.11 -m src.cli prepare --output outputs/v4/prepared.jsonl --limit 100
python3.11 -m src.cli search --prepared outputs/v4/prepared.jsonl --query 'triệu chứng tăng huyết áp'
```

`prepare` reads the pinned official URL Parquet, reuses available raw HTML only when the official ID and URL match, and writes a new V4 snapshot and manifest. It never changes existing source data. The current in-memory backend accepts at most 1,000 sources; `--limit` defaults to 100. If `pyarrow` or matching raw HTML is unavailable, preparation fails explicitly.

For a locked Phase 2 subset, `prepare-phase2` loads the release's existing passages and exact normalized-text offsets into the same V4 `Pipeline`; it does not rechunk the source. Supply explicit ID lists and a new output path:

```bash
python3.11 -m src.cli prepare-phase2 \
  --doc-ids outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/doc_ids.json \
  --query-ids outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/query_ids.json \
  --output outputs/v4/phase2_subset/20261016/selection_seed20261016
python3.11 -m src.cli search \
  --prepared outputs/v4/phase2_subset/20261016/selection_seed20261016 \
  --query 'triệu chứng tăng huyết áp'
```

The Phase 2 V4 snapshot also has a 1,000-source in-memory limit. This wires the release data into V4's current lexical workflow; it does not yet enable the BGE models or provide a full-corpus index.

For a bounded real-model V4 run, pass `--config configs/v4_phase2_subset.yaml`; it enables the pinned BGE-M3 encoder and BGE reranker under the same `src/medrag` pipeline. The API accepts the same config path via `MEDRAG_CONFIG`. Dense and reranker remain disabled in `v4_core.yaml` until a scalable backend and full-corpus runtime are ready.

The read-only three-document smoke can run without Parquet:

```bash
python3 -m src.cli pilot
python3 -m unittest discover -s tests/v4 -v
```

That smoke uses the **historical local corpus version**. It does not establish full coverage or a leaderboard score for the pinned official corpus.

## HTTP and Docker

After preparing either `outputs/v4/prepared.jsonl` or a Phase 2 V4 snapshot directory:

```bash
MEDRAG_PREPARED=outputs/v4/prepared.jsonl python3.11 -m uvicorn src.medrag.api:create_app --factory --host 127.0.0.1 --port 8000
```

`GET /health` returns 503 until a verified V4 snapshot is loaded. `POST /search` accepts `{"id": 1, "query": "..."}` and returns ViBioMIR-style document IDs and exact-source evidence. `docker compose up --build` serves the same API, mounting `data/` and `outputs/` read-only. Its health check also fails closed until the prepared snapshot exists.

For the real-model Phase 2 subset path, set `MEDRAG_CONFIG=configs/v4_phase2_subset.yaml` and point `MEDRAG_PREPARED` to the snapshot directory. The default core config keeps model stages disabled.

## Structure

The [working-tree checkpoint and artifact inventory](docs/repository_hygiene/README.md) describe the preserved 2026-10-23 research state, validation limits and recovery files.

- `src/medrag/`: V4 data contracts, versioned loader, extraction, exact-offset chunking, retrieval, fusion, evidence verification, API and optional model adapters.
- `configs/v4_core.yaml`: active V4 configuration; `configs/v5_extensions.yaml`: optional features, all disabled.
- `scripts/v4/`, `tests/v4/`: V4 smoke and contract checks.
- `docs/refactor_v4_v5/`: audit, protected-data manifest, migration and remaining tasks.
- `archive/docs_pre_v4/`, `archive/legacy_runtime/`: historical references. Older `r2ai/`, `scripts/runtime/` and `src/service.py` remain for reproducibility and are not imported by the default V4 entrypoints.

No full-corpus index build, large embedding job, Git push or leaderboard submission is performed by these commands.

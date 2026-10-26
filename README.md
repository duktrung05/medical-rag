# Medical-RAG 

The active implementation is `src/medrag/`. The current competition champion is **S2**, with user-reported BTC FINAL **0.2318**. The CLI/API demo and competition submission pipeline use different configurations.

Start with the [V4 workspace guide](docs/V4_WORKSPACE_GUIDE.md) and the [verified project review](docs/v4_project_review_20261025_vi.md).

## Competition baseline

| Component | Location |
|---|---|
| Champion S2 | `outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/submission.zip` |
| Score receipt and manifest | Same directory as the champion ZIP |
| Frozen precursor config | `configs/v4_submission_p2_a0.yaml` |
| Full eligible index | `outputs/v4/task4/full_index_gpu3_20261019/` |
| Refined checkpoints | `outputs/v4/a0_refined_candidates_20261024/` |
| S2 refinement/export | `scripts/v4/refine_a0_submission.py` |

The effective S2 policy also includes supplemental reranking, context expansion and adaptive document selection in Python. The precursor YAML alone does not reproduce S2. R3/R4/R5/R8 experiments are not the current champion.

S2 SHA-256: `081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.

## Demo CLI and API

The default `configs/v4_core.yaml` disables dense retrieval and reranking and uses bounded prepared snapshots:

```bash
python3 -m src.cli check-config
python3 -m src.cli pilot
python3 -m src.cli search --prepared outputs/v4/prepared.jsonl --query 'triệu chứng tăng huyết áp'
```

Search requires an existing verified prepared snapshot. The API is `src.medrag.api:create_app`; `MEDRAG_CONFIG` and `MEDRAG_PREPARED` select its configuration and snapshot. Docker serves the same API. See the [script guide](scripts/README.md) for the competition workflow.

## Workspace layout

- `src/medrag/`: V4 runtime, retrieval, ranking, evidence and validation.
- `scripts/v4/`: competition runners and shared diagnostic helpers.
- `configs/`: pinned corpus/model configs and preserved frozen configurations.
- `tests/v4/`: V4 contracts and regression tests.
- `docs/`: current workflow, champion history and verified findings.
- `outputs/v4/`: active artifacts and retained provenance required by V4/S2.
- `data/`, `.cache/`, `.venv-*/`: preserved datasets, model cache and environments.
- `archive/` and older source modules: retained compatibility/history.

Old outputs, failed runs and superseded reports have been moved outside the checkout. The [cleanup inventory](docs/workspace_cleanup_20261025/README.md) records their archive location and recovery command. Source/config/test files remain because active modules share historical helpers.

The [2026-10-23 checkpoint](docs/repository_hygiene/README.md) records earlier working-tree provenance and predates the current cleanup.

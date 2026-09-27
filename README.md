# Medical Retrieval — ViMedQA

Current workflow: prepare context-only corpus -> BM25 / BGE-M3 -> RRF -> BGE reranker -> selection -> source-context evaluation -> comparison UI.

## Run the UI

From the repository root, using the existing CUDA environment and prepared data:

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'
$env:HF_HUB_OFFLINE = '1'
.venv/Scripts/python.exe -m uvicorn src.inspector:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

Open **http://localhost:8001**. The UI compares BM25, dense, hybrid and hybrid + reranker; supports benchmark replay, live GPU queries, source-context labels, miss filters and human relevance labels.

Human labels are stored in `outputs/vimed_validation/human_labels.jsonl`.

## Docker

With Docker Desktop's Linux/WSL2 engine and NVIDIA GPU support enabled:

```powershell
docker compose up -d --build
```

If port 8001 is busy, set `$env:UI_PORT = '8002'` first. Use one GPU server at a time to avoid loading both models twice. See [Docker guide](docs/docker_vi.md).

## Prepare data and indexes

```powershell
.venv/Scripts/python.exe -m scripts.prepare_vimed
.venv/Scripts/python.exe -m scripts.prepare_vimed_validation
.venv/Scripts/python.exe -m scripts.build_dense_index --config configs/vimed_dense_bge_m3.yaml --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.verify_dense_index
```

The preparation command requires local raw parquets in `data/raw/vimedaqa/all`. It prepares the corpus and BM25 index. Dense build refuses to overwrite an existing index; skip the build if the verified index is already present. The current BGE-M3 index uses CLS pooling, FP16 CUDA and a pinned model revision.

## Validation benchmark

Run stages in this order:

```powershell
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage bm25 --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage dense --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage hybrid --timeout-seconds 600
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage rerank --timeout-seconds 7200
.venv/Scripts/python.exe -m scripts.report_vimed_validation
```

Completed compatible stages are reused. When changing data/config, use a new `--output` directory. The benchmark uses 2,210 validation questions and 17,955 deduplicated chunks. Positives are inferred source contexts, not exhaustive relevance judgments.

| Configuration | Recall@5 | Recall@10 | Recall@100 |
|---|---:|---:|---:|
| BM25 | 69.19% | 75.25% | 88.37% |
| BGE-M3 dense | 76.61% | 81.63% | 92.22% |
| Hybrid RRF | 77.19% | 82.26% | 92.49% |
| Hybrid + reranker | 88.64% | 90.63% | 92.49% |

Selected configuration: `configs/vimed_selected_validation.yaml`. Report: [validation benchmark](docs/vimed_validation_benchmark_vi.md).

## Batch retrieval, evaluation and submission

```powershell
.venv/Scripts/python.exe -m scripts.retrieve --config configs/vimed_selected_validation.yaml --queries data/vimed/validation/queries.jsonl --output outputs/predictions/vimed.jsonl
.venv/Scripts/python.exe -m scripts.evaluate --prediction outputs/predictions/vimed.jsonl --ground-truth data/vimed/validation/ground_truth.jsonl
.venv/Scripts/python.exe -m scripts.make_submission --predictions outputs/predictions/vimed.jsonl --chunks data/vimed/chunks.jsonl --queries data/vimed/validation/queries.jsonl
```

`src.api` provides a separate JSON API; set `R2AI_CONFIG` to the chosen config when using it. Demo fixtures and MedQuAD utilities remain for regression tests and optional retrieval experiments. They are not the default ViMed workflow.

## Tests and review

```powershell
.venv/Scripts/python.exe -m pytest -q
```

- [Code workflow review and cleanup](docs/workflow_review_vi.md)
- [Encoder profile and dense verification](docs/encoder_profiles_vi.md)
- [Historical retrieval review](docs/vimed_retrieval_review_vi.md)

Model caches, raw data, indexes, benchmark rankings and labels are local artifacts; keep them when transferring this workspace. `.venv` and model downloads are not bundled inside the Docker build context.

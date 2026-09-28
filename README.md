# R2AI Medical Retrieval

Hệ thống truy xuất thông tin y khoa cho câu hỏi tiếng Việt. Pipeline kết hợp **BM25**, **BGE-M3 dense retrieval**, **RRF fusion** và **BGE reranker**, sau đó trả về danh sách document/chunk liên quan. Dự án không sinh câu trả lời y khoa.

> Dự án phục vụ nghiên cứu retrieval. Kết quả không phải chẩn đoán, chỉ định điều trị hoặc tư vấn y khoa.

## Tổng quan

```text
Dữ liệu thô → chuẩn hóa corpus → BM25 index + dense index
                                      ↓
Query → BM25 + dense → RRF → reranker → chunk/document selection
                                      ↓
                         JSONL / API / Inspector UI
```

Tính năng chính:

- BM25, exact/FAISS dense retrieval và hybrid RRF.
- BGE cross-encoder reranking.
- Scoring và selection ở cấp chunk/document.
- Parent consistency và submission validator.
- Batch CLI, JSON API và giao diện phân tích retrieval.
- Recall@K, MRR, Precision, Recall và Macro F2.
- Manifest, checksum và model revision để tái lập thí nghiệm.

## Cấu trúc dự án

```text
configs/      Cấu hình pipeline
data/         Dữ liệu thô và dữ liệu đã chuẩn hóa
artifacts/    BM25/dense indexes
outputs/      Predictions, metrics và benchmark
scripts/      Chuẩn bị dữ liệu, build index, evaluate
src/          Pipeline, API và Inspector UI
tests/        Bộ kiểm thử
docs/         Thiết kế và báo cáo kỹ thuật
```

## Chạy nhanh không cần GPU

Demo dùng dữ liệu nhỏ trong `data/dev/` và không tải model.

```bash
cd medical-rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[api,dev]"

R2AI_CONFIG=configs/demo.yaml \
python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
```

Windows PowerShell dùng `.venv\Scripts\Activate.ps1` và:

```powershell
$env:R2AI_CONFIG = "configs/demo.yaml"
python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
```

Truy cập:

- API health: <http://127.0.0.1:8000/health>
- OpenAPI UI: <http://127.0.0.1:8000/docs>

Ví dụ truy vấn:

```bash
curl -X POST http://127.0.0.1:8000/search \
  -H "Content-Type: application/json" \
  -d '{"id":"demo-001","query":"Triệu chứng tăng huyết áp là gì?"}'
```

## Chạy ViMed Inspector

Luồng đầy đủ cần Python 3.11+, dữ liệu ViMed đã chuẩn bị và NVIDIA GPU cho dense/reranker live.

### 1. Cài dependency

```bash
python -m pip install -e ".[api,retrieval,dev]"
export HF_HOME="$PWD/.cache/huggingface"
```

### 2. Chuẩn bị dữ liệu và index

Đặt các file `train-*.parquet`, `validation-*.parquet`, `test-*.parquet` vào `data/raw/vimedaqa/all/`, sau đó chạy:

```bash
python -m scripts.prepare_vimed
python -m scripts.prepare_vimed_validation
python -m scripts.build_dense_index \
  --config configs/vimed_dense_bge_m3.yaml \
  --timeout-seconds 3600
python -m scripts.verify_dense_index
```

### 3. Chạy benchmark

Các stage phải chạy tuần tự:

```bash
python -m scripts.benchmark_vimed_validation --stage bm25 --timeout-seconds 1200
python -m scripts.benchmark_vimed_validation --stage dense --timeout-seconds 1200
python -m scripts.benchmark_vimed_validation --stage hybrid --timeout-seconds 1200
python -m scripts.benchmark_vimed_validation --stage rerank --timeout-seconds 7200
```

### 4. Mở Inspector UI

```bash
HF_HUB_OFFLINE=1 python -m uvicorn src.inspector:create_app \
  --factory --host 127.0.0.1 --port 8001 --workers 1
```

Mở <http://127.0.0.1:8001>. UI hỗ trợ benchmark replay, live query, so sánh bốn stage và lưu human labels.

## Docker

Docker hiện dành cho **NVIDIA GPU**, CUDA 12.8 và `linux/amd64`. Máy ARM hoặc máy không có NVIDIA không chạy được cấu hình này nếu không điều chỉnh.

Compose mount các thư mục `data/`, `artifacts/`, `outputs/` và `.cache/huggingface/`. Các thư mục và artifact ViMed cần được chuẩn bị trước.

```bash
mkdir -p data artifacts outputs .cache/huggingface
HF_HUB_OFFLINE=1 docker compose up -d --build

docker compose logs -f ui
curl http://127.0.0.1:8001/health
```

Cho phép tải model ở lần đầu:

```bash
HF_HUB_OFFLINE=0 docker compose up -d --build
```

Đổi cổng hoặc dừng service:

```bash
UI_PORT=8002 docker compose up -d
docker compose down
```

## Batch retrieval và evaluation

```bash
python -m scripts.retrieve \
  --config configs/vimed_selected_validation.yaml \
  --queries data/vimed/validation/queries.jsonl \
  --output outputs/predictions/vimed.jsonl

python -m scripts.evaluate \
  --prediction outputs/predictions/vimed.jsonl \
  --ground-truth data/vimed/validation/ground_truth.jsonl \
  --corpus-chunks data/vimed/chunks.jsonl
```

Tạo và kiểm tra submission:

```bash
python -m scripts.make_submission \
  --predictions outputs/predictions/vimed.jsonl \
  --chunks data/vimed/chunks.jsonl \
  --queries data/vimed/validation/queries.jsonl \
  --output outputs/submissions/submission.jsonl

python -m src.cli validate outputs/submissions/submission.jsonl \
  --corpus-chunks data/vimed/chunks.jsonl \
  --test-queries data/vimed/validation/queries.jsonl
```

## Cấu hình chính

| Cấu hình | Pipeline |
|---|---|
| `configs/demo.yaml` | Demo CPU không cần model |
| `configs/vimed_bm25.yaml` | BM25 |
| `configs/vimed_dense_bge_m3.yaml` | BGE-M3 dense |
| `configs/vimed_hybrid.yaml` | BM25 + dense + RRF |
| `configs/vimed_hybrid_rerank.yaml` | Hybrid + BGE reranker |
| `configs/vimed_selected_validation.yaml` | Cấu hình được chọn trên validation |

## Kết quả validation

Benchmark gồm 2.210 câu và 17.955 context tiếng Việt:

| Pipeline | Recall@5 | Recall@10 | Recall@100 |
|---|---:|---:|---:|
| BM25 | 69,19% | 75,25% | 88,37% |
| BGE-M3 dense | 76,61% | 81,63% | 92,22% |
| Hybrid RRF | 77,19% | 82,26% | 92,49% |
| Hybrid + reranker | 88,64% | 90,63% | 92,49% |

Ground truth là context nguồn suy ra từ dữ liệu QA, chưa phải relevance judgment đầy đủ. Kết quả dùng để so sánh nội bộ và không chứng minh độ an toàn lâm sàng hay khả năng retrieval đa ngôn ngữ.

## Kiểm thử

```bash
pytest -q
make lint
```

## Tài liệu

- [Đánh giá dự án hiện tại](docs/tong-quat.md)
- [Project brief](docs/00_project_brief.md)
- [Pipeline và workflow](docs/pipeline_workflow_presentation_vi.md)
- [Review workflow](docs/workflow_review_vi.md)
- [Review ViMed retrieval](docs/vimed_retrieval_review_vi.md)
- [Nhật ký nghiên cứu](docs/01_research_log.md)

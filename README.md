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
tests/
  unit/                     Data, retrieval và ranking dùng chung
  integration/              Runtime, API và tooling
  workflows/vibiomir/       Dataset chính thức
  workflows/vimedqa/        Benchmark lịch sử/regression
  workflows/medquad/        Regression tiếng Anh
  repository/               Kiểm tra cấu trúc repository
docs/         Thiết kế và báo cáo kỹ thuật
```

**Dataset chính thức:** ViBioMIR. ViMedQA và MedQuAD được giữ làm benchmark regression/đối chứng.

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
$env:HF_HOME = Join-Path (Get-Location) '.cache/huggingface'
$env:HF_HUB_OFFLINE = '1'
.venv/Scripts/python.exe -m uvicorn src.inspector:create_app --factory --host 127.0.0.1 --port 8001 --workers 1
```

Open **http://localhost:8001**. The UI compares BM25, dense, hybrid and hybrid + reranker; supports benchmark replay, live GPU queries, source-context labels, miss filters and human relevance labels.

Human labels are stored in `outputs/vimed_validation/human_labels.jsonl`.

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

```powershell
python -m pytest -q tests/workflows/vibiomir
python -m pytest -q tests/unit
python -m pytest -q tests/integration
python -m pytest -q tests/workflows/vimedqa tests/workflows/medquad
python -m pytest -q tests
make lint
```

Chi tiết từng luồng: [`tests/README.md`](tests/README.md).

## Tài liệu

- [Đánh giá dự án hiện tại](docs/tong-quat.md)
- [Review workflow](docs/workflow_review_vi.md)
- [Review ViMed retrieval](docs/vimed_retrieval_review_vi.md)

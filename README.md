# R2AI Medical Information Retrieval System

Hệ thống truy xuất thông tin y tế đa ngôn ngữ (Multilingual Medical Information Retrieval) phục vụ cuộc thi R2AI.

## Bắt đầu với bộ khung (chưa cần training)

Bản hiện tại có runtime demo chạy xuyên suốt, API tùy chọn, schema dữ liệu,
pipeline chọn kết quả, kiểm tra submission và đánh giá. Backend demo chỉ so khớp
từ khóa để kiểm tra luồng; BM25/dense/hybrid/reranker chưa được triển khai đầy đủ.
Các định dạng và quy tắc bên dưới là quy ước nội bộ, cần đối chiếu đề kỹ thuật BTC.

Chạy từ thư mục gốc với Python 3.11 trở lên:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,api]"
python -m pytest -q
python -m scripts.retrieve --config configs/demo.yaml --queries data/dev/sample_queries.jsonl --output outputs/predictions/demo.jsonl
python -m src.cli validate outputs/predictions/demo.jsonl --corpus-chunks data/dev/sample_chunks.jsonl --test-queries data/dev/sample_queries.jsonl
python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
```

API docs: <http://127.0.0.1:8000/docs>. `GET /health` kiểm tra trạng thái;
`POST /search` nhận `{"id":"q1","query":"tài liệu y khoa"}` và trả danh sách
`relevant_docs`, `relevant_chunks`. Truy vấn không khớp trả danh sách rỗng.

Chi tiết điểm mở rộng: [docs/architecture.md](docs/architecture.md).

Kế hoạch học và tự vận hành từng bước: [docs/smoke_test_learning_plan.md](docs/smoke_test_learning_plan.md).

Backlog cải thiện có thể thực hiện trước khi BTC công bố đầy đủ đề kỹ thuật:
[docs/pre_competition_backlog.md](docs/pre_competition_backlog.md).

## 1. Tổng quan bài toán
- **Input**: Vietnamese query và kho dữ liệu dạng chunk (`chunk_id`, `doc_id`, `chunk_index`, `language`, `text`).
- **Kho ngữ liệu**: Đa ngôn ngữ (Vietnamese `vi`, English `en`, Chinese `zh`).
- **Output**: JSONL chứa `id` (query ID), `relevant_docs` (list doc ID), `relevant_chunks` (list chunk ID).
- **Metric**: Precision, Recall, và Macro $F_2$ ở cả document-level và chunk-level:
  $$F_2 = \frac{5 \cdot P \cdot R}{4P + R}$$
  Điểm số được tính macro-average qua tất cả query.
- **Ràng buộc nhất quán cha-con (Parent Consistency)**: Tất cả chunk trong `relevant_chunks` bắt buộc phải có `doc_id` cha tương ứng nằm trong `relevant_docs`.

## 2. Cấu trúc thư mục
```
r2ai-medical-retrieval/
├── configs/               # Cấu hình thực nghiệm (BM25, Dense, Hybrid, Reranker)
├── data/
│   ├── raw/               # Dữ liệu gốc (queries.jsonl, chunks.jsonl)
│   ├── processed/         # Dữ liệu đã chuyển sang Parquet & doc_map
│   └── dev/               # Development set để calibrate threshold và đánh giá
├── artifacts/             # Sparse index, dense index, vector embeddings
├── src/
│   ├── data/              # Schemas Pydantic, data loader, tiền xử lý
│   ├── query/             # Normalization, mở rộng từ khóa y khoa
│   ├── indexing/          # Xây dựng sparse (Tantivy/BM25) & dense index (FAISS)
│   ├── retrieval/         # BM25, Dense (BGE-M3), Hybrid (RRF)
│   ├── reranking/         # Cross-encoder reranker (BGE-Reranker-v2-m3)
│   ├── scoring/           # Chunk score & Document aggregation
│   ├── selection/         # Dual thresholding & delta calibration cho F2
│   ├── evaluation/        # Macro P, R, F2 evaluator
│   └── submission/        # Xây dựng & kiểm tra tính hợp lệ submission
├── scripts/               # Scripts thực thi từng giai đoạn
├── tests/                 # Unit test suite cho schemas, metrics, submission
└── outputs/               # Kết quả predictions, evaluations, submissions
```

## 3. Lộ trình triển khai (Day 1 - Day 7)
- **DAY 1**: Cấu trúc repo, Pydantic schemas, loaders, submission validator, metric evaluator, test suites.
- **DAY 2**: Sparse retrieval (BM25 baseline).
- **DAY 3**: Dense multilingual retrieval (BGE-M3 + FAISS).
- **DAY 4**: Hybrid retrieval với Reciprocal Rank Fusion (RRF).
- **DAY 5**: Cross-Encoder Reranking (BGE Reranker v2 M3).
- **DAY 6**: Grid search calibration threshold và relative delta theo Macro $F_2$.
- **DAY 7**: Medical dictionary và multilingual query expansion.

## 4. Hướng dẫn nhanh (Day 1 Foundation)
```bash
# Cài đặt môi trường & dependencies
uv pip install -e .

# Chạy unit tests
python -m pytest tests/ -v

# Chuẩn bị dữ liệu và tạo doc_map
python scripts/prepare_data.py --input-chunks data/dev/sample_chunks.jsonl --output-dir data/processed

# Đánh giá submission mẫu
python scripts/evaluate.py --prediction outputs/submissions/sample_submission.jsonl --ground-truth data/dev/sample_ground_truth.jsonl

# Validate submission
python -m src.submission.validate outputs/submissions/sample_submission.jsonl --corpus-chunks data/dev/sample_chunks.jsonl --test-queries data/dev/sample_queries.jsonl
```

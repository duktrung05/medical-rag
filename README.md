# R2AI Medical Information Retrieval System

Hệ thống truy xuất thông tin y tế đa ngôn ngữ (Multilingual Medical Information Retrieval) phục vụ cuộc thi R2AI.

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

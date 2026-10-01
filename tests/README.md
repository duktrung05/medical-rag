# Luồng kiểm thử

Test được chia theo phạm vi để có thể chạy và đọc kết quả từng luồng độc lập.

```text
tests/
├── unit/
│   ├── data/           schema và quan hệ document/chunk
│   ├── retrieval/      BM25, dense, hybrid, encoder và reranker
│   └── ranking/        metric, scoring và selection
├── integration/
│   ├── runtime/        API, Inspector, service và retrieve CLI
│   └── tooling/        config, dataset CLI, build timeout, submission
├── workflows/
│   ├── vibiomir/       luồng dataset chính thức
│   ├── vimedqa/        benchmark lịch sử và regression
│   └── medquad/        regression tiếng Anh
└── repository/         kiểm tra cấu trúc source, config và tài liệu
```

## Chạy theo luồng

```powershell
# Gate chính cho dữ liệu chính thức
python -m pytest -q tests/workflows/vibiomir

# Core dùng chung
python -m pytest -q tests/unit

# API, runtime và CLI
python -m pytest -q tests/integration

# Benchmark/regression cũ
python -m pytest -q tests/workflows/vimedqa tests/workflows/medquad

# Toàn bộ repository
python -m pytest -q tests
```

Fixture tạo bằng `tmp_path` nằm cùng test của workflow sử dụng nó. Dữ liệu thật,
output benchmark và model cache không đặt trong `tests/`.

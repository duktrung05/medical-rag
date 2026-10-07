# Kiểm chứng final evidence selector — 2026-10-06

Selector đã được tích hợp vào `r2ai/rank.py`. Mặc định giữ chế độ Top-K để so
sánh; `--selector threshold` bật relevance filtering. Hướng dẫn chạy, cache,
gán nhãn và hiệu chỉnh nằm trong `r2ai/README.md`.

## Hành vi đã triển khai

- Lọc chunk bằng raw cross-encoder logit, thêm relative gate tùy chọn.
- Giữ riêng dense score; chunk chưa được rerank có score `null`, không dùng
  score giả để quyết định relevance.
- Cho phép chọn 0 evidence, không backfill; áp dụng document/chunk/per-document caps.
- Loại text trùng hoàn toàn trong chế độ threshold, giữ nguyên text được chọn.
- Các chế độ document `parents`, `baseline`, `threshold`; bảo đảm quan hệ cha.
- Cache JSONL/gzip tự chứa text, query, score, thứ tự và fingerprint đầu vào.
- Pin model revision, gắn embedding cache với nội dung đầu vào và encoder config.
- Replay và calibration chạy bằng thư viện chuẩn Python, không import Torch.
- Calibration chọn trên dev, kiểm tra holdout độc lập, tính đủ query trả rỗng,
  phân biệt nhãn toàn corpus và nhãn trong candidate pool.

## Kiểm tra

Lệnh kiểm tra trong môi trường CPU có NumPy/PyArrow/Pytest:

```bash
python -m pytest tests/workflows/vibiomir/test_evidence_selector.py \
    tests/workflows/vibiomir/test_taxonomy.py -q
```

Kết quả: **41 passed**. Các kiểm tra gồm authentic/null scores khi partial
rerank, threshold boundary, abstention, caps, deduplication, parent consistency,
cache round-trip/truncation, baseline equivalence, dev/holdout separation và
pipeline với pool/corpus rỗng. Model được thay bằng fixture trong integration test;
đây không phải kiểm chứng chất lượng hay tương thích inference GPU thực tế.

Ruff kiểm tra các module mới, `rank.py`, `validate.py` và test mới: không còn lỗi.

CLI calibration → CPU replay → validator đã chạy thành công trên fixture giả
lập có dev/holdout và query không có evidence. Fixture này chỉ kiểm tra hành vi;
điểm của nó không phản ánh precision trên ViBioMIR.

Validator bổ sung kiểm tra chunk cha và evidence trùng. ZIP hiện có
`outputs/submissions/vibiomir_test_best_20261006.zip` vẫn **VALID**: 1.200 queries,
200 documents/query, 120 chunks/query, khoảng 44,2 MB. ZIP không được thay đổi.

## Chạy ZIP thử nghiệm trên dữ liệu thật

Sau kiểm tra triển khai ban đầu, đã chạy BGE reranker thật trên GPU `cuda:1`
cho toàn bộ 144.000 cặp query–chunk của ZIP baseline. 121.508 chunk riêng biệt
được đối chiếu text với corpus gốc trước inference. Môi trường mới dùng Torch
2.6.0+cu126, sentence-transformers 5.1.2 và transformers 4.55.4; model revision
được pin và ghi trong cache. Code cập nhật đã qua **43 tests** và lint.

ZIP nộp thử: `outputs/evidence/test_20261006/selected/submission.zip`.
Kết quả validator **VALID**, 1.200 query, 39,31 MB. Document predictions giữ
nguyên baseline (200/query). Selector raw threshold `-6`, document mode
`baseline`, giữ 120.262/144.000 chunk: loại 12.403 chunk dưới ngưỡng và 11.335
chunk trùng text. Không có query rỗng evidence; text của tất cả chunk được giữ
khớp baseline. Toàn bộ cache/checkpoint là score thật, không dùng score giả.

Chi tiết có trong `outputs/evidence/test_20261006/experiment_report.json` và
`experiment_report.md`. Môi trường CPU ban đầu thiếu model dependencies đã được
thay bằng môi trường GPU riêng cho lượt test. Đã bổ sung thu gom reference cycle
của CrossEncoder sau mỗi block để ổn định bộ nhớ khi chạy có checkpoint.

## Chưa có điểm benchmark

Chưa có nhãn dev/holdout để hiệu chỉnh selector hoặc điểm leaderboard của ZIP
mới. Vì vậy chưa kết luận mức tăng precision trên ViBioMIR. Ngưỡng `-6` là cấu
hình thử nghiệm cố định; ngưỡng 0 trong config ví dụ cũng chỉ mang tính minh họa.

Evaluator mới dùng macro set metrics theo `(doc_id, chunk_index)`, không thay thế
scorer chính thức có luật matching chunk text. Khi dùng nhãn pool, recall chỉ
phản ánh pool đã gán nhãn. Có thể gán nhãn từ cache score thật hiện có, hiệu chỉnh
dev và xác nhận holdout trước khi chọn cấu hình chính thức.

# Kết quả Task 4: nghiệm thu dense và full index

Ngày 21/10/2026, giờ Việt Nam. Phạm vi: thực hiện I0–I4 của [plan full index](v4_task4_full_index_execution_plan_20261021_vi.md). **Kết quả cuối: `FULL_INDEX_PASS`.** Điểm dừng hiện tại: trước inference 1.200 official queries, export và nộp BTC.

## Kết quả đã xác nhận

| Gate | Kết quả | Bằng chứng |
|---|---|---|
| Dense release | `DENSE_FULL_PASS` | `outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550/dense_full_integrity.json` |
| Combined index CPU | `FULL_INDEX_ARTIFACTS_PASS` | `.../cpu_acceptance.json`, `.../full_integrity.json`, `.../full_coverage.json` |
| Full corpus smoke | `FULL_QUERY_SMOKE_PASS` | `.../full_query_smoke.json`, 20 query checkpoints, score cache và launch report |
| Tổng hợp | `FULL_INDEX_PASS` | `.../full_index_acceptance.json` |

Thư mục reports chung: `outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550/` (mọi đường dẫn `.../` trong bảng nằm dưới thư mục này). Combined index: `outputs/v4/task4/full_index_gpu3_20261019/`. Manifest SHA-256: `efdad2cf0e1db51730f7eaf0f931242e9bba9d5b0998910f6cb168f8dede6941`.

### Dense và combined index

- Dense verifier đã rehash vectors, mapping, contract và kiểm SQLite `integrity_check`/metadata/row sequence/unique parents. Ordered mapping được đối chiếu **từng row** với 922 Phase 2 shards, eligible sparse store, input hash `title + newline + passage`, source offsets/URL/provenance và 49 excluded passages. Kết quả đúng **2.951.224 rows**, **1.073.525 parents**, chiều **1.024**.
- Toàn ma trận được quét theo block 8.192 rows: dtype FP32, finite và norm trong tolerance `[0.99, 1.01]`; không encode lại hay sửa vectors đã publish.
- Sparse reports đã PASS trước đó được nối qua exact sparse manifest SHA. Assembler CPU tạo combined index bằng symlinks tới các source đã khóa. Verifier kiểm exact symlink targets, sáu file hashes và count/contract. Bước tổng hợp cuối rehash lại sáu file sau smoke; tất cả vẫn khớp.
- `FULL_INDEX_ARTIFACTS_PASS` là gate CPU bổ sung theo plan. `FULL_INDEX_PASS` chỉ được ghi sau smoke GPU thành công.

### Smoke trên full corpus

- Đúng 20 official query IDs từ canary reference đã khóa; `result_limit=200`, A0 `document_selection=with_chunks`, dense/reranker cùng trên local `cuda:0` sau khi launcher mask duy nhất GPU vật lý 3 UUID `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7`.
- **20/20 complete, 0 failed**; audit đủ **4.000 rerank pairs** trong score cache, selection replay khớp, grounding/source offsets/provenance và budget đều PASS. Hai lượt repeat query 92/927 khớp ranked candidates và selected results với checkpoint gốc.
- Tổng selected documents trong 20 query là **1.200** (60/query theo cap A0); đây **không phải** 1.200 official queries của lượt inference còn chờ. Có **193/4.000 rerank pairs** vượt 512 tokens trước truncation; giữ nguyên baseline 512-token đã khóa.
- Latency chỉ cho 20 query smoke: mean **248,75 s**, P50 **185,84 s**, P95 nearest-rank **589,63 s**, max **840,50 s**; model/index load và hash ban đầu **49,05 s**. Query dài tốn nhiều CPU vì BM25/postings và exact dense search trên full corpus. Không ngoại suy tuyến tính để hứa thời gian 1.200 queries.
- Process peak RSS khoảng **15,69 GiB**, PyTorch GPU peak reserved khoảng **3,99 GiB**; nằm trong budget 32 GiB RSS của plan. GPU 3 rảnh sau khi smoke hoàn tất.

## Triển khai và kiểm thử

Các scripts mới: `scripts/v4/verify_release_dense.py`, `verify_full_index.py`, `smoke_full_index.py`, `finalize_full_index_acceptance.py`; test mới `tests/v4/test_full_index_acceptance.py`. Bốn scripts chạy theo thứ tự, không sửa builder/encoder/disk dense đã khóa hoặc source release. Tất cả PASS/FAIL được ghi vào attempt directory riêng.

`pytest -q tests/v4/test_full_index_acceptance.py tests/v4/test_disk_dense.py tests/v4/test_score_cache.py tests/v4/test_submission_validation.py`: **10 passed**. `pytest -q tests/v4 --ignore=tests/v4/test_cli_api_tail.py`: **59 passed**. Lượt full `pytest -q tests/v4` trước đó không hoàn tất và đã được ngắt khi bị kẹt, vì vậy không báo là full suite PASS; các test API bị loại trong lượt 59 không được dùng làm bằng chứng cho full index.

## Điểm dừng và giới hạn kết luận

Không có `outputs/v4/task4/full_a0_gpu3_20261019/`, `predictions.json` hay `submission.zip` của full Task 4. `FULL_INFERENCE_PASS`, `FULL_V4_EXPORT_PASS` và F2 vẫn chưa đo. Tập 20 query là smoke vận hành/grounding, không phải bằng chứng điểm relevance hoặc parity kết quả với snapshot 1.000 docs.

Bước tiếp theo nếu có chỉ đạo mới là review findings R3–R6 trong [review sau dense](v4_task4_post_dense_plan_review_20261021_vi.md), chuẩn bị runner inference/resume/cache audit, rồi mới mở 1.200 queries. Lượt này kết thúc đúng tại `FULL_INDEX_PASS`.

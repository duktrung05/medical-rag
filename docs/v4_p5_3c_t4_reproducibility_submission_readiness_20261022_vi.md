# P5.3C — Tesla T4 reproducibility và submission readiness

Ngày chuẩn bị: 22/10/2026.

## Trạng thái hiện tại

| Gate | Trạng thái | Lý do |
|---|---|---|
| Exact parity với P2-A0 reference | `NO-GO` | 608/1.200 scores khác và BTC payload khác; kết quả P5.3B được giữ nguyên |
| T4 reproducibility | `UNVERIFIED` | Mới có một fresh T4 run; máy server hiện không có CUDA driver |
| Local BTC submission format | `GO` | Payload và ZIP qua validator, source grounding và caps |
| Official BTC metric semantics | `UNVERIFIED` | Repository không có official evaluator hoặc scoring contract |
| `A0_T4_V1` full inference gate | `NO-GO` | Chưa import fresh T4 repeat run độc lập |

Không có P2-A0 config, score, pair, final hoặc export reference nào bị sửa.
`A0_T4_V1` dùng output namespace riêng và không lấy exact parity với mixed-hardware
reference làm điều kiện promotion.

## Repeat T4 canary

Notebook `notebooks/p5_3c_t4_repro.ipynb` dùng lại bundle SHA-256
`2d3869ea89c6e5b8d7c9943e3fbeb4fa8d78a2b358c2ec5409103a62b52e57f7`
và ghi vào `MyDrive/medical-rag/P5_3C_T4_REPEAT`. Người chạy chỉ cần chọn T4,
Run All và upload bundle. Namespace tách biệt ngăn notebook lấy sáu checkpoint
của P5.3A làm repeat evidence.

Sau khi đưa `p5_3c_t4_repeat_scores.zip` về server:

```bash
.venv-phase2-batch/bin/python -m scripts.v4.verify_p5_3c_t4_reproducibility \
  --repeat-zip outputs/v4/p5_3c_colab_import/p5_3c_t4_repeat_scores.zip
```

Verifier kiểm tra cả hai ZIP bằng P5.3A integrity contract, yêu cầu session IDs
không giao nhau, đúng Tesla T4/runtime/model snapshot, 1.200 fresh model calls
mỗi run, zero cache hits, rồi so exact float scores, pair ranking, selected full
objects và canonical BTC payload. Dùng lại cùng ZIP làm hai input luôn trả
`NO-GO`, ngay cả khi 1.200 scores giống tuyệt đối.

## BTC submission contract

Local payload chỉ cho phép ba keys `id`, `relevant_docs`, `relevant_chunks`.
`rerank_score` không được serializer xuất ra; thêm trường này ở record hoặc
chunk làm local validator từ chối. Serializer giữ thứ tự document/evidence.
Validator chấp nhận payload khi chỉ hoán vị hai arrays, còn evaluator F2 nội bộ
tính theo membership và không nhạy thứ tự.

Trang công khai chính thức mô tả đầu ra document/chunk retrieval và metric
`Macro F2 (Recall-Weighted)`, nhưng không công bố evaluator code, cutoff, rank
weighting hoặc quy tắc xử lý thứ tự arrays.

Các kết luận local không chứng minh BTC evaluator chính thức xử lý order giống
như vậy. Official order sensitivity và metric equivalence giữa A0 với
`A0_T4_V1` vì thế giữ `UNVERIFIED`.

## Điều kiện promotion `A0_T4_V1`

T4 reproducibility chỉ `GO` khi đồng thời đạt:

1. Hai result ZIP khác nhau và session provenance độc lập.
2. Cùng pinned runtime, model revision và toàn bộ model snapshot hashes.
3. 1.200/1.200 scores giống tuyệt đối, hữu hạn và đúng input identities.
4. Pair ranking, selected documents/chunks/full objects giống tuyệt đối.
5. Canonical BTC payload giữa hai T4 runs giống tuyệt đối.
6. Local submission validation và source grounding vẫn đạt.

Exact parity với P2-A0 tiếp tục được báo riêng là `NO-GO`; gate không điều chỉnh
model để ép T4 khớp cache mixed hardware.

## Full inference commands

`scripts/v4/run_a0_t4_v1_full.sh` đã chứa các lệnh prepare, T4 score, seal score
manifest và CPU finalize. Script đọc `submission_readiness_report.json` và dừng
ngay nếu status hoặc `full_inference_gate` chưa phải `GO`.

Khi được chạy sau promotion, workflow ghi vào
`outputs/v4/a0_t4_v1_full_20261022`, kiểm tra frozen config hash, tạo immutable
pair shards, khóa exact T4 runtime và model snapshot hashes, giữ score shard và
SQLite checkpoints để resume, seal hashes của đủ 240.000 scores, rồi mới chạy
existing CPU finalization và exporter.

Trong P5.3C hiện tại, các lệnh full chỉ được chuẩn bị; chưa chạy retrieval hoặc
inference cho 1.200 queries.

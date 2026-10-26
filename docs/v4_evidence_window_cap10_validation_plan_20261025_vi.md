# Kế hoạch bước tiếp: khóa cap 10 và kiểm tra validation

Trạng thái: **EXECUTED — SCREENED_CANDIDATE**. Validation và held-out đều
PASS gate đã khóa. Chi tiết:
[execution report](v4_evidence_window_cap10_validation_execution_20261025_vi.md).

## Quyết định duy nhất

Giữ nguyên package evidence-window, model, ba windows 192 tokens, max aggregation,
fixed contexts, S2 documents và 60 chunk slots. Chỉ đổi giới hạn allocation từ
6 lên **10 swaps/query**.

Cap 10 là mức nhỏ nhất trong cached development diagnostic đạt ngưỡng cũ:
manual 3 gains/0 losses, historical preservation 8 gains/1 loss. Vì cap được
chọn sau khi xem development, development từ đây là tập chọn policy, không còn
là bằng chứng xác nhận độc lập.

## Evaluation boundary

- Validation: dùng nguyên 20 query IDs và 7.062 pairs đã khóa trước khi scoring.
  Không thay cohort hoặc context theo kết quả.
- Gate validation: documents/chunk count/identity PASS; manual net không âm;
  không mất manual-visible evidence; historical preservation net không âm và
  historical losses không quá 2 queries.
- Nếu validation fail: STOP, không chạy held-out.
- Nếu validation PASS: chạy nguyên 10 held-out queries, 3.835 pairs. Gate giống
  validation nhưng historical loss tối đa 1 query.
- Held-out không có manual outsider; nó chỉ kiểm regression, không thể tự chứng
  minh gain relevance.

Validation và held-out labels lịch sử không phải pristine benchmark. Kết quả qua
cả hai chỉ mang trạng thái `SCREENED_CANDIDATE`; official F2 vẫn cần full output
và BTC. Không mở full 1.200 queries trong bước này.

## Runtime và dừng lỗ

Tái sử dụng exact locked pair shards và parity 128/128 của parent package.
Chấm tối đa 7.062 pairs validation và chỉ thêm 3.835 pairs held-out nếu gate mở.
Giới hạn 15 GPU phút/stage. Mọi sai hash, runtime, source identity, non-finite
score hoặc document equality đều fail-closed.

Namespace mới: `outputs/v4/evidence_window_cap10_validation_20261025/`.

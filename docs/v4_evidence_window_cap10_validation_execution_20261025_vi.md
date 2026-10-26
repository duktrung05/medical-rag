# Cap-10 evidence-window — kết quả validation và held-out

Ngày chạy: 2026-10-25. Trạng thái: **SCREENED_CANDIDATE**.

## Treatment đã khóa

Giữ nguyên S2 documents, fixed contexts, BGE model, ba windows 192 tokens và
max aggregation. Thay đổi duy nhất so với pilot trước là tối đa 10 thay vì 6
chunk-parent swaps/query. Development được dùng chọn cap và không được tính là
independent validation.

## Kết quả

| Stage | Queries / pairs | Swaps | Manual gain/loss | Historical gain/loss | Decision |
|---|---:|---:|---:|---:|---|
| Development, policy selection | 30 / 11.213 cached | 290 | 3/0 | 8/1 | chọn cap 10 |
| Validation | 20 / 7.062 | 176 | 0/0 | 3/0 | OPEN_HELD_OUT |
| Held-out | 10 / 3.835 | 94 | 0/0 | 3/1 | SCREENED_CANDIDATE |

Validation model time 16,96 giây; held-out 8,65 giây. Documents giữ nguyên
từng query, mỗi output giữ 60 chunks và toàn bộ structural validators PASS.

Validation/held-out không làm mất manual-visible evidence. Historical proxy
trên hai stage có tổng 6 gains/1 loss. Đây không phải official evidence recall
và không chứng minh FINAL/F2 tăng.

Manual target duy nhất nằm ngoài slots ở validation là `71:311495`. Fixed
context chứa evidence nhưng target ở entrant position 15 nên cap 10 không chọn.
Không tăng cap tiếp theo kết quả này vì đó sẽ là tuning trên validation.

Held-out historical loss là `1165:196917`, bị thay bởi `510485`; evidence-window
scores lần lượt -2,2266 và -2,1289. Loss đã nằm trong budget một query nhưng cho
thấy treatment không đơn điệu tốt hơn S2.

## Quyết định kỹ thuật

Cap 10 đã vượt regression screens và tốt hơn cap 6 trên development. Tuy vậy,
bằng chứng relevance độc lập vẫn yếu vì validation/held-out không có manual
gain; chúng chủ yếu đo preservation. Không gọi đây là score improvement và S2
vẫn là champion cho đến khi BTC đo official F2.

Bước hợp lý tiếp theo là tạo đúng **một** full 1.200-query cap-10 candidate,
không sweep cap/window khác, rồi submit BTC. Upper bound 504.000 pairs; theo
throughput pilot khoảng 430 pairs/giây, model time ước tính gần 20 phút. Trước
full run phải khóa all-query manifest, exact S2 document equality và ZIP gate.
Nếu không còn BTC slot hoặc không định submit, không nên chạy full inference.

## Artifacts

- Namespace: `outputs/v4/evidence_window_cap10_validation_20261025/`
- Lock: `lock.json`
- Validation report: `reports/validation.json`
- Held-out report: `reports/held_out.json`
- Implementation: `scripts/v4/run_evidence_window_cap10_validation.py`
- Tests: 44 PASS.

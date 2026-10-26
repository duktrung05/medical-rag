# Evidence-window allocation — kết quả thực thi

Ngày chạy: 2026-10-25. Quyết định: **STOP_DEVELOPMENT**.

## Kết quả chính

S2 vẫn là champion và không bị sửa. ZIP SHA256 vẫn là
`081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.

Pilot đã khóa 60 queries và 22.110 scoring pairs; development gồm 30 queries,
11.213 pairs. Exact-batch runtime parity PASS 128/128, max absolute difference
0.0. GPU model time là 24,95 giây, throughput 449,46 pairs/giây.

Arm B giữ documents giống hệt S2, thay tối đa 6/60 chunk parents mỗi query.
Development đổi 180 slots trên 30/30 queries và cho kết quả:

- Manual evidence: 2 gains (`5:313055`, `747:51610`), 0 losses.
- Historical preservation proxy: 5 gains, 0 losses.
- Structural validators và document equality: PASS.
- Official F2: chưa đo.

Gate đã khóa yêu cầu ít nhất 3 manual gains trên 3 queries. Pilot chỉ đạt 2,
do đó dừng trước validation và held-out; không tiêu thêm 10.897 pairs.

## Lỗi nào đã được kiểm tra

Thử nghiệm kiểm tra giả thuyết điểm document relevance của S2 không đủ phù hợp
để cấp 60 evidence slots. Ba cửa sổ evidence 192 tokens tạo tín hiệu tốt hơn
S2 ở một số ca và không làm mất known evidence trong treatment khóa, nhưng với
budget 6 swaps/query tín hiệu chưa đủ vượt gate.

Phân rã 11 manual outsiders bằng scores development đã có:

- 10/11 fixed contexts chứa manual evidence; q566 là lỗi context selection vì
  context được chọn không chứa manual evidence.
- Hai targets được chọn với 6 swaps; bốn targets khác sẽ vào ở vị trí entrant
  10, 11, 12 và 15; năm targets còn lại không đủ score để swap.
- Vì vậy kết quả này không chứng minh evidence-window scoring vô ích. Nó cho
  thấy policy 6 swaps quá chặt, đồng thời scorer vẫn không xếp đủ cao 5 targets
  và context selection còn sai ít nhất một target.

## Phân tích counterfactual chỉ dùng cached development scores

Đây là phân tích post-hoc, không phải kết quả gate và không được gọi là một
challenger đã validate:

| Max swaps/query | Manual gain/loss | Historical gain/loss | Kết luận |
|---:|---:|---:|---|
| 6 | 2/0 | 5/0 | Kết quả khóa, STOP |
| 10 | 3/0 | 8/1 | Đạt các ngưỡng development nếu policy này được khóa từ đầu |
| 12 | 5/0 | 11/1 | Nhiều gain hơn nhưng tăng mức can thiệp |
| 16 | 6/0 | 11/3 | Sát loss budget, rủi ro cao hơn |

Phương án tiếp theo có căn cứ là khóa **max 10 swaps/query** — mức nhỏ nhất đạt
development threshold trong cached diagnostic — rồi chấm nguyên validation
chưa mở. Không được đổi thêm window size, aggregation, context policy hoặc
threshold cùng lúc. Nếu validation mất manual-visible evidence hoặc historical
net âm thì dừng nhánh này. Nếu qua validation và held-out regression gate, mới
dự toán full 1.200-query inference và BTC; chưa có căn cứ nâng S2 ở thời điểm
hiện tại.

Audit DOCS cap 140→160 trên ba R7 replay queries thu hồi một known-positive ở
q53 nhưng thêm 59 documents chưa được judged. Đây chỉ là diagnostic, chưa đủ
căn cứ thay document cap.

## Artifact

- Package, scores và reports: `outputs/v4/evidence_window_allocation_20261025/`
- Development report: `reports/development.json`
- Swap rows: `swaps/development.json`
- DOCS cap audit: `docs_cap_audit.json`
- Implementation: `scripts/v4/run_evidence_window_allocation.py`
- Regression tests: 43 PASS.

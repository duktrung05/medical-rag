# Chuẩn bị candidate phục hồi evidence end-to-end — 26/10/2026

## Policy đã khóa

Một policy duy nhất, áp dụng trên 1.200 query:

- Retrieval ba nhánh ở depth 800; gom parent trước khi cắt; giữ toàn bộ
  documents cap-10; tối đa 800 parent mới và chấm top 200 parent mới.
- Lấy mọi indexed passages trong từng parent. Sinh cửa sổ 192 tokens, stride
  96; chấm bằng BGE reranker CUDA FP16 đã khóa; dựng context quanh window thắng,
  tối đa 5.500 ký tự, giữ context cap-10 làm fallback.
- Tối đa 10 chunk-parent replacements/query, trong đó tối đa 3 parent mới;
  document mới chỉ xuất hiện khi nhận chunk slot. Giữ 60 slots, tối đa một
  chunk/parent.
- Tạo một ZIP và gọi BTC tối đa một lần. Không sweep policy.

Effective policy và hashes được lưu ở
`outputs/v4/end_to_end_recovery_20261026/policy_lock.json`.

## Kỹ thuật đã kiểm tra

Ba query canary `53, 130, 380`:

- SQL-aggregate document/passsage BM25 trả đúng thứ tự 800 identities so với
  implementation tham chiếu ở cả 6 branch-query. Sai khác score tối đa
  `7.11e-14`.
- Dense FP32 và FP64 có cùng thứ tự 800 identities trên cả ba query.
- Prefix parity/fusion với frozen S2 khớp trên ba query theo canary R11.
- Contract test policy: **1 passed**.

Chi tiết từng branch nằm trong
`outputs/v4/end_to_end_recovery_20261026/technical_preflight.json`.

## Trạng thái dừng

Artifact hiện ở `WAITING_FOR_PINNED_GPU`. `nvidia-smi` không kết nối được
driver; pinned scorer environment báo `torch.cuda.is_available() == false` và
0 thiết bị. Vì vậy chưa chạy 1.200-query inference, chưa tạo reranker score,
candidate ZIP hoặc BTC receipt. Kế hoạch yêu cầu giữ cache FP16 tách biệt và
không trộn CPU logits vào score đó; trạng thái host này vì thế không thể mở
score gate an toàn.

R11 đã đo raw retrieval CPU cho 94 query dự kiến khoảng 22.793 giây CPU, cao
hơn budget 14.400 giây. Batch dense GPU path ở 1.200 query chưa được thực thi
hoặc validate trên host này. Candidate full chỉ nên mở trên host có GPU đúng
runtime đã khóa; tiếp tục CPU retrieval toàn cohort sẽ vượt ngân sách.

Stage trước của lượt này đã tạo pool diagnostic 87/87 manual spans tại
`outputs/v4/manual87_parent_recovery_20261026/`. Pool đó dùng depth 4.000 cho
11 query misses để đo coverage, không được dùng làm retrieval policy candidate
depth 800.

## Mã và artifacts

- Lock/preflight runner: `scripts/v4/prepare_end_to_end_evidence_recovery.py`.
- Contract: `tests/v4/test_end_to_end_recovery_policy.py`.
- Status/runtime/preflight artifacts: `outputs/v4/end_to_end_recovery_20261026/`.
- Candidate output chính thức chưa được tạo; cap-10 vẫn là champion.

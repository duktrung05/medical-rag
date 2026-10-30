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
- Contract tests policy, allocator và batched dense: **3 passed**.

Chi tiết từng branch nằm trong
`outputs/v4/end_to_end_recovery_20261026/technical_preflight.json`.

## Trạng thái thực thi hiện tại

Runner của candidate đã được viết tại
`scripts/v4/run_end_to_end_evidence_recovery.py`. Nó có ba stage `retrieve`,
`score`, `export`, checkpoint theo query và xuất ZIP sau khi source/slot/ZIP
validation PASS. `notebooks/v4_end_to_end_evidence_recovery_colab.ipynb` là
launcher cho Colab với GPU CUDA tương thích. Candidate chấm lại toàn bộ
seed/window trên GPU mới, không so logits với A6000 cũ và không dùng ngưỡng
manual trước khi xuất ZIP.

Host hiện tại vẫn không có CUDA: `nvidia-smi` không kết nối được driver và
`torch.cuda.is_available() == false`. Lệnh `retrieve --start 1 --stop 1` đã
được thử và dừng trước khi tạo retrieval shard. Vì vậy chưa có full inference,
reranker score, candidate ZIP hoặc BTC receipt. Không có API phiên Colab đang
kết nối trong workspace này; notebook cần được mở trong Colab và Drive phải
có repository, index backing files khoảng 40 GB và cap-10 artifacts.

R11 đã đo raw retrieval CPU cho 94 query dự kiến khoảng 22.793 giây CPU, cao
hơn budget 14.400 giây. Batch dense GPU path ở 1.200 query chưa được thực thi
hoặc benchmark trên host này. Thời gian chạy và khả năng giữ phiên Colab đủ
lâu chưa được xác nhận; CPU retrieval toàn cohort sẽ vượt ngân sách.

Stage trước của lượt này đã tạo pool diagnostic 87/87 manual spans tại
`outputs/v4/manual87_parent_recovery_20261026/`. Pool đó dùng depth 4.000 cho
11 query misses để đo coverage, không được dùng làm retrieval policy candidate
depth 800.

## Mã và artifacts

- Lock/preflight runner: `scripts/v4/prepare_end_to_end_evidence_recovery.py`.
- Full runner: `scripts/v4/run_end_to_end_evidence_recovery.py`.
- Colab launcher: `notebooks/v4_end_to_end_evidence_recovery_colab.ipynb`.
- Contracts: `tests/v4/test_end_to_end_recovery_policy.py` và
  `tests/v4/test_end_to_end_recovery_runner.py`.
- Status/runtime/preflight artifacts: `outputs/v4/end_to_end_recovery_20261026/`.
- Candidate output chính thức chưa được tạo; cap-10 vẫn là champion.

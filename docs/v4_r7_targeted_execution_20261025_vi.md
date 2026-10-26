# Kết quả R7 targeted parent loss — 25/10/2026

Trạng thái: **POOL GATE PASS; SCORE GATE BLOCKED vì không có pinned CUDA runtime**.
S2 tiếp tục là champion. Không có reranker score mới, candidate ZIP hoặc BTC result.

## Phạm vi đã chạy

Cohort được khóa trước retrieval gồm 30 development queries: 8 manual misses,
12 historical misses và 10 retained controls. Mọi label trong cohort đều có
source trong eligible index; evidence span round-trip đúng source và hash.

Ba query 53, 130 và 380 được chạy canary trên CPU. Document BM25 top-400,
passage BM25 top-200, dense top-200 và fused mixed-item top-400 đều khớp chính
xác IDs và thứ tự với explicit frozen traces. Canary mất 198,38 giây.

Sau canary, toàn bộ 30 queries được trace trong 1.922,38 giây, thấp hơn budget
7.200 giây. Không chạy reranker, rebuild index hoặc submission export.

## Stage attribution

| Cohort | Raw union miss | Bị cắt ở fused top-400 | Còn trong S2 docs |
|---|---:|---:|---:|
| Manual misses | 4 | 4 | 0 |
| Historical misses | 8 | 4 | 0 |
| Controls | 0 | 0 | 10 |

Bốn manual fusion drops được kiểm sâu:

| Query / parent | Raw rank | Mixed-item rank | Unique-parent rank | Parent floor 400 |
|---|---:|---:|---:|---|
| 610 / 305379 | dense 142 | 406 | 285 | Recover |
| 991 / 528284 | dense 193 | 576 | 442 | Không recover |
| 1035 / 546970 | document BM25 344 | 704 | 488 | Không recover |
| 1159 / 514656 | dense 148 | 433 | 310 | Recover |

Query 610 và 1159 mất vì hệ thống cắt 400 **items** trước khi gom parent; các
duplicate items của parent khác chiếm slot. Query 991 và 1035 vẫn nằm ngoài
top-400 unique parents, vì vậy không được tính là lỗi cùng loại.

## Patch và replay trước/sau

Patch thêm policy opt-in `fused_parent_limit`. Policy giữ nguyên toàn bộ prefix
mixed items, scores và thứ tự, sau đó bổ sung item đầu tiên của unseen parents
đến parent floor. Cấu hình không có field mới tiếp tục dùng hành vi cũ; frozen S2
không bị thay đổi.

Replay parent floor 400 trên 30 traces:

- Thu hồi 2/8 manual parents: query 610 và 1159.
- Hai recovered passages đều chứa đúng registered manual evidence.
- Thu hồi 4/12 historical parents.
- Mất 0/10 control parents.
- Prefix mixed-item cũ giữ nguyên trên 30/30 queries.

Kết quả này vượt retrieval pool gate, nhưng chưa chứng minh document/chunk output
tốt hơn. Candidate thêm tối thiểu 3.844 parent/passage pairs trên cohort. Các
parents mới không có same-input reranker score frozen, nên chưa thể xếp hạng với
S2 hoặc tính downstream gross gains/losses một cách hợp lệ.

Máy hiện tại không có CUDA; `nvidia-smi` không kết nối được driver và
`torch.cuda.is_available()` là false. Baseline score contract dùng pinned CUDA
FP16 runtime. Chạy CPU hoặc dtype khác sẽ phá score comparability, vì vậy execution
dừng fail-closed tại score gate. Không mở full inference.

## Artifacts và kiểm tra

Namespace: `outputs/v4/r7_targeted_parent_loss_20261025/`.

- `cohort_lock.json`, `label_source_check.csv`: cohort và source identity.
- `canary_report.json`: exact parity PASS.
- `traces/*.json`, `trace_report.json`: raw/fusion traces đủ 30 queries.
- `parent_loss_attribution.csv`: stage đầu tiên mất parent.
- `candidate_gate.json`, `candidate_parent_replay.csv`: replay patch.
- `execution_manifest.json`: code/artifact hashes và score-gate status.

Tests liên quan pipeline/fusion/evaluator: **42 PASS**. `git diff --check` và
Python compilation PASS. Test mới xác nhận prefix bất biến, supplement unique
parents và default path giữ hành vi cũ.

S2 ZIP SHA-256 vẫn là
`081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.

## Bước được phép tiếp theo

Chỉ chạy score canary cho 30-query cohort bằng đúng pinned CUDA FP16 runtime,
với trần các pairs mới đã materialize từ parent-floor pool. Sau khi merge scores,
đo parent entry vào ranked top-200/S2 document policy và gross gains/losses trên
manual/control cases. Nếu không thu hồi ít nhất một manual parent vào output hoặc
làm mất control, dừng patch. Chỉ khi score gate PASS mới cân nhắc validation và
chi phí full inference; chưa chuẩn bị BTC submission ở trạng thái hiện tại.

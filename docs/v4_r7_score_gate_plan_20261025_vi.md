# Kế hoạch tiếp: kiểm chứng parent-floor qua reranking và output

Trạng thái: **EXECUTED THROUGH DOWNSTREAM CANARY — NO_GO**. Exact-batch runtime
parity PASS và 385 new pairs đã được score trên GPU 2. Cả hai manual targets vào
final documents nhưng không vào chunks; control 53 mất một known historical
positive document. Gate khóa không cho mở phần còn lại của cohort. Không có
submission mới; S2 vẫn FINAL 0.2318.

Kết quả thực thi chi tiết: [v4_r7_score_gate_execution_20261025_vi.md](v4_r7_score_gate_execution_20261025_vi.md).

## Mục tiêu và dữ liệu đã có

Câu hỏi cần trả lời: parent đã thu hồi ở pool có vượt reranking, document gate
và 60 chunk slots để tạo output có ích hay không?

R7 đã có đầy đủ raw traces của 30 development queries, exact-parity canary PASS
và patch opt-in. Parent-floor thu hồi 2 manual parents (610/305379,
1159/514656), 4 historical parents và giữ 10/10 controls ở **pool**. Chưa có
bằng chứng gain ở final documents/chunks hoặc official F2.

Con số 3.844 hiện là tổng added parents trên cohort, được báo là **minimum new
pairs**, chưa phải một scoring manifest hoàn chỉnh. Nhất là document-only hits
cần passage fallback; không được coi parent count là số score inputs đã xác minh.

## 1. Chuẩn bị package scoring trên CPU

Tái dùng 30 traces đã khóa; không truy vấn lại dense index. Khóa seed policy
trước khi đọc labels: mỗi added parent dùng đúng item đầu tiên được parent-floor
bổ sung. Nếu item là document-only, dùng frozen passage BM25 fallback top-1 của
chính parent. Không chọn seed bằng vị trí manual evidence.

Mỗi pair có query, doc ID, passage ID, exact `title + newline + passage` payload,
query/input hashes và source provenance. Kiểm source substring, offsets, parent
alignment, duplicate IDs và finite scores khi import. Fallback không có passage
phải ghi unscorable; không tự thêm source text hoặc gán score.

Reuse các old scores từ refined S2 candidates theo exact input identity. Khóa
riêng scoring manifest và code revision mới vì source pipeline đã thay đổi sau
patch; không bỏ qua các hash guards trong frozen GPU runner cũ.

Quan trọng: không chạy lại `prepare_rerank` với cap 200 rồi cho rằng mọi parent
mới được chấm. Chúng có thể bị cắt trước scoring. Worker phải score riêng các
new pairs đã khóa rồi merge với **toàn bộ cached S2 candidate scores** trước khi
sort parent theo max logit và chọn ranked top-200.

Đầu ra: `pair_manifest.json`, `pairs.jsonl`, package nhỏ chỉ chứa input cần
score, reference anchor pairs và runtime contract. Package chạy được trên máy
GPU mà không cần mang theo full corpus/dense index.

## 2. Runtime gate và canary 3 queries

Ưu tiên runtime S2 đã ghi: BGE reranker gốc revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, max_length 512, CUDA FP16,
raw logits, batch size 32 và package/model hashes theo runtime manifest A6000.
Kiểm CUDA trong chính môi trường worker; UUID thiết bị dùng để chọn tài nguyên,
không phải thước đo chất lượng.

Chấm trước tối đa **128 old anchor pairs** lấy deterministic, có các case gần
ngưỡng -4/-2 và biên chunk/document selection. Khóa trước tolerance tuyệt đối
1e-3 cho logits và yêu cầu không đổi quyết định ở các biên được kiểm. Đây là
compatibility canary có giới hạn, không chứng minh mọi score trên mọi máy bằng nhau.
Khác model/input hash, non-finite score, sai count hoặc không đạt parity thì dừng;
không trộn score mới vào cached logits S2. Một runtime khác chỉ được dùng sau
khi có phương án so sánh hai nhánh trong cùng runtime và chi phí được khóa lại.

Sau runtime gate, score toàn bộ added parents của đúng 3 queries:

| Query | Vai trò | Added parents trước materialization |
|---|---|---:|
| 610 | Manual parent 305379 được thu hồi | 121 |
| 1159 | Manual parent 514656 được thu hồi | 112 |
| 53 | Retained control | 152 |
| Tổng |  | 385 |

385 là số dự kiến nếu một valid fallback/seed tương ứng một parent. Manifest
thực tế phải được khóa trước scoring; nếu vượt số này, giải thích nguyên nhân
và dừng trước khi tăng budget. Không chỉ score hai target passages vì sẽ bỏ qua
cạnh tranh giữa các added parents và tạo output thiếu công bằng.

Replay đầy đủ selection với cùng S2 policy: parent max logit, ranked top-200,
60 chunks, adjacent context 4096 rồi expansion 5500, pre-gate -4, confidence
threshold -2, adaptive document target [80,140], bảo vệ chunk parents. Nếu một
old chunk parent được giữ, giữ nguyên exact expanded context S2 của nó.

**Gate canary:** ít nhất một target manual parent phải vào final relevant_docs;
control 53 không mất known relevant parents/evidence; structural validators PASS.
Ghi riêng target có vào 60 chunk slots và context cuối có giữ evidence hay không.
Gain chỉ ở docs phải được báo là docs-only, không gọi là cải thiện CHUNKS_F2.
Nếu cả hai targets không vào final docs, dừng nhánh trước khi score toàn cohort.

## 3. Chỉ mở toàn cohort khi canary có ích

Score phần còn thiếu của 30-query manifest, reuse 385 canary pairs. Dự kiến
3.844 candidate pairs tổng cộng nếu seed/fallback audit xác nhận. Với 128 runtime
anchors, budget dự kiến 3.972 model evaluations, không score lại pairs đã có.
Đo thời gian từ canary và khóa dự toán trước khi chạy phần còn lại.

Xuất bảng trước/sau cho từng manual/historical parent tại pool, scored candidates,
ranked top-200, final docs và final chunks. Đếm gross gains, gross losses và
queries có loss; net bằng 0 không che được một known-positive bị mất.
Đối chiếu mọi known-positive thuộc 30 queries trong registry, không chỉ parent
đại diện đã dùng để chọn mỗi query vào cohort.

**Gate cohort:** recover ít nhất một manual parent vào final docs; không mất
known positive docs/chunks trên controls; không có regression structural.
Các labels này không đầy đủ, nên vượt gate chỉ cho phép bước kiểm tiếp, chưa đủ
để tuyên bố precision, recall hay official F2 tăng. Chỉ chấm trực tiếp contexts
thay đổi khi cần đánh giá relevance mới, trần 60 contexts cho pilot.

## 4. Quyết định kết thúc

- Canary hoặc cohort fail: giữ S2, lưu nguyên nhân và dừng parent-floor 400.
- Cohort PASS: khóa policy/code trước khi dùng validation/held-out của registry;
  công bố rõ đây là historical diagnostic, không pristine benchmark.
- Chỉ sau xác nhận trên queries khác và structural gates mới dự toán full run
  và chuẩn bị một ZIP để BTC kiểm chứng. Tiêu chí promotion hiện tại: FINAL
  >0.2318, DOCS_F2 không giảm dưới 0.2788, CHUNKS_F2 không giảm dưới 0.1848.

Nếu chưa có GPU phù hợp, hoàn thành package và dừng tại READY_FOR_GPU. Không
thuê tài nguyên hoặc chuyển dữ liệu sang dịch vụ mới trong phạm vi kế hoạch này.
Kết quả mong đợi của lượt tiếp theo là quyết định canary có/không tạo gain ở
**output**, không phải thêm một báo cáo pool coverage.

Namespace dự kiến: `outputs/v4/r7_parent_floor_score_20261025/`.

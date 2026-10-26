# Kế hoạch bước tiếp theo: tìm và sửa một nguyên nhân mất relevant documents

Trạng thái: **EXECUTED THROUGH POOL GATE**. Kết quả tại
[R7 targeted execution](v4_r7_targeted_execution_20261025_vi.md). Patch thu hồi
2 manual và 4 historical parents ở candidate pool, mất 0 controls; score gate
dừng vì máy hiện tại không có pinned CUDA runtime.
Mốc so sánh: S2, BTC FINAL 0.2318, DOCS_F2 0.2788, CHUNKS_F2 0.1848.
Đầu ra cần đạt là một nguyên nhân có thể tái hiện và một bản sửa có đo trước/sau.
Nếu chưa tìm được nguyên nhân đủ bằng chứng, dừng ở gate chẩn đoán.

## Cơ sở chọn bước này

F01/F03/F06 đã sửa. Audit 1.200 queries cho thấy sắp lại cùng parent reranker
logit không tạo swap có margin dương; điều đó chưa kiểm nghiệm một evidence
signal mới. Pilot annotation 13.546 contexts đã dừng trước khi fit hoặc chấm.

Trong registry R6 cũ, 87 manual labels thuộc 87 official queries. Split chung
có 63 development, 14 validation và 10 held-out trong nhóm này. Có 11 manual
parents thiếu ở fused pool; 8 thuộc development. Cả 11 vẫn thiếu trong S2 docs.
Historical positives có 113 development queries với parent thiếu ở fused pool.

Các con số này chỉ dùng chọn case tìm lỗi. Manual relevance cần kiểm lại
query/source/evidence trước khi dùng. Historical labels không phải complete
gold; thiếu raw branch trace phải ghi UNKNOWN, không được gọi retrieval miss.

## 1. Khóa mẫu chẩn đoán và kiểm đầu vào

Giới hạn **30 development queries**:

- 8 query manual có parent thiếu ở fused pool, sau kiểm lại label/source.
- 12 query historical có parent thiếu; chọn deterministic và không trùng nhóm trên.
- 10 control queries có known relevant parent còn trong S2; ưu tiên cases đã có
  raw trace từ pilot cũ để kiểm parity.

Khóa IDs trước khi chạy, lấy assignments từ `query_split_registry.json` hiện có.
Case không vượt kiểm identity/source phải ghi invalid và thay bằng quy tắc đã
khóa; không thay bằng case có kết quả thuận lợi. Reuse release coverage audit
hiện có; chỉ truy nguyên những parent trong mẫu, không scan lại 922 release parts.

Đầu ra: `cohort_lock.json`, `label_source_check.csv`. Mỗi known parent có query,
doc ID, label provenance, source hash, evidence span hợp lệ và trạng thái index.
Budget dự kiến: 45–60 phút công việc; chưa chấm thêm hàng nghìn contexts.

## 2. R7 targeted: xác định stage mất parent

Tái sử dụng raw trace/cached hits đúng lineage trước. Khi cần truy vấn lại, chạy
đúng competition path: document BM25 400 + passage BM25 200 + dense 200,
RRF k=60, top-400 mixed items rồi group parents. Script probe B0 hiện tại chỉ
fusion passage BM25+dense nên không dùng trực tiếp làm replay S2.

Lưu đầy đủ raw branch hits, passage/parent identities, ranks, scores và kết quả
pre/post-fusion. Liên kết với frozen scored candidates, ranked parents và S2 docs.
Phân loại mỗi known parent vào stage đầu tiên có bằng chứng mất:

1. Source chưa vào eligible index: truy nguyên crawl/extraction/eligibility.
2. Có source nhưng không vào raw union: retrieval coverage/depth.
3. Có trong raw union nhưng bị loại sau fusion: fusion/truncation.
4. Qua fusion nhưng không vào ranked parents: pair coverage/reranking/selection.
5. Có trong ranked parents nhưng bị loại khỏi S2 docs: document gate.
6. Trace hoặc parity thiếu: UNKNOWN.

Chạy **3 queries canary** trước để đo thời gian và kiểm ranks/IDs với explicit
trace sẵn có. Khác runtime/hardware mà chưa giải thích được parity thì dừng việc
gán nguyên nhân cho frozen S2. Với replay đầy đủ, cả ba branches phải thực sự
được đo; không suy raw absence từ `candidate.ranks` đã qua truncation.

Giới hạn tối đa 30 queries, tổng budget retrieval mới **2 CPU-giờ**. Chỉ encode
những query thiếu cache; chưa chạy GPU reranker hoặc rebuild index. Nếu vượt
budget, giữ kết quả canary và báo stage UNKNOWN thay vì tự mở rộng compute.

Đầu ra: `raw_branch_trace.jsonl`, `parent_loss_attribution.csv` và một bảng đếm
theo stage, tách manual/historical/control. Kết luận phải dẫn tới query/doc cụ thể.

## 3. Chỉ sửa một nguyên nhân đã xác nhận

Gate mở bước sửa: có case tái hiện, xác định được stage lỗi và có sửa đổi khả thi
trong phạm vi nhỏ. Chọn theo số manual cases hợp lệ, sau đó mức ảnh hưởng trên
historical cases và chi phí. Không chọn chỉ vì logit tăng hoặc oracle cao.

Ví dụ quyết định: sửa identity/extraction nếu relevant source bị loại sai;
sửa fusion cutoff nếu raw hits có parent nhưng truncation làm mất; sửa document
gate nếu parent đã nằm trong ranked list rồi mới bị loại. Trường hợp chỉ thiếu
traces hoặc chỉ có low-score historical anchors chưa đủ để tạo challenger.

Đầu ra: một patch, regression test bắt đúng lỗi đã tái hiện và replay trước/sau
trên cohort. Dùng cached scores có đúng input hash. Nếu bản sửa cần score mới,
ghi rõ số pairs và chi phí tại gate; chỉ một canary nhỏ sau khi retrieval/pool
gate đã chứng minh recover relevant parents. Không chạy full 1.200-query inference.

Gate tiếp: recover ít nhất một manual relevant parent đã xác minh, không mất
known relevant parents trên control, structural validators PASS và ghi đủ gross
gains/losses. Đây là điều kiện tiếp tục kiểm chứng, chưa phải bằng chứng F2 tăng.
Nếu cần chấm context mới, chỉ chấm phần thay đổi, trần **60 contexts** cho pilot;
không khôi phục bộ annotation 13.546 contexts.

## 4. Kiểm chứng rồi mới chuẩn bị BTC

Khóa patch/policy trước khi mở tập kiểm tra khác query. Dùng validation có sẵn
để phát hiện regressions, rồi held-out theo registry; ghi rõ các labels lịch sử
đã được xem, không gọi đây là pristine benchmark hoặc independent F2 estimate.
Theo dõi đồng thời DOCS và CHUNKS, vì thay retrieval có thể tác động cả hai.

Chỉ khi qua gate mới chuẩn bị **một candidate ZIP**: đủ 1.200 queries, source
grounding, parent alignment, cardinality, payload/ZIP validators và input/code
hashes. Full inference chỉ được mở khi pilot có kết quả đủ rõ và chi phí đã đo.
Điểm thực tế được xác nhận bằng BTC; tiêu chí promotion khóa trước khi nộp.

Mặc định mục tiêu promotion là FINAL > 0.2318 và cả DOCS_F2/CHUNKS_F2 không
giảm so với S2. Nếu BTC không cải thiện, giữ S2 và dừng patch đó; không sinh một
grid biến thể để tiếp tục thử trên cùng feedback.

## Điều kiện hoàn thành lượt kế tiếp

Trong một lượt chẩn đoán có giới hạn, phải trả lời được: parent nào mất ở stage
nào, bằng chứng là gì, sửa cụ thể ở đâu và kết quả trước/sau ra sao. Nếu chưa có
patch đủ điều kiện, đầu ra là quyết định dừng với dữ liệu còn thiếu được nêu rõ.
Không hứa tăng lên 0.4 từ bộ diagnostic này; không dùng số tests PASS làm bằng
chứng retrieval relevance hoặc official score tăng.

Namespace: `outputs/v4/r7_targeted_parent_loss_20261025/`.

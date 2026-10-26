# Kế hoạch cải thiện output: điểm evidence riêng cho 60 chunk slots

Trạng thái: **EXECUTED — STOP_DEVELOPMENT**. Development đạt 2 manual gains,
0 manual losses, 5 historical gains và 0 historical losses; thiếu ngưỡng khóa
trước là 3 manual gains. Validation/held-out không được chạy. Chi tiết:
[execution report](v4_evidence_window_allocation_execution_20261025_vi.md).
Mốc so sánh: S2, BTC FINAL 0.2318, DOCS_F2 0.2788, CHUNKS_F2 0.1848.
Namespace khi triển khai: `outputs/v4/evidence_window_allocation_20261025/`.

## 1. Vấn đề cần giải quyết và bằng chứng

S2 xếp parents theo max seed logit và cấp chunks cho 60 parents đầu. Điểm
document relevance đang đồng thời quyết định evidence allocation. Chưa có
điểm evidence riêng được chứng minh tốt hơn.

R7 canary đã cho một cải thiện output có thật: 3 known-positive query/parent
gains, 1 historical-positive query/parent loss. Có 22 chunk replacements nhưng
known evidence gains/losses là 0/0; relevance của các replacements còn chưa rõ.
Con số 5 gains trong report gốc đếm registry rows trùng parent, không phải 5 docs.

Hai target manual được thu hồi vào docs ở ranks 104 và 68. Kiểm tokenizer mới
cho thấy input lần lượt 401 và 432 tokens, manual evidence đều nằm trọn trong
input. Với hai ca này, không có bằng chứng truncation gây mất chunk slots.
Do đó tăng max_length riêng lẻ chưa phải biện pháp có căn cứ.

Giả thuyết thử nghiệm: chấm các đoạn ngắn trong context có thể giảm ảnh hưởng
của phần nội dung ngoài câu trả lời, tạo thứ tự evidence khác với document
ranking. Đây vẫn là BGE gốc trên representation mới, chưa phải model evidence
được huấn luyện hoặc tín hiệu đã xác thực. Nếu không cải thiện, dừng giả thuyết
này; không coi đổi representation tự động là cải thiện relevance.

## 2. Khóa một treatment trước inference

So sánh hai arms:

- A: nguyên S2 documents, 60 chunk parents và exact chunk texts.
- B: nguyên S2 documents; chỉ thay tối đa 6/60 chunk parents mỗi query bằng
  evidence score mới. Không thay context text của parent đã được khóa.

Candidate parents chỉ thuộc **S2 exported documents**, tối đa 140/query. Không
trộn parent-floor vào treatment này. q610/305379 và q1159/514656 không thuộc S2
docs nên không phải targets có thể thu hồi trong arm B; chúng chỉ giải thích
vấn đề của reranker. Nhánh này nhắm parents đã có trong S2 docs nhưng thiếu slots.

Mỗi parent có đúng một export context cố định: parent đang có slot giữ nguyên
exact S2 text; parent chưa có slot lấy context có seed logit cao nhất từ frozen
R5 context pool, tie-break bằng context ID. Kiểm hash, source substring, offsets
và parent identity. Không tự tạo full-source context khi lookup thất bại.

Scoring representation duy nhất:

1. Tokenize toàn export context bằng pinned tokenizer.
2. Chọn ba cửa sổ contiguous dài tối đa 192 body tokens, neo ở đầu, giữa và
   cuối context; lấy exact source slices bằng offsets. Context ngắn dùng toàn
   context. Deduplicate windows giống hệt nhau.
3. Payload `query` và `title + newline + window`, tối đa 512 pair tokens.
   Nếu vượt vì query/title, giảm body quanh cùng anchor bằng thuật toán cố định;
   không sửa query hoặc chọn anchor theo nhãn. Không còn body hợp lệ thì ghi lỗi.
4. Evidence score = max raw logit của các windows. Tie-break: S2 parent rank,
   rồi official ID. Không fit weights, không sweep window sizes/margins.
5. Chấm cả parents đang giữ slots và parents ngoài 60 bằng cùng policy/runtime.
   Không so logit cửa sổ mới trực tiếp với cached seed logit S2 để quyết định swap.

Ba windows không đảm bảo phủ toàn context dài; ghi token coverage để biết giới
hạn. Max aggregation có thể thiên về contexts có nhiều windows hợp lệ; báo phân
bố số windows và kiểm lỗi này khi đọc kết quả, không tự thêm normalization sau
khi xem validation.

Allocation: tạo top-60 theo evidence score; ghép entrants và victims theo thứ tự
score, chỉ swap khi entrant cao hơn victim và lấy tối đa 6 swaps/query. Giữ đủ
60 chunks nếu baseline có 60. Export context vẫn là context cố định ban đầu;
scoring window không trở thành exported chunk.

## 3. Cohort và giới hạn công việc

Khóa 30 development, 20 validation, 10 held-out queries theo registry
`outputs/v4/evidence_evaluation_lock_20261025/query_split_registry.json`.
Kiểm trước khi lập plan cho thấy random sample 30/20/10 chỉ có 2/0/0 manual
rows; development chỉ có một manual parent ngoài slots, không thể vượt gate
3 gains. Không dùng thiết kế chắc chắn thiếu support này.

Thiết kế lấy mẫu có chủ đích, công bố rõ trước inference:

- Development: cả 11 queries có manual-positive parent trong S2 docs nhưng
  ngoài 60 chunk slots: 5, 65, 130, 353, 398, 566, 579, 594, 747, 910, 1150;
  thêm 10 queries manual evidence đang visible làm controls, không trùng 11 ca
  trên; thêm 9 queries từ development còn lại.
- Validation: giữ cả 14 queries có manual labels trong split này, thêm 6 queries
  còn lại. Hiện chỉ query 71 có manual parent trong docs ngoài slots.
- Held-out: cả 10 queries có manual labels trong split này; hiện không có manual
  parent ngoài slots và có 9 queries manual evidence đang visible.
- Khi cần chọn controls/fill, sort SHA256 của
  `evidence-window-allocation-20261025-v1:{query_id}`. Không lọc bằng kết quả
  window scoring hoặc thay query vì baseline/challenger có điểm xấu.

Đây là **enriched diagnostic cohort**, không đại diện tỷ lệ lỗi toàn bộ queries.
Labels chỉ dùng xác định cohort/đánh giá; không chọn windows, context hoặc swaps.
Khóa query IDs, baseline support và controls trước scoring. Query chỉ có 60 docs
vẫn được giữ nếu thuộc nhóm manual; khi không có outsider, allocator không swap.

Ghi membership, exclusions và input/code/model hashes trước inference. Không
thay query khác nếu kết quả xấu. Historical labels đã từng được xem nên split
chỉ là boundary thực nghiệm, không tuyên bố pristine benchmark.

Budget tối đa:

| Stage | Queries | New scoring pairs tối đa |
|---|---:|---:|
| Development | 30 | 12.600 |
| Validation, chỉ sau development gate | 20 | 8.400 |
| Held-out, chỉ sau validation gate | 10 | 4.200 |
| Tổng candidate windows | 60 | 25.200 |

Runtime compatibility: 128 anchors trong bốn **intact historical batches**,
tolerance 1e-3, threshold decisions giữ nguyên. Giữ batch/order khi resume.
Không lặp lỗi anchors rời batch trong R7. Scores arm B được so với arm A bằng
output, không dùng parity để khẳng định evidence relevance.

GPU 2 hoặc 3: kiểm UUID/process/memory trước lúc chạy, expose một A6000 cho
worker, dùng pinned S2 environment. Chốt manifest pair count trước model load.
Giới hạn 15 GPU minutes/stage và 2 CPU hours cho toàn pilot, báo wall/model time
riêng. Chạm budget thì checkpoint và báo phần thiếu, không tự mở full 1.200.

## 4. Gate nghiên cứu và cách đo

Tách lỗi kỹ thuật với trade-off chất lượng:

- Sai hash/source/identity, non-finite score, split overlap hoặc vi phạm caps:
  STOP ngay.
- DOCS lists arm B phải giống A từng query, đúng thứ tự. Đây là kiểm tra bằng
  output, không chỉ dựa trên cờ cấu hình.
- Một historical loss riêng lẻ được ghi trong gross losses; không còn là veto
  tự động của nghiên cứu. Không ghi đè quyết định NO_GO của R7 cũ.

Báo riêng theo từng query và gộp:

- Unique query/parent document gains/losses; arm B phải bằng 0/0.
- Manual evidence visibility gains/losses trong exact exported contexts.
- Historical anchor preservation gains/losses, dedup registry identity khi
  đếm parents; không gọi số này là official recall hoặc relevance judgment.
- Số changed slots, queries đổi, gross-loss queries dù query đồng thời có gain.
- Relevance trực tiếp của paired gained/lost contexts nếu có reviewer độc lập.

Development screening gate, khóa trước khi đọc kết quả:

- Có ít nhất 3 query/parent manual evidence gains trên ít nhất 3 queries.
- Manual evidence losses tối đa 1 query, và gross gains > gross losses.
- Historical preservation net không âm, gross-loss queries tối đa 3/30.
- Structural/identity/document-equality validators PASS.

Đây là ngưỡng sàng lọc thực dụng, không phải suy ra F2 từ incomplete labels.
Nếu sample không có đủ manual support để kết luận thì ghi INCONCLUSIVE, không
đổi sang dùng historical proxy để tự động cấp GO và không thay cohort sau đó.

Validation giữ nguyên treatment: manual evidence net không âm, không mất manual
evidence đang visible; historical preservation net không âm và tối đa 2/20
gross-loss queries. Held-out dùng cùng điều kiện manual, historical preservation
net không âm và tối đa 1/10 gross-loss query. Không yêu cầu manual gains ở split
không có manual outsider để thu hồi. Hai stage này kiểm regression; muốn kết luận
relevance tăng vẫn cần direct judgments trên contexts mới hoặc BTC.
Không cứu development fail bằng validation, không chọn lại policy trên held-out.

Nếu có direct context review: chuẩn bị tối đa 36 gained/lost pairs (72 contexts)
cho cả pilot, chia tối đa 18/12/6 pairs theo stage, lấy deterministic trước
judgments; review opaque IDs với query và full context, giấu arm/score/rank.
Chấm đúng context, không gán nhãn từ parent. Samples này chỉ bổ sung bằng chứng
và không được coi là exhaustive relevance accounting của toàn bộ swaps.
Reviewer chưa sẵn có thì vẫn hoàn thành GPU screening và ghi kết luận tương
ứng; không quay lại yêu cầu annotation 13.546 contexts. Self-review/model review
không được ghi thành independent gold nếu không có provenance phù hợp.

## 5. Phép kiểm phụ DOCS bằng cached scores

Dùng nguyên ba replay queries R7 đã có scores, so cap 140 với 160; giữ chunk
outputs cố định của từng arm và pre-gate -4. Không score thêm 3.459 R7 pairs
trong kế hoạch này, không tạo treatment phối hợp với evidence allocator.

Ghi số docs tăng, unique known-positive gains/losses, và bao nhiêu docs thêm
chưa được judged. Cap 160 có thể tránh control loss ở q53 nhưng không đủ bằng
chứng F2 tăng; chỉ khi có tín hiệu trên cohort khác mới mở một DOCS experiment
riêng. Không chọn cap 145 để vừa khớp ca lỗi đã quan sát.

## 6. Đầu ra và quyết định tiếp theo

Mỗi stage xuất baseline/challenger predictions, pair/score manifests, danh sách
swaps và report tự tính validators/gains/losses. Không chỉ xuất báo cáo pool.

- Screening fail: dừng window scoring/max aggregation này; giữ bằng chứng vì
  sao không đủ phân biệt evidence. Hướng tiếp là thay scorer/task supervision
  với budget mới; không thêm lexical variants hoặc tăng retrieval một cách mặc định.
- INCONCLUSIVE do nhãn: báo rõ output đã thay bao nhiêu, dữ kiện còn thiếu;
  không đánh dấu INDEPENDENT_RELEVANCE_GO. Có thể chuyển một candidate đã qua
  các structural/regression screens sang BTC để đo nếu chọn đường official
  evaluation, thay vì tạo một vòng annotation lớn. Việc này cần dự toán full
  inference và slot submission rõ ràng; không tự tuyên bố gain offline.
- Qua development/validation/held-out: ghi SCREENED_CANDIDATE, khóa một policy,
  dự toán full inference
  theo actual throughput/pair count và output size trước khi chạy 1.200 queries.
  Với policy ba windows, upper bound full scoring là 504.000 pairs; chi phí này
  chỉ được mở sau pilot, không xem là đã được cấp trong lượt lập plan.
- Mục tiêu handoff sau khi vượt các gates là **một candidate ZIP** để BTC kiểm
  chứng, không một grid submissions. Promotion dựa official FINAL >0.2318,
  DOCS_F2 >=0.2788 và CHUNKS_F2 >0.1848.

Plan này bổ sung đường nghiên cứu mới; S2 là mốc so sánh frozen, không phải
ràng buộc cấm challenger có bất kỳ loss nào trong development.

# Roadmap hướng tới FINAL > 0,4

Trạng thái: **STRATEGIC PLAN — chưa mở experiment full mới**.

Champion hiện tại: cap-10 evidence-window, BTC do người dùng cung cấp:
FINAL `0,2324`, DOCS_F2 `0,2788`, CHUNKS_F2 `0,1860`.

## 1. Khoảng cách thật tới mục tiêu

BTC FINAL là trung bình hai component. Muốn FINAL > 0,4 cần:

```text
DOCS_F2 + CHUNKS_F2 > 0,8
```

Hiện tổng chỉ là `0,4648`; để vượt 0,4 cần tăng tổng hơn `0,3352`. Một cấu hình
mục tiêu hợp lý về mặt số học là DOCS_F2 `0,46` và CHUNKS_F2 `0,35`, cho FINAL
`0,405`. Điều này đòi hỏi thay đổi
cấp kiến trúc; margin/cap/window tuning không thể tạo mức nhảy đó.

Cap-10 đã chứng minh allocation riêng tốt hơn S2, nhưng thay 11.445 slots chỉ
tăng CHUNKS_F2 `0,0012`. Vì vậy dừng sweep cap 11/12/15 và không dùng thêm BTC
để tune cùng cached scores.

## 2. Kiến trúc đích

Tách hệ thống thành hai objective và hai scorer khác nhau:

```text
query understanding / expansion
  → high-recall parent retrieval (đến 2.000 candidates)
  → document reranker + calibrated dynamic document selection
  → exhaustive evidence-window generation trong selected documents
  → evidence reranker + grounded span/context selection
  → 60 chunk allocation
```

### Document path

- Chạy raw và normalized Vietnamese query; bổ sung medical entity, spelling và
  synonym variants bằng rule/lexicon đã khóa, không sinh query theo kết quả BTC.
- Union document BM25, passage BM25 và dense ở depth cao hơn; group/fuse theo
  parent trước khi cắt, loại lỗi mixed-item top-400 đã thấy ở R7.
- Cascade: cheap retriever xếp khoảng 2.000 parents, semantic pre-ranker rút còn
  300–400, document cross-encoder/listwise reranker chọn top-200.
- Chọn số documents động bằng calibrated relevance, không chỉ clip count [80,140].

### Evidence path

- Không dùng ba head/middle/tail windows như representation cuối. Sinh sliding
  windows 192 tokens, stride 96 trên toàn source/context để không bỏ vùng giữa.
- Cheap evidence encoder lọc windows, sau đó evidence cross-encoder chấm top
  windows. Document relevance và evidence relevance không dùng chung logit.
- Học trực tiếp query–exact-span relevance; export source-grounded compact
  context quanh best span. Baseline context luôn là fallback để kiểm gross loss.
- Allocation dùng calibrated evidence probability và loss budget, không fixed
  10 swaps bất kể margin.

## 3. Dữ liệu huấn luyện và evaluation bắt buộc

Không dùng lại lỗi R3/R5: negative chunk không được tự động biến thành negative
document; nhãn của old text không chuyển sang context mới.

- 31.294 feedback rows cũ chỉ được dùng theo exact `(query, doc, text hash)`;
  2.190 positives có thể làm evidence training positives, không làm complete gold.
- Manual labels và historical anchors chỉ dùng đúng provenance; historical là
  preservation proxy.
- Tạo `Gold-300` query-disjoint: 180 development, 60 validation, 60 held-out.
  Với mỗi query, judge trực tiếp query–document và query–exact-context trên một
  candidate set nhỏ, gồm champion results và candidates mới. Mục tiêu khoảng
  2.400 document judgments, không quay lại annotate 13.546 contexts.
- Nếu dùng model/LLM teacher thay người, nhãn phải ghi `weak_teacher`; audit người
  tối thiểu 10% và không gọi đó là independent gold.

Không có direct judged set thì không có cách đáng tin để chọn một bước nhảy lớn
từ aggregate BTC score; khi đó mọi full run chỉ là blind search.

## 4. Các stage và fail-closed gate

### Stage A — R11 high-recall coverage, bước chạy kế tiếp

Khóa toàn bộ 87 manual queries và controls trước retrieval. So depths 400, 800,
2.000 và 4.000 cho từng branch; thêm parent-level fusion và normalized-query arm.
Chỉ đo source/index presence và verified parent recall, chưa rerank/full ZIP.

GO khi arm thắng:

- Thu hồi ít nhất 10 verified manual parents ngoài champion pool hoặc tăng
  verified parent recall ít nhất 10 điểm phần trăm.
- Giữ 100% retained controls trong top-2.000.
- Ít nhất 80% gains nằm trong top-400 sau cheap parent fusion; nếu gains chỉ nằm
  ở rank hàng nghìn thì candidate generator chưa đủ dùng.

Nếu depth 4.000 không tăng coverage đáng kể, dừng tăng depth và chuyển sang
query reformulation/corpus eligibility. Không đưa 4.000 pairs/query vào reranker.

Budget: tối đa 4 CPU-hours trên cohort và chỉ encode queries thiếu cache.

### Stage B — document ranker

Huấn luyện query-grouped listwise/pairwise ranker trên exact document judgments,
với hard negatives từ arm thắng Stage A. Split theo query registry, không để R3
train queries lọt validation/held-out.

Gate trước full:

- Direct judged Recall@140 tăng ít nhất 15 điểm phần trăm so champion.
- Direct judged precision không giảm; query gross-loss rate <=5%.
- Validation và held-out cùng chiều; không chọn checkpoint bằng held-out.

Mục tiêu BTC trung gian cho document branch: DOCS_F2 >=0,35. Chưa đạt mức này
thì không kỳ vọng FINAL 0,4.

### Stage C — evidence model và full-span selection

Huấn luyện scorer riêng trên exact positive spans và judged hard negatives.
Chạy sliding-window coverage audit trước model inference; candidate union phải
giữ mọi champion manual evidence và bổ sung được evidence mới.

Gate:

- Direct judged evidence F2 tăng ít nhất 20% tương đối trên validation và held-out.
- Manual evidence gross gains > gross losses, loss <=2% queries.
- Historical preservation chỉ là guardrail, không phải promotion metric.

Mục tiêu BTC trung gian: CHUNKS_F2 >=0,27; sau đó mới mở vòng nâng lên khoảng
0,35 bằng better span/context selection.

### Stage D — một combined candidate

Chỉ kết hợp document path và evidence path sau khi mỗi path qua gate riêng.
Replay full structural/source/ZIP validators, tạo đúng một BTC submission.

Promotion milestones:

| Milestone | DOCS_F2 | CHUNKS_F2 | FINAL kỳ vọng |
|---|---:|---:|---:|
| M1 | 0,35 | 0,25 | 0,30 |
| M2 | 0,42 | 0,28 | 0,35 |
| M3 target | 0,46 | 0,35 | 0,405 |

Các mốc là target engineering, không phải oracle hoặc cam kết score.

## 5. Những việc dừng ngay

- Không sweep cap evidence-window bằng BTC feedback mới.
- Không chỉ tăng retrieval depth rồi cross-encode 4.000 parents/query.
- Không fine-tune lại R3 trên inherited labels/context mismatch.
- Không chạy full 1.200 queries trước khi coverage và direct-label gates PASS.
- Không coi historical net gains là dự báo định lượng official F2.

Quyết định thực thi kế tiếp: **Stage A R11 high-recall coverage**. Đây là bước
nhỏ nhất có thể xác định liệu bottleneck document nằm ở depth/fusion hay ở query,
corpus/model; kết quả của nó quyết định có đáng xây document reranker mới hay không.

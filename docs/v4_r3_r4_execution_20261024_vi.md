# Báo cáo thực thi vòng 3 và vòng 4 — 24/10/2026

## Kết luận

- S2 vẫn là mốc đã được BTC chấm cao nhất: FINAL `0,2318`.
- R3 đã huấn luyện được chunk reranker tốt hơn baseline trên validation, nhưng
  các policy áp trực tiếp làm mất known-positive manual; vì vậy không xuất ZIP R3.
- R4 bổ sung sparse retrieval đã hoàn thành cho đủ 1.200 query. Hai policy rộng
  không qua validation. Policy high-confidence cuối cùng qua gate và đã xuất ZIP
  `r4_high_confidence_plus_s2` để chấm thử trên BTC.
- Mục tiêu `0,4` chưa thể được xác nhận bằng proxy local. Điểm R4 vẫn là
  `UNMEASURED` cho đến khi BTC trả kết quả.

## Mốc BTC do người dùng cung cấp

| Arm | FINAL | Docs F2 | Chunks F2 | Docs P/R | Chunks P/R |
|---|---:|---:|---:|---:|---:|
| S1 fixed-120 | 0,2291 | 0,2735 | 0,1848 | 0,1933 / 0,3714 | 0,2050 / 0,2108 |
| S2 adaptive | **0,2318** | **0,2788** | 0,1848 | 0,1946 / 0,3695 | 0,2050 / 0,2108 |

Hai receipt được lưu cạnh đúng ZIP và ghi rõ nguồn là feedback trong chat.

## Vòng 3 — supervised chunk reranker

Training data query-disjoint:

- 31.294 feedback rows: 2.190 positive và 29.104 negative.
- 5.668 train triples trên 335 train queries.
- Validation: 10.956 pairs trên 211 queries, gồm 773 positives.
- Model: `BAAI/bge-reranker-v2-m3`, revision
  `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, epoch tốt nhất là 2.

| Metric validation | Baseline | R3 epoch 2 |
|---|---:|---:|
| ROC-AUC | 0,525892 | **0,707866** |
| Average precision | 0,076349 | **0,169848** |
| Query macro AP | 0,167453 | **0,304622** |
| Query macro MRR | 0,246298 | **0,462028** |

Model gate PASS. Việc chấm toàn bộ 208.124 passage pairs của S2 đã hoàn tất cho
1.200 query. Tuy nhiên ba selector thử nghiệm đều không qua guardrail:

- Global blend alpha 0,50 làm validation manual parent/exact giảm `20/19` xuống
  `18/17`.
- Conservative alpha 0,25 làm giảm xuống `19/18`.
- Within-parent không tăng exact anchor trên development.

Do đó R3 được giữ làm scoring model/cached scores cho R4, không tạo submission
độc lập.

## Vòng 4 — sparse expansion và fusion

Retrieval dùng SQL aggregate BM25, giới hạn 12 term có IDF cao nhất:

- 1.200 query hoàn tất bằng 4 CPU workers.
- 180.000 passage mới, đúng 150/query.
- Tổng worker-time retrieval: 3.721,37 giây.
- R3 reranker chấm 180.000 pairs trên GPU trong 901,705 giây.
- Diagnostic trên 529 query đầu cho thấy pool mới tìm thêm parent của 67
  known-positive labels và exact anchor của 12 labels.

Hai policy đầu bị loại:

- Absolute replacement (cap 2/5/10/20): không policy nào qua development guards.
- Rank fusion: development tăng nhưng validation parent/exact giảm `554/449`
  xuống `553/448`, manual giảm `20/19` xuống `19/18`.

### R4 high-confidence đã qua gate

Policy được khóa chỉ thay tối đa một item/query, yêu cầu R3 score mới `>= 4,0`
và cao hơn item bị bỏ ít nhất `2,0`.

| Tập | Docs trước → sau | Parent trước → sau | Exact trước → sau | Manual docs/parent/exact |
|---|---:|---:|---:|---:|
| Development | 1.217 → 1.220 | 1.014 → 1.017 | 812 → 814 | 55 / 44 / 43, không đổi |
| Validation | 663 → 664 | 554 → 555 | 449 → 450 | 21 / 20 / 19, không đổi |

Đây là exact-containment proxy trên labels chưa đầy đủ, không phải official F2.

Submission R4:

- 1.200 query; 140.095 documents; 72.000 chunks.
- 609 replacements so với S2; tối đa một replacement/query.
- ZIP: 90.641.686 bytes = 90,64 MB decimal = 86,44 MiB.
- SHA-256:
  `20c4b2a17ff2be0c80641d00441d547820f6975e57b4f1d44ec84d6a30a56b77`.
- Local schema, ID, parent consistency, CRC và giới hạn 100.000.000 bytes: PASS.
- 43 targeted regression tests cho A0/R3/R4: PASS; toàn bộ script mới compile:
  PASS.

## Quyết định submit

Giữ S2 làm champion có điểm chính thức. Có thể submit đúng ZIP R4 high-confidence
như một challenger; không dùng các output selection R3 hoặc hai policy R4 đã
fail gate. Sau khi có điểm BTC của R4, so trực tiếp với S2 trước khi tiếp tục
tuning để tránh overfit proxy local.

# Kiểm tra và sửa chất lượng A0 — 24/10/2026

## Phạm vi và mốc đối chứng

Mốc BTC do người dùng cung cấp: FINAL_SCORE 0,2032; DOCS_F2MACRO 0,2289;
CHUNKS_F2MACRO 0,1775; DOCS_PRECISION/RECALL 0,1193/0,3845;
CHUNKS_PRECISION/RECALL 0,1975/0,2021. Đây là feedback từ ảnh người dùng,
chưa tải độc lập từ BTC. ZIP liên quan theo ngữ cảnh hội thoại có SHA-256
`c3426c0720e1bfb81b78754120ef91c3ec32c35e274a2e4efeef706c37de8084`.

Đã kiểm tra 1.200 final checkpoints, 239.160 documents và 72.000 chunks.
Mỗi query xuất đúng 60 chunks, trung bình 199,3 documents. Có 153.401 passage
ứng viên chưa được chấm do ngân sách rerank 200 pairs/query. Cache vẫn giữ
nguyên các passage này nên có thể chấm bổ sung mà không chạy lại retrieval.

Nhãn chẩn đoán gồm 2.190 chunks từng được BTC chấm đúng ở bản cũ, và 87
nhãn thủ công có official query ID. 20 nhãn EN/ZH tổng hợp trong bộ 107 nằm
ngoài 1.200 IDs nên được loại khỏi đánh giá này. Nội dung query của 519 query
có historical positives đã khớp SHA-256 với query hiện tại.

## Điểm mất evidence trong 2.190 historical positives

| Trạng thái đầu tiên | Số |
|---|---:|
| Đã chứa nguyên anchor cũ trong chunk hiện tại | 1.172 |
| Parent không có trong fused candidates | 253 |
| Parent có trong candidates nhưng chưa được score/chọn | 179 |
| Parent được chọn nhưng không có chunk slot | 253 |
| Có chunk, cửa sổ ngữ cảnh hiện tại quá hẹp | 24 |
| Có chunk nhưng cần passage khác hoặc ngữ cảnh xa hơn | 48 |
| Có chunk nhưng anchor cũ không nguyên vẹn trong source hiện tại | 260 |
| Source không có trong index | 1 |

Đây là phân loại first-failure trên nhãn lịch sử chưa đầy đủ. Exact containment
là proxy; không phải token-LCS chính thức, không biến các output chưa có nhãn
thành negative và không suy ra F2 từ các tỷ lệ này.

## Thay đổi và đối chứng

1. Chấm bổ sung toàn bộ passage chưa có score trong frozen candidate cache.
   Tái sử dụng 240.000 scores gốc, cùng BGE reranker revision, tokenizer,
   FP16, max_length=512 và batch_size=32. Chạy một worker trên GPU 2 đã
   preflight trống; output riêng, không thay baseline config hoặc score cache.
2. Merge score theo đúng `(query, doc_id, passage_id, input_sha256)`, từ chối
   coverage thiếu/thừa, duplicate, non-finite hoặc ghi đè score cũ. Chọn lại
   parent/seed từ các passage đã được score đầy đủ trong pool hiện có.
3. Thử document-only gate `score >= -4`, luôn giữ parent của mọi exported chunk.
   Gate được chọn trên development vì giữ 1.128/1.128 historical positives
   đã có và 51/51 manual positives, đồng thời giảm documents 24,02%.
   Validation riêng giữ 626/629 historical positives và 20/20 manual positives,
   giảm documents 24,40%. Đó là tradeoff đo được, chưa phải gain F2.
4. Mở rộng thêm tối đa một registered passage mỗi phía, cap 5.500 ký tự,
   bắt buộc chứa nguyên span đã xuất. Không dịch, viết lại hoặc nối text rời.
   Trên A0 gốc, exact historical anchors tăng 1.172 → 1.196, manual 57 → 59.

Các arms: baseline; document gate only; context only; baseline + gate + context;
supplemental rerank only; supplemental rerank + gate + context. Tất cả dùng
1.200 official query IDs và cùng giới hạn 200 documents, 60 chunks, 1 chunk/doc.
ZIP được kiểm với cap 100.000.000 bytes theo xác nhận của người dùng.

Split query theo SHA-256, 789 development + 411 validation; sáu query pilot cũ
nằm trong development. Đây không phải holdout chưa từng được nhìn thấy vì
historical labels đã từng phục vụ chẩn đoán. Quy tắc chọn candidate đã được ghi
trước khi đọc kết quả supplemental rerank: không giảm exact-anchor count ở
cả hai bộ nhãn, giữ ít nhất 95% baseline known-positive documents; kiểm lại cùng
gate trên validation. Nếu supplemental rerank không đạt, fallback là baseline
với document gate và context expansion. Official F2 cần BTC chấm riêng.

## Artifacts và tái lập

- `outputs/v4/a0_quality_diagnosis_20261024/`: stage audit, query split,
  policy development/validation, policy lock và quy tắc đánh giá.
- `outputs/v4/a0_remaining_passages_20261024/`: runtime lock và GPU scores bổ sung.
- `outputs/v4/a0_refined_candidates_20261024/`: final checkpoints đã kiểm source.
- `outputs/v4/a0_quality_comparison_20261024/`: so sánh các arms và positive losses/gains.
- `outputs/v4/a0_quality_submissions_20261024/`: các ZIP ứng viên đã validate.

Mã mới nằm trong `src/medrag/evidence/refinement.py` và các script
`diagnose_a0_quality.py`, `score_a0_remaining_passages.py`,
`refine_a0_submission.py`, `compare_a0_refinements.py` dưới `scripts/v4/`.

Kiểm thử tập trung: 47 PASS, gồm bảo toàn chunk parents/text, cap không được cắt
span cũ, grounding/provenance, merge coverage và từ chối ghi đè score baseline,
cùng các tests selector, context, submission validation và split runner hiện có.

## Kết quả hoàn tất

GPU bổ sung hoàn tất 1.200/1.200 query: tái sử dụng 240.000 scores gốc và chấm
153.401 pairs mới trong 650,21 giây model time. Replay/source validation hoàn tất
393.401 pairs. Candidate `rerank_refined` qua gate đã khóa trên development và
validation.

| Tập / metric known-positive | Baseline | Candidate | Gain | Loss |
|---|---:|---:|---:|---:|
| Development historical document | 1.128 | 1.247 | 121 | 2 |
| Development historical exact anchor | 746 | 812 | 121 | 55 |
| Validation historical document | 629 | 679 | 56 | 6 |
| Validation historical exact anchor | 426 | 449 | 52 | 29 |
| Development manual document / exact | 51 / 39 | 55 / 43 | 4 / 4 | 0 / 0 |
| Validation manual document / exact | 20 / 18 | 21 / 19 | 1 / 1 | 0 / 0 |

Trong số known-positive documents baseline, candidate giữ 1.126/1.128 trên
development và 623/629 trên validation, đều vượt gate 95%. Tổng document output
giảm 239.160 → 209.508; chunks giữ 72.000. Bản mới đổi chunk parent ở 11.312
slots và mở rộng context ở 51.789 slots.

ZIP cuối: `outputs/v4/a0_quality_submissions_20261024/rerank_refined/submission.zip`.
Dung lượng 91.492.826 bytes, dưới cap 100.000.000; SHA-256
`164fbed38d3d0afdab29a736f6a654d59886fead72560e600a946def3136967d`.
ZIP CRC/schema/query/payload validation PASS. Kiểm tra độc lập xác nhận đủ 1.200
queries, 72.000 chunks và toàn bộ 72.000 chunk texts là substring nguyên văn
trong đúng registered source; có 52.775 unique chunk parents.

Candidate có bằng chứng proxy tốt hơn trên known positives nhưng chưa có official
F2. Chỉ BTC scoring mới xác nhận nó có vượt FINAL_SCORE 0,2032 hay không.

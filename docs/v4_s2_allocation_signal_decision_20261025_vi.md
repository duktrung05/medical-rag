# Quyết định sau audit tín hiệu allocation S2 — 25/10/2026

**Dừng pilot chấm 13.546 contexts trước annotation, fit model và GPU inference.**
Các sửa F01/F03/F06 tiếp tục có giá trị; quyết định này chỉ bác bỏ việc dùng
**lại chính parent reranker logit S2** để đổi 60 chunk slots. Nó không bác bỏ
evidence-aware allocation nếu về sau có tín hiệu relevance mới đáng tin.

Audit đọc tất cả 1.200 refined query checkpoints, đối chiếu SHA-256 với frozen
manifest và S2 ZIP hash. Không chạy retrieval, model inference hoặc tạo submission.
Chi tiết per query tại `outputs/v4/s2_allocation_signal_audit_20261025/per_query.csv`.

| Chỉ số | Kết quả |
|---|---:|
| Queries có đúng 60 S2 chunk parents ở top-60 ranked parents | 1.200 / 1.200 |
| Không có parent ngoài 60 trong S2 documents | 19 |
| Parent ngoài 60 có logit cao hơn parent thấp nhất giữ slot | 0 |
| Chỉ hòa điểm ở biên | 25 |
| Parent ngoài 60 có logit thấp hơn ở biên | 1.156 |

Do đó một allocator chỉ sắp theo `max(seed reranker logit)` sẽ tái tạo thứ tự
S2. Thay tie-break ở 25 query hòa điểm không phải bằng chứng relevance. Đào tạo
pilot logistic trên ba features cũ sẽ cần chấm hàng nghìn contexts rồi mới biết
có tín hiệu mới hay không; chi phí này không phù hợp ưu tiên hiện tại. Namespace
`s2_evidence_allocation_20261025` được giữ để truy vết và đánh dấu
`STOP_BEFORE_ANNOTATION`; không điền các CSV review trong đó.

Đường tiếp theo cần bắt đầu từ lỗi đã đo được, không tạo tiếp R9/R10. Với
CHUNKS_F2, chỉ mở lại allocation khi có tín hiệu evidence độc lập ở inference
và phép thử nhỏ chấm riêng những contexts bị swap; cùng lúc DOCS_F2 0.2788
đòi hỏi chẩn đoán retrieval coverage/document ranking theo R7. Những con số
324/278 vẫn là historical anchor preservation, chưa phải official positives.

Điều kiện cho bất kỳ challenger mới nào: chính sách và split khóa trước khi mở
validation, đánh giá trực tiếp các context thay đổi, ghi gross gains/losses,
qua structural validators và chỉ sau đó xem xét BTC. Không suy diễn official F2
từ logit, historical anchors hoặc oracle ceilings.

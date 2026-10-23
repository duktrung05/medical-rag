# P1.5: Phân tích score và failure cases (21-10-2026)

**Trạng thái: đóng chẩn đoán.** Giữ A0 và chuyển các giả thuyết về lựa chọn passage sang P2. Không promote policy allocation từ sáu query này.

## Phạm vi

Đọc lại cached scores, branch traces và token diagnostics của P0 cho năm exact-recoverable parents. Không gọi retrieval/model và không dùng labels để chọn candidates hoặc fit policy. `dense_similarity` là cosine score đã có trong dense branch trace; không tính embedding mới. Đánh giá ngữ nghĩa dưới đây là đọc passage thủ công để phân biệt “có từ/cụm anchor” với “evidence trả lời đúng câu hỏi”.

| Query / doc | Rerank rank / score | Cách cutoff top-60 | Gap score tới rank 60 | Dense rank / cosine | BM25 rank | Query coverage | Pair tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| 53 / 756661 | 99 / 1.813 | 39 slot | 0.923 | 105 / 0.7111 | — | 0.636 | 402 |
| 130 / 553119 | 75 / 1.060 | 15 slot | 0.867 | 96 / 0.7118 | — | 0.468 | 474 |
| 130 / 2968260 | 126 / -1.901 | 66 slot | 3.828 | 64 / 0.7192 | — | 0.013 | 538 (truncated) |
| 754 / 2713777 | 70 / -1.545 | 10 slot | 0.513 | 70 / 0.6404 | — | 0.000 | 371 |
| 771 / 484565 | 100 / 3.305 | 40 slot | 0.766 | — | 77 | 0.667 | 423 |

Khoảng cách là `rerank_score(rank 60) - candidate_score`; số slot là `rerank_rank - 60`. Semantic score/rank vắng ở query 771 vì parent không xuất hiện trong dense branch trace. Các điểm so với median của top-60 cũng nằm trong CSV: bốn dense candidates có percentile cosine từ 0.327 đến 0.686 trong phân bố top-60 của query tương ứng; cosine không tách riêng được positive. Cả năm đều có fused rank tệ hơn rõ rệt so với median fused rank của top-60. BM25 không trả về bốn parent; parent còn lại có BM25 rank 77.

## Đọc từng passage

- **53 / 756661:** Passage bàn về tức ngực và nêu các nhóm nguyên nhân tim mạch, phổi, trung thất, cơ xương khớp; đây là evidence liên quan ở mức triệu chứng nhưng câu trả lời chung chung. Reranker có thể đã phạt vì evidence không giải quyết sâu câu hỏi. Dense score thấp hơn median top-60, fused rank 176 và không có BM25, nên chưa đủ bằng chứng gọi đây là lỗi reranker thuần túy.
- **130 / 553119:** Passage giải thích tắc tuyến lệ, đường thoát nước mắt và nói tình trạng có thể gặp ở trẻ sơ sinh. Đây là match trực tiếp nhất trong năm ca. Dense score gần median top-60 nhưng fused rank 197 và không có BM25. Đây là trường hợp có cơ sở mạnh nhất cho under-ranking; nguồn yếu nhất nằm ở fusion/lexical survival trước allocation.
- **130 / 2968260:** Dense rank 64 và cosine cao, nhưng passage tiếng Trung mô tả bệnh nhân nữ 65 tuổi khác với trẻ sơ sinh trong query, rồi trả lời rằng ca được mô tả không phải viêm túi lệ sơ sinh. Query coverage chỉ 0.013 do khác ngôn ngữ; reranker pair dài 538 tokens nên vượt giới hạn 512. Có overlap chủ đề và anchor, nhưng evidence này yếu hoặc trái hướng; không nên xem score thấp là lỗi chỉ dựa vào gold containment.
- **754 / 2713777:** Passage ghi CA125 là 40.41, còn query hỏi 64.80; không đưa ra diễn giải hay bước tiếp theo. Dense cosine nhỉnh hơn median top-60 của query nhưng coverage bằng 0, không có BM25 và rerank rank 70. Candidate liên quan chủ đề nhưng thiếu bằng chứng trả lời câu hỏi; thay thế top-60 sẽ có rủi ro.
- **771 / 484565:** Passage nói trực tiếp về xuất tinh sớm và tâm lý/điều trị, nên phù hợp với chủ đề query. Score 3.305 không thấp tuyệt đối nhưng còn 0.766 dưới cutoff; document BM25 rank 77 và không có dense hit. Đây là ứng viên under-ranking khả dĩ, nhưng feature hiện có chưa tạo một ngưỡng an toàn.

`context_contains_anchor_if_selected=true` trong bảng/JSON chỉ xác nhận khả năng chứa historical anchor nếu cấp slot, không khẳng định passage luôn có relevance lâm sàng đủ mạnh. `a0_exact_anchor=false` ở cả năm vì hiện tại chưa được xuất vào chunk.

## Trả lời ba câu hỏi

1. Có ít nhất một passage bị đánh thấp dù liên quan trực tiếp: rõ nhất là `130/553119`, và có thể cả `771/484565`. Nhưng các ca còn lại là evidence rộng, lệch bệnh nhân/ngôn ngữ hoặc lệch giá trị xét nghiệm. Do đó không thể kết luận chung rằng reranker đang bỏ qua evidence tốt; một số score thấp có thể là hành vi đúng.
2. Không có feature đơn lẻ đáng tin để tách cả năm khỏi top-60. Dense cosine của bốn candidates không vượt trội phân bố top-60 một cách nhất quán; fused ranks của cả năm yếu; BM25 thiếu ở bốn ca; query coverage trải từ 0 đến 0.667. Khác biệt quan trọng nhất là nội dung và quan hệ trực tiếp với câu hỏi, nhưng đánh giá đó hiện chưa có feature tự động đã được kiểm chứng.
3. Chưa có cơ sở để bật conservative replacement an toàn. Một quota/threshold theo dense rank, coverage hoặc score gap sẽ lấy cả positive yếu/trái hướng, còn rule theo historical anchor sẽ là leakage. Giữ A0 làm baseline; chỉ thử replacement trên development set riêng sau khi có relevance signal được định nghĩa trước và được kiểm nghiệm độc lập.

## Quyết định

Không thêm công thức weighted score và không chạy thêm arm trên sáu query pilot. Dùng các ca `130/553119` và `771/484565` làm giả thuyết phân tích cho P2 passage-aware reranking; với `130/2968260`, cần lưu ý truncation và mismatch ngôn ngữ/đối tượng, nhưng không mặc định tăng score là tốt hơn. P1.5 không thay đổi allocation production/baseline.

Chi tiết có thể tái lập nằm tại `outputs/v4/p1_5_score_diagnostics_20261021_v2/`; harness CPU tại `scripts/v4/analyze_p1_5_score_diagnostics.py`. Manifest ghi hashes input/output, `model_calls=0`, `retrieval_calls=0`.

# Đánh giá V1/V2

V1 leaderboard đã ghi nhận: FINAL_SCORE 0.1681; DOCS_F2MACRO 0.2802; CHUNKS_F2MACRO 0.0560. V2 chỉ trên pilot 192 source chunks, 50 query; không có gold/scorer tương ứng nên **PENDING_OFFICIAL_SCORING**. Không tuyên bố V2 vượt V1.

Technical metrics trên cùng pilot: trung bình 163.7 unique documents ở global top-200 so với 192 ở diverse top-200; retrieval latency trung bình 0.04338450199999897 giây/query (không gồm encode query/rerank). Mỗi nhánh có hit count theo query trong CSV. Các số này không đo relevance.

CPU rerank budget 4/query giới hạn output thực tế tối đa 4 document và 4 evidence/query dù schema cho phép 10. Không dùng tỷ lệ này để ước lượng F2 full-scale.

Lexical A/B trên **cùng mẫu 72 document, 50 query**: V1-style title + 6 chunk đầu so với V2 toàn source chunks, top-20. V2 có trung bình 1.3 document/query không nằm trong V1 top-20; 42/50 query có ít nhất một document như vậy. Đây chỉ là thay đổi candidate set, chưa xác minh document đó relevant.

Thiếu full-corpus prediction trên cùng 1.200 query, official matching/duplicate logic, và nhãn exhaustively adjudicated. Schema preview có 1.150 query rỗng có chủ ý để test validator; không phải submission thi đấu.

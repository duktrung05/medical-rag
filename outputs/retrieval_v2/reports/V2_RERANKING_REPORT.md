# Reranking và document scorer

Model baseline `BAAI/bge-reranker-v2-m3` revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, Identity logits, CPU, max length 384 tokens, tối đa 4 candidates/query. Authentic logits được ghi vào `V2_RERANKED_CANDIDATES.parquet`; tokenizer kiểm tra input trước truncation: 3/200 cặp vượt giới hạn. Nếu vượt, logit chỉ ứng với input đã truncate.

`V2_DOCUMENT_SCORING.csv` so control max-logit với scorer đơn giản dùng passage thứ hai và fusion rank. Có 0/200 document-query có từ hai passage đã rerank; nếu bằng 0 thì pilot chưa đo được tác động của tín hiệu hỗ trợ thứ hai. Chưa có nhãn để quyết định scorer nào tăng relevance; không hiệu chỉnh theo gold. Rerank hiện đã chạy.

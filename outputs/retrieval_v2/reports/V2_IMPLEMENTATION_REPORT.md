# Tình trạng triển khai

**Trạng thái:** `V2_PILOT_RUNNING`.

- Đã triển khai: remote pinned Parquet sampler, audit mẫu, source windows, document/passage FTS5, dense FAISS pilot, fusion RRF, document diversity, reranker CPU, document scorer, evidence selector, provenance validation, official-schema preview và test.
- Mẫu smoke: 192 source chunks, 229 passages, 50 query đã truy hồi; dense index đã tạo; rerank đã chạy.
- Mẫu nhiều chunk/document: 969 source chunks, 1080 passages của 72 document; chỉ benchmark lexical, chưa có dense/rerank cho mẫu này.
- Chưa có: index full corpus, official gold/scorer, full-corpus V2 submission, đánh giá F2 tương đương V1, resume/sharded dense build cho 20,5 triệu passage.

# Index pilot

- Smoke sparse SQLite FTS5: 1,564,672 bytes; dense FAISS IndexFlatIP: 938,029 bytes, 229 vectors × 1024 dimensions. Inner product trên embeddings L2-normalized.
- Mẫu nhiều chunk sparse SQLite: 7,573,504 bytes. Index document chứa **toàn bộ source chunks** dưới dạng postings từng chunk, aggregate về document; passage index tra trực tiếp trên mọi window.
- FAISS ID 0-based + 1 ánh xạ `passages.rowid`, từ đó truy ra `passage_id`, `doc_id`, chunk nguồn và exact text. Index dense hiện là exact **chỉ cho pilot**; không được mở rộng IndexFlatIP này lên 20,5 triệu passage trong RAM 16,85 GB.
- Kiểm tra lại substring và deterministic ID từ source extraction: smoke 229/229, mẫu nhiều chunk 1080/1080 passages hợp lệ.
- Chưa có benchmark ANN recall/memory nên chưa chọn IVF/HNSW/PQ. Full-scale sparse cũng cần phân shard/resume và đo kích thước thực.

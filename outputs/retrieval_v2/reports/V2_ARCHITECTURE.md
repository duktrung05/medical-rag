# Retrieval V2: kiến trúc pilot

Nguồn là extracted Parquet tại revision `bcb5a08fcff00385d7dabd0ac1807441630890c1`. Mỗi passage là substring nguyên văn của đúng một `(doc_id, chunk_index)`, có offset ký tự trong chunk và SHA-256 identity. Không suy diễn offset của HTML gốc.

Hai index FTS5 độc lập: `document_fts` chứa title + toàn bộ từng source chunk rồi gộp điểm về document; `passage_fts` chứa title + cửa sổ passage. Analyzer tạo token Unicode và bigram Hán. Dense BGE-M3 mã hóa passage, chuẩn hóa L2 và tìm bằng FAISS inner product. RRF hợp nhất theo rank với provenance từng nhánh; phân bổ một passage cho mỗi document trước khi dùng budget còn lại. BGE reranker chấm tối đa 4 passage/query trong CPU pilot. Document scorer kết hợp logit tốt nhất, passage hỗ trợ thứ hai và fusion rank. Evidence giữ đúng text nguồn, kiểm tra cha và substring trước khi xuất.

V1 trong `r2ai/` biểu diễn BM25 bằng title + tối đa 6 chunk đầu, dense document bằng title + 2 chunk đầu, rồi rerank pool document. V2 index trực tiếp passage ở mọi vị trí và cho phép passage đưa document mới vào pool. Production V1 không bị sửa.

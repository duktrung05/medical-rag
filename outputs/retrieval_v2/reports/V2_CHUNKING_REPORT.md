# Chunking và provenance

Pilot dùng target 192 từ, overlap 15%, ưu tiên kết thúc câu/dòng gần mốc. Chỉ cửa sổ nằm trong **một** source chunk; không nối các chunk khác nhau. Passage lưu `source_doc_id` dưới tên `doc_id`, `source_chunk_index`, `start_char`, `end_char`, `passage_id`, title và text substring. Cửa sổ ngắn giữ nguyên toàn chunk.

Smoke: 192 source chunks → 229 passages. Mẫu nhiều chunk/document: 969 → 1080 passages. Kích thước 256/384 chưa được đo về relevance; cấu hình chỉ ghi chúng như thí nghiệm sau.

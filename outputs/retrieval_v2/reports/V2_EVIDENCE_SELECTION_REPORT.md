# Chọn evidence

Từ reranked passages, chọn document theo score tổng hợp, rồi chọn exact source windows có parent trong document output. Loại trùng `(doc_id, chunk_text)`. `assert_grounded` kiểm tra mỗi text là substring của source chunk đúng document.

Chưa thử mở rộng cửa sổ ngoài một source chunk vì extracted Parquet không chứa offset body/HTML gốc. Không có LLM rewrite. Official text/LCS scorer chưa có tại local nên không thể tối ưu F2 từ kết quả pilot.

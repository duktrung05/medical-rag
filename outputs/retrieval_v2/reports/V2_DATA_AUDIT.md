# Kiểm tra dữ liệu

- HF revision `bcb5a08fcff00385d7dabd0ac1807441630890c1`; footer báo 20,523,261 rows/103 row groups. Remote revision đã ghim; SHA-256 nguồn `7580d46bb839ac694312a7a4e24b7afc00a052cdf64da254a4f11edd4ee98be8` lấy từ manifest audit trước, **chưa băm lại toàn file qua mạng**. SHA-256 pilot local `567a061e2f2d800801812890eff9f5ebead7709d3621252d8477eefafa82ac68`.
- Query local 1,200 rows, URL mapping 4,394,718 IDs. Không có ID mẫu thiếu trong mapping: 0. Duplicate `(doc_id, chunk_index)` trong mẫu smoke: 0; text rỗng: 0.
- Mẫu nhiều chunk: 969 hàng/72 documents; title không nhất quán 0 documents; text lặp trong cùng document 0; 603 chunks dưới 10 từ; 11 documents trên 20 chunks; nhiều nhất 59 chunks/document.
- Phân bố độ dài p10/p50/p90: [2, 5, 197] từ. Proxy hệ chữ: {'latin_dominant': 205, 'han_dominant': 764}; đây không phải nhãn ngôn ngữ.
- Có 506 chunks với `chunk_index > 6` trong mẫu nhiều chunk. Chỉ kiểm tra mẫu 3 row groups [0, 51, 102]; chưa kiểm tra uniqueness và chất lượng toàn bộ 20,5 triệu hàng.

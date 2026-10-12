# Tài nguyên mở rộng

Pilot smoke: 229 passages; sparse 1.56 MB; dense 0.94 MB. Mẫu lexical lớn hơn: 1080 passages, SQLite 7.57 MB. Tỉ lệ pilot SQLite không thể nội suy tuyến tính tin cậy do vocabulary/domain/row-group sampling.

Nếu 20,52 triệu chunks tạo ít nhất một vector 1.024-d float32, riêng vector thô khoảng **84,1 GB** (chưa gồm FAISS, metadata, model và OS); máy có RAM tổng 16.85 GB. Nếu mọi chunk chia thêm passage, chi phí tăng tiếp. Vì vậy cần sharded encode và benchmark ANN recall so exact trên sample trước khi chọn IVF/HNSW/PQ. GPU RTX 4050 6 GB không đủ để giữ full index trong VRAM; chưa có phép đo throughput full-scale để ước lượng thời gian chắc chắn.

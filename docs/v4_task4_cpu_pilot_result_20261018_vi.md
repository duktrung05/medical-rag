# Task 4 V4 — kết quả pilot dense trên CPU (18/10/2026)

## Phạm vi

- Dùng sparse pilot 2 shard đã khóa: 10.000 sources, 12.347 passages.
- Encode BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`, input title + newline + passage, tối đa 512 tokens, vector FP32 1.024 chiều.
- Cấu hình riêng `configs/v4_phase2_cpu_pilot.yaml` đặt cả dense và reranker trên `cpu`; không sửa cấu hình GPU hay sparse full.
- Worker dense chạy với affinity 4 CPU, `nice 15`, I/O priority idle, `CUDA_VISIBLE_DEVICES=''`, model cache offline. Bị dừng có chủ đích sau 256 rows rồi resume cùng contract để kiểm tra checkpoint.

## Kết quả

- `outputs/v4/task4/dense_cpu_pilot_2parts_20261017/manifest.json`: `complete`, 12.347/12.347 rows, 2/2 parts, dimension 1.024; SHA-256 của ba artifact trong manifest đều khớp.
- Mapping SQLite có đúng 12.347 rows; so bằng SQL `EXCEPT` với pilot dense trước cho 0 row khác. Mọi vector hữu hạn; norm L2 nằm trong khoảng `0.999999821..1.000000119`.
- So với pilot dense trước cùng contract: vector không byte-identical; max absolute element delta `7.13393e-6`, cosine từng cặp tương ứng tối thiểu `0.99999999805`. Đây là kiểm tra representation, **không** tự động chứng minh top-K/selected parity hoặc chất lượng retrieval.
- Thời gian từ lúc tạo contract đến lúc xuất manifest: 10.307 giây (~2,86 giờ), gồm cả lần dừng/resume và nạp model; throughput wall khoảng 1,20 passages/giây. Batch ổn định thứ hai: khoảng 90 giây/128 passages trên 4 CPU.

## Quyết định vận hành

Ngoại suy tuyến tính 2.951.224 passages theo throughput wall pilot là khoảng 28,5 ngày. Đây chỉ là phép tính thô; độ dài passage, cạnh tranh tài nguyên, hashing và I/O toàn corpus có thể làm thay đổi thời gian. Không mở dense full trên CPU và không mở pilot lớn hơn chỉ để xác nhận rằng nó chậm. Sparse full tiếp tục chạy CPU độc lập. Bước tiếp theo cần GPU được cấp quyền hoặc thiết kế thực thi phân tán/đa máy được phê duyệt; W4 near-tie canary và resource gate vẫn chưa PASS.

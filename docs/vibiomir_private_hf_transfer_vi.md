# Bản chụp ViBioMIR trên Hugging Face private

Đích: <https://huggingface.co/datasets/duktrung/med-rag>.

Phạm vi gồm các tệp hiện có trong `data/vibiomir/` và `data/raw/vibiomir/`.
Không lấy các dataset khác. Trang crawl được gom thành tar khoảng 1 GiB;
extract, index, cache embedding và manifest crawl được tải trực tiếp từ đường
dẫn gốc. Không tạo thêm một bản sao toàn bộ dữ liệu và không xóa dữ liệu nguồn.
Tệp `.partial.*`, `.tmp`, `.lock` bị loại. Danh sách tệp được chốt trong plan;
index/cache mới tạo sau lúc lập plan không tự động được thêm vào bản chụp này.

Lệnh upload kiểm tra repo private và quyền ghi trước khi tải dữ liệu. Mỗi tar
tạm chỉ được xóa sau khi kiểm tra dung lượng và checksum LFS trên Hub. Các tệp
extract đang ghi được chờ đóng trước khi upload; Parquet được kiểm tra footer.
Manifest hoàn chỉnh chỉ được xuất bản sau khi mọi tệp đã tải thành công.

## Đăng nhập trên server

Không đưa token vào code hoặc chat. Dùng token có quyền ghi vào repo đích:

```bash
HF_HOME="$HOME/.cache/huggingface-medical-rag" \
  /data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/hf auth login
```

Job chờ đăng nhập sử dụng cùng `HF_HOME`, tránh lưu token trong cache model dùng
chung của server. Log và PID nằm trong `outputs/hf_med_rag_20261007/`.

## Khởi chạy hoặc tiếp tục upload

```bash
HF_HOME="$HOME/.cache/huggingface-medical-rag" \
  /data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/python -u \
  scripts/tooling/hf_vibiomir_snapshot.py upload \
  --repo-id duktrung/med-rag \
  --wait-for-auth \
  --extract-pid-file outputs/retrieval_extract_20261007.pid \
  --extract-log-file outputs/retrieval_extract_20261007.log
```

Chỉ chạy một uploader cho cùng thư mục state. Khi có lỗi mạng, chạy lại cùng
lệnh và cùng plan để tiếp tục; các tệp đã hoàn tất sẽ được kiểm tra trên Hub.
Không sửa hoặc tạo lại plan giữa chừng. Chạy thêm bản chụp mới cần state riêng
và đường dẫn/repo đích riêng để tránh ghi đè các shard của bản cũ.

## Khôi phục trên máy local

Trong môi trường Python 3.10 trở lên đã cài `huggingface_hub`, đăng nhập với
quyền đọc repo private rồi chạy:

```bash
hf download duktrung/med-rag snapshot/hf_vibiomir_snapshot.py \
  --repo-type dataset --local-dir .
python snapshot/hf_vibiomir_snapshot.py restore \
  --repo-id duktrung/med-rag --destination /duong/dan/medical-rag
```

Trên Windows có thể dùng đường dẫn `D:/medical-rag` cho `--destination`.
Lệnh sẽ phục hồi trực tiếp cấu trúc `data/` của dự án, tải và giải nén từng tar
rồi xóa tar tạm. Tệp có sẵn khác nội dung sẽ được giữ lại và báo lỗi; chỉ dùng
`--overwrite` khi chủ động muốn thay thế. Khôi phục dữ liệu không cần GPU.

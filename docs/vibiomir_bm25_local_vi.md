# Thử BM25 ViBioMIR trên Windows

Bộ index: https://huggingface.co/datasets/duktrung/med-rag/tree/main/data/vibiomir/bm25idx_crawl_20261007

Chạy các lệnh dưới đây từ thư mục gốc của dự án. Nếu cần cài môi trường mới:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[vibiomir]"
```

Repo private yêu cầu tài khoản có quyền đọc. Nhập token trực tiếp vào CLI, không đưa vào code/chat:

```powershell
.\.venv\Scripts\python.exe -c "from huggingface_hub import login; login(add_to_git_credential=False)"
.\.venv\Scripts\python.exe scripts/tooling/download_vibiomir_bm25.py
```

Downloader tải thư mục index, `bm25idx_crawl_20261007_ids.npy`, `bm25idx_crawl_20261007_vocab.json`,
`query.parquet` và `links_corpus.parquet` về đúng cấu trúc `data/`. Nó chốt cùng một commit,
kiểm tra kích thước và SHA-256 LFS, giữ các tệp local có nội dung khác, và có thể chạy lại để tiếp tục.
Giới hạn mặc định là 64 GiB dữ liệu chưa có trên máy; kích thước thực tế được in trước khi tải.
Không cần tải các tar HTML crawl, embeddings hay chunks để truy vấn document BM25.
Dùng `--index-only` nếu chỉ cần document ID và điểm, không cần query gốc hay URL.

Thử một câu hỏi:

```powershell
.\.venv\Scripts\python.exe scripts/runtime/query_vibiomir_bm25.py --query "Triệu chứng bệnh tiểu đường là gì?" --top-k 10 --output outputs/predictions/vibiomir_bm25_smoke.json
```

Kết quả gồm query, các term có trong vocab, document ID chính thức, điểm BM25, URL nếu có bảng lookup.
Tokenizer được dùng lại từ `r2ai/bm25index.py` để tương thích index đã build.
Lệnh dùng memory mapping cho mảng index/ID, đọc streaming vocab và URL; mỗi lần chạy sẽ quét vocab,
và có thể quét bảng URL. `elapsed_seconds` gồm đọc vocab, nạp index và retrieval, không gồm lookup URL.
Query không có term trong vocab trả danh sách rỗng; kết quả điểm 0 bị loại.

Đây là thử truy xuất document. Xem nội dung chunk/rerank cần tải thêm corpus chunks tương ứng;
đánh giá Recall/F2 cần nhãn gold phù hợp.

Sau khi tải xong, code batch hiện có vẫn dùng được nhưng nạp toàn bộ vocab/index vào RAM:

```powershell
.\.venv\Scripts\python.exe r2ai/bm25index.py search --out bm25idx_crawl_20261007 --top-k 10 --n-threads 1 --candidate-out bm25_crawl_local_test
```

Trên máy ít RAM, nên thử lệnh query đơn phía trên trước.

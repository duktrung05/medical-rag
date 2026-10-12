# Môi trường pilot

- Repo branch `master`, commit `25b7ac1`; source V1 baseline nằm trong `r2ai/` (`bm25index.py`, `index.py`, `fuse.py`, `rank.py`, `validate.py`). Các thay đổi V2 hiện ở working tree, chưa push.
- Python 3.12.14, logical CPUs 8, RAM tổng 16.85 GB, khả dụng khi lấy mẫu 4.36 GB, disk trống 180.0 GB.
- GPU thấy qua `nvidia-smi`: NVIDIA GeForce RTX 4050 Laptop GPU, 6,141 MiB; pilot chạy CPU theo giới hạn tác vụ.
- Venv đã cần `huggingface-hub>=1.5,<2` để tương thích Transformers cài tại máy. Model cache cục bộ, `HF_HUB_OFFLINE=1` cho bước encode/rerank.

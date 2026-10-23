# P5.3A — Google Colab T4 canary

## Tệp bàn giao

- Notebook: `notebooks/p5_3a_colab.ipynb`.
- Bundle upload: `outputs/v4/p5_3a_colab_canary_20261022/p5_3a_colab_bundle.zip`.
- Bundle SHA-256: `2d3869ea89c6e5b8d7c9943e3fbeb4fa8d78a2b358c2ec5409103a62b52e57f7`.
- Sáu query đã khóa: `53, 130, 380, 754, 771, 1144`; mỗi query 200 pairs, tổng 1.200 pairs.
- Model: `BAAI/bge-reranker-v2-m3`, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`; fp16, `max_length=512`, cùng `BGEReranker` và hàm `score()` của P5.2.

Bundle chứa pair shards, lock, pair manifest, input hashes, score contract SQLite rỗng, mã Python cần để import P5.2 scorer và dependency pins. Không có corpus, dense/BM25 index, score cache hoặc score shards. Bundle được quét các mẫu secret thông dụng trong bước tạo.

## Chạy trên Colab

1. Mở `p5_3a_colab.ipynb` bằng Google Colab.
2. Chọn **Runtime → Change runtime type → T4 GPU**.
3. Chọn **Runtime → Run all**. Cho phép Colab truy cập Google Drive khi được hỏi.
4. Tại hộp upload, chọn đúng **một** file `p5_3a_colab_bundle.zip` ở trên. Không cần sửa code hoặc đường dẫn.

Cell đầu đặt sẵn `DRIVE_OUTPUT_DIR = /content/drive/MyDrive/medical-rag/P5_3A_T4`. Kết quả được ghi vào `canary_2d3869ea89c6/` dưới thư mục đó: `run_lock.json`, sáu `scores/*.json`, sáu `provenance/*.json`, `score_manifest.json`, `gpu_canary_report.json` và `p5_3a_colab_scores.zip`.

Notebook kiểm tra hash bundle, cài đúng PyTorch/CUDA và dependencies, xác nhận Tesla T4, tải đúng Hugging Face revision đã khóa, rồi gọi trực tiếp P5.2 `score()` với `seed_cache=None` và `device=cuda:0`. Mỗi shard chỉ được checkpoint lên Drive khi P5.2 ghi nhận `200 model calls, 0 cache hits`; hash và nội dung score shard phải hợp lệ. Báo cáo ghi hashes model snapshot, tốc độ pairs/giây, peak VRAM, thời gian và ETA **chỉ cho GPU reranking** của 240.000 pairs dự kiến. ETA không gồm CPU retrieval, CPU finalization, I/O hay xuất BTC.

Nếu Colab ngắt, mở lại cùng notebook, chọn T4, Run all và upload **cùng bundle**. Các shard có score/provenance hợp lệ trên Drive được dùng làm checkpoint; query còn thiếu sẽ chạy fresh inference. Chạy lại sau khi đã hoàn tất sẽ kiểm tra và trả về ZIP đã lưu.

## Import và quyết định GO/NO-GO

Sau khi Colab hoàn tất, chép `p5_3a_colab_scores.zip` từ Drive về máy có repository rồi chạy:

```bash
.venv-phase2-batch/bin/python -m scripts.v4.verify_p5_3a_colab_canary \
  --scores-zip /duong/dan/den/p5_3a_colab_scores.zip
```

Importer kiểm tra toàn bộ score shards, provenance của fresh inference, hashes, runtime/model revision; tính score differences cho từng pair; chạy CPU finalization hiện có; so ranking, selected documents, chunks, contexts và BTC payload với pilot P2-A0 đã khóa. Báo cáo `verification_<hash>.json` có `GO` hoặc `NO-GO`, cùng CSV score differences và BTC artifacts trong thư mục `verification_<hash>/`.

**Trạng thái hiện tại:** bundle và notebook đã chuẩn bị, chưa chạy trên Colab T4 nên chưa có fresh GPU scores, throughput/VRAM hoặc GO/NO-GO thực nghiệm. Không chạy full 1.200 queries trong P5.3A.

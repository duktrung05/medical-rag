# Kết quả Task 3 — Đưa Phase 2 vào V4 core workflow

Ngày: 16/10/2026.

## Mục tiêu

Cho V4 CLI/API nạp trực tiếp dữ liệu Phase 2 theo `Source`/`Passage` contracts, dùng chung một V4 `Pipeline` cho BM25, dense retrieval, RRF, reranking và evidence grounding. Không tạo lại passages từ HTML.

## Thay đổi đã làm

- `src/medrag/data/phase2_batch.py` giữ `normalized_text` trong output batch để V4 có source representation mà passage offsets tham chiếu.
- `src/medrag/data/phase2_prepared.py` ghi/đọc V4 Phase 2 snapshot có manifest/checksums, giữ nguyên `passage_id`, offsets, source hashes và Phase 1/2 lineage; kiểm hash/offset trước khi dựng index.
- `src/medrag/core/factory.py` tạo chung retrieval pipeline theo cấu hình; `src/medrag/core/pipeline.py` nhận candidate/RRF/rerank budgets, prefix title cho dense/reranker và tránh tính passage BM25 fallback lặp lại theo từng parent.
- BGE-M3 và BGE reranker dùng revisions được pin và kiểm tra local snapshot trước khi nạp. `configs/v4_core.yaml` giữ model stages tắt; `configs/v4_phase2_subset.yaml` bật chúng cho run subset có model.
- CLI `prepare-phase2` nối explicit Phase 1/2 subset vào V4. CLI `search` và API cùng gọi `load_prepared` → `build_pipeline`, với cấu hình truy vấn được chọn rõ.
- `src/medrag/data/loader.py` chỉ nạp HTML extractor khi đường HTML được gọi; V4 Phase 2 CLI không phụ thuộc không cần thiết vào `bs4`.

## Run thực tế

Đã tạo V4 snapshot riêng từ subset khóa Task 2 tại:

`outputs/v4/phase2_subset/20261016/selection_seed20261016/`

- Phase release preflight: PASS; 1.000 documents, 2.743 passages, 20 official queries.
- Official corpus revision: `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`.
- Phase 1 release SHA-256: `483d7dd4e606ee1f6ed011e66d62116c97fb2db7bf6b75027e89b7be58198bae`.
- Phase 2 release SHA-256: `e1ecd3302dde6b705e2368daaf6c864a18a2831c08acfdbde8cac3c211e141f8`.
- `sources.jsonl` SHA-256: `8597987c203aba439c643026d053995ce6e4e551bf27e222fa405878c621492b`.
- `passages.parquet` SHA-256: `180cd3ae0e1cbe9ff475d249dadd9b5169a82417fee4a10ca5d678e5e1fe45f4`.

Sau đó đã gọi V4 CLI trên query ID 46 với `configs/v4_phase2_subset.yaml`, GPU vật lý 2. BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181` và bge-reranker-v2-m3 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` nạp thành công. Pipeline trả top 3 có evidence và reranker scores; top document IDs là `1144523`, `750399`, `1140752`. Đây là một truy vấn smoke trên corpus 1.000 documents, không phải nhãn relevance hay F2.

## Phạm vi hoàn tất

Task 3 đạt **V4 subset workflow integration PASS**: Phase 2 identity/provenance được nạp vào V4; CLI chạy được với retrieval, dense, RRF, reranker và exact-source verification. API dùng cùng loader/factory trong code nhưng chưa khởi chạy server trong lượt này.

Task 3 chưa xác nhận candidate quality/F2 và chưa thay đổi default `v4_core.yaml` để tự bật model. V4 dense implementation vẫn in-memory, snapshot giới hạn 1.000 sources. Task 4 còn phải build backend full-release, chạy đủ 1.200 queries, validate submission và nhận score BTC. Vì vậy chưa gọi đây là full-corpus V4 readiness.

## Lệnh tái lập

```bash
.venv-phase1/bin/python -m src.cli prepare-phase2 \
  --doc-ids outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/doc_ids.json \
  --query-ids outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/query_ids.json \
  --output outputs/v4/phase2_subset/20261016/selection_seed20261016

CUDA_VISIBLE_DEVICES=2 HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv-phase2-batch/bin/python -m src.cli search \
  --prepared outputs/v4/phase2_subset/20261016/selection_seed20261016 \
  --config configs/v4_phase2_subset.yaml \
  --query '<query text>' --limit 3
```

Các kiểm tra chỉ gồm preflight/snapshot validation và gọi CLI với model thật; không chạy test suite trong lượt triển khai này.

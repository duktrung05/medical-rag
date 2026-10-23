# Task 4 V4 — trạng thái triển khai 16/10/2026

Trạng thái: **CHƯA ĐẠT FULL_INDEX_PASS / FULL_V4_EXPORT_PASS**. Không có submission full 1.200 queries hay F2 chính thức. Tài liệu này ghi bằng chứng thực thi, không thay thế acceptance plan.

## Đã xác nhận

- Sparse pilot 2 shard: 10.000 eligible sources, 12.347 passages. Bản chạy liên tục tại `outputs/v4/task4/sparse_pilot_2parts_v2_20261016/`; bản resume tại `outputs/v4/task4/sparse_resume_pilot_2parts_20261016/`. Cả tám bảng dữ liệu `sources`, `passages`, hai bảng `docs`, hai bảng `postings`, hai bảng `df` so bằng SQL `EXCEPT` đều có **0 dòng khác** hai chiều. Hash SQLite không giống vì fingerprint/metadata và bố trí file khác; không dùng hash file làm phép so nội dung logic giữa hai lần build.
- Dense pilot 2 shard: `outputs/v4/task4/dense_pilot_2parts_20261016/`, 12.347 hàng, dim 1.024. Index kết hợp `outputs/v4/task4/combined_pilot_2parts_20261016/` đã load đủ sparse/dense/reranker từ manifest. Smoke trên CPU với giới hạn rerank 3 pairs trả 3 tài liệu có evidence, offsets và rerank score. Đây không phải cấu hình A0 đầy đủ hoặc đo chất lượng.
- Sparse pilot 50 official queries: mean 0,088 giây, P95 0,182 giây, RSS peak 297.144 KiB, theo `outputs/v4/task4/sparse_pilot_2parts_v2_20261016/benchmark_50q.json`.
- Canary 1.000 docs/2.743 passages/20 queries với real models: selected results và canonical payload bằng nhau; dense top-200 đảo thứ tự ở query IDs 92 và 927, max absolute score delta khoảng 1,16e-6, theo `outputs/v4/dense_canary_comparison_20261016/report_cosine64.json`. Trong disk ranking, các khoảng cách score kề nhau nhỏ nhất lần lượt 1,6383e-7 (rank 176) và 1,5018e-7 (rank 195); đây là **dấu hiệu near-tie, chưa chứng minh chính xác nguyên nhân**. Gate dense parity vẫn FAIL/chưa giải quyết; không tự nới tolerance hậu nghiệm.
- Đối chiếu cô lập **cùng FP32 passage vectors và cùng query vector**: RAM `Dense.search` và `DiskDense.search` khớp top-200 cả 20/20 queries, max absolute score delta 2,55e-15 (`outputs/v4/dense_canary_comparison_20261016/report_same_vectors_cpu.json`). Kết quả này loại trừ sai khác thuật toán search trên cùng input; nó **không** giải quyết canary hai lần encode model độc lập ở trên.
- Audit identity toàn bộ 922 release shards PASS (`outputs/v4/task4/release_coverage_audit_20261016.json`): official 1.220.598 IDs, Phase 1 PASS 1.073.531, semantic exclusions 6 sources/49 passages, eligible 1.073.525 sources/2.951.224 passages; không có PASS ID ngoài official, eligible parent thiếu passage hoặc duplicate passage ID.
- Sparse full 922 shard được khởi tạo và **dừng chủ động sau shard 2** tại `outputs/v4/task4/sparse_full_20261016.building/checkpoint.json`: `next_part=2`, 10.000 sources/12.347 passages. Chưa publish index full. Resume chỉ với code và input lock đúng fingerprint `8f8245e8357137b53a1a4a927d2e6532c65aa2ed60caf495997b74ca97603eb8`.
- Full runner đã bổ sung `selected/selected_results.jsonl` và `export/SHA256SUMS`, ghi atomically và so bytes khi resume. Chưa chạy 1.200 query nên đây mới là code contract, không phải artifact full.
- 30/30 V4 tests không dùng HTTP `TestClient` pass, bao gồm guard cấm full runner dùng pilot index. Hai tests HTTP không chạy xong trong phiên này: ngay cả `with TestClient(FastAPI())` tối giản cũng treo; thử `httpx.AsyncClient` vẫn treo khi gọi sync route. Đây là lỗi môi trường test/AnyIO–Starlette cần khắc phục riêng; không coi đó là kiểm thử API đã pass.

## Gate và việc tiếp theo

1. Phục hồi khả năng GPU: `nvidia-smi` hiện exit code 9, không giao tiếp được với driver. Không chạy full BGE-M3/reranker trên CPU chỉ để ép hoàn thành.
2. Khi GPU trở lại, chạy lại canary dense và xuất chính xác cặp ID/rank/scores lệch. Khóa quy tắc near-tie trước khi regenerate reference; phải xác nhận top-200 set và thứ tự ngoài tie band, selected/export exact.
3. Hoàn tất benchmark sparse+dense+rerank pilot ở cấu hình A0, kiểm resource/ETA và coverage ledger. Sau G4 PASS mới resume sparse full, build dense full, assemble và xác nhận G5.
4. Chỉ sau `FULL_INDEX_PASS` mới chạy 1.200 queries, replay, strict validation và deterministic ZIP. Bước scorer/BTC và F2 là riêng, hiện `UNMEASURED`.

Không có background worker full còn chạy khi ghi báo cáo này. Không chỉnh các file được liệt kê trong `implementation_sha256` của sparse lock nếu muốn resume checkpoint; nếu chỉnh phải tạo build namespace mới.

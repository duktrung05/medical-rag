# Kết quả thực thi sửa evidence V4 trước Task 4

Ngày 16/10/2026. Trạng thái: **PARTIAL — chưa mở full-query Task 4**.

Kế hoạch tiếp theo với chuẩn output, thứ tự triển khai và gates nghiệm thu: [Task 4 V4 output acceptance](v4_task4_output_acceptance_plan_20261016_vi.md).

## Đã thực hiện

- Khóa reference 20 official queries trên snapshot Task 3 (1.000 docs, 2.743 passages), model BGE-M3 và reranker pin revision, GPU 2. `outputs/v4/reference_20q_20261016_pre_fix/` cho canonical prediction SHA `bf30f81539531cd262ecc311fba69a1d294081ebed2b02d65c38277c18ae3a2b`. Lưu ý manifest pre-fix ghi config hash sau khi config được sửa song song; không dùng trường đó làm bằng chứng config pre-fix. Những run sau chụp hash trước inference.
- Tách selection ra khỏi serializer. `SelectionPolicy` chọn evidence theo cap cấu hình; adapter xuất toàn bộ selected evidence, không cắt `[:1]`. Evidence chưa rerank không được xuất khi reranker bật. Hai parent trùng text không bị dedup. Fallback BM25 tìm theo đúng parent.
- Rerank có hai allocation (`document_order`, `parent_first`) cùng 200 pair/query, trạng thái scored/budget và kiểm score hữu hạn. Dùng tokenizer model thật đo số pair dài hơn max length. API POST được sửa lỗi 422 do `Query` model đặt trong hàm với postponed annotations.
- Chạy A0/A1/A2 trên cùng 20 queries và snapshot: mỗi arm chấm 4.000 pairs. A0 chọn 200 docs/200 evidence, SHA `bf30f81539531cd262ecc311fba69a1d294081ebed2b02d65c38277c18ae3a2b`; A1 parent-first chọn 200/200, SHA `cea1cb4c1c64e9badda635237cc1d8d63801ff1c0acb92f64fe3ac88e0ac5077`; A2 parent-first, cap 2/doc chọn 200/224, SHA `825aaaa652a725ebeb5f0913bacf8c2ef64f0e207296a27027b860b4ca863512`. A2 có 181/4.000 pair dài hơn 512 token; evidence text/offset không bị cắt. Replay A2 cap1/cap2 khớp output gốc của A1/A2. Đây là integration/diagnostics, **không phải F2**.
- Tạo SQLite source/passage store và BM25 đầy đủ full-body/full-passage, global DF/average length. Xây canary từ snapshot khóa tại `outputs/v4/disk_subset_canary_20261016/`: 1.000 source rows, 2.743 passage rows, manifest kiểm SHA. Chạy lexical-only 20 query: output backend đĩa bằng chính xác output in-memory V4 (cả thứ tự/score/evidence). Chạy thêm 20 query với BGE-M3 và reranker thật, chia sẻ cùng dense/reranker in-memory và chỉ thay BM25/store: **20/20 output bằng chính xác**. Đây là phép so backend lexical/store, không chứng minh dense index trên đĩa. Test tail passage riêng bảo đảm passage cuối được truy hồi. Backend đĩa từ chối cấu hình bật dense/reranker khi stage chưa cung cấp, không âm thầm downgrade.
- Kiểm thử: `pytest -q tests/v4 tests/integration/runtime/test_api.py` đạt 21/21. Các test gồm tail passage qua CLI/API, multi-evidence export, budget/unscored/NaN, BM25 disk parity và corruption/fail-closed.

## Chưa hoàn tất / điều kiện trước Task 4

1. Disk canary hiện **chỉ lexical**. Chưa có dense vector index toàn corpus, full release streaming builder, checkpoint/resume/reconciliation 1.073.525 docs và 2.951.224 passages. Không đưa canary vào CLI/API production bằng cách bỏ giới hạn 1.000 docs.
2. F2–F3 còn thiếu score cache có fingerprint, strict validator và qrels regression. Chưa có strict full-query validator/submission ZIP cho 1.200 IDs, chưa có feedback BTC. F2 chính thức vẫn `UNMEASURED`.
3. A1/A2 thay đổi output nhưng subset 1.000 docs không có nhãn relevance đầy đủ, nên chưa chứng minh A1/A2 tốt hơn A0. Giữ A0 làm policy mặc định cho production cho đến khi có phép đo chất lượng.
4. Chỉ mở full-query inference sau khi F4 đạt coverage, fingerprint, subset parity với dense/reranker, benchmark tài nguyên và fail/restart tests. Job build artifact đúng fingerprint có thể tiếp tục; không dùng prediction/ZIP từ adapter cũ làm submission mới.

# Plan và nghiệm thu sửa giới hạn ngầm 60 documents — 21/10/2026

## Mục tiêu và phạm vi

Tách document selection khỏi chunk selection: một query có thể xuất 120/200 documents và tối đa 60 chunks. Mọi document vẫn phải có evidence hợp lệ để chứng minh lựa chọn nội bộ; evidence này không bắt buộc phải được xuất trong `relevant_chunks`. Mọi chunk xuất ra phải có parent thuộc `relevant_docs` và giữ nguyên source grounding.

Người dùng đã yêu cầu lập plan và thực hiện. Lượt này triển khai code/config/tests và CPU replay từ scored traces có sẵn. Không mở full inference, assemble/verifier corpus, nộp bài hoặc gọi model. Giữ nguyên code/input/config trong dense build contract, giữ nguyên config A0 và outputs cũ. Không crawl/embed lại.

## Thiết kế được chọn

1. Bổ sung `SelectionPolicy.document_selection` với hai mode: `with_chunks` giữ hành vi baseline A0 để đối chứng; `independent` chọn document độc lập chunk cap. Cấu hình mới phải bật explicit `independent`; không đổi ngầm semantics của config/checkpoint lịch sử.
2. Giữ contract runtime list-of-documents để CLI/API/runner/replay không cần đổi format checkpoint toàn bộ. Trong mode mới mỗi selected document có `document_evidence` là một passage scored/grounded dùng làm căn cứ nội bộ; `evidence` chỉ chứa chunks được xuất, có thể rỗng khi hết chunk budget. Adapter chỉ xuất `official_id` và `evidence`, không xuất `document_evidence`.
3. Chọn document theo thứ tự ranking hiện có với `min(limit,max_docs)`. Chỉ nhận document có ít nhất một eligible passage; khi reranker bật, passage phải có finite numeric rerank score. Không nâng unscored documents thành scored hoặc ép đủ cap. Không thay fusion, score aggregation, thresholds hoặc model.
4. Chunk selection giữ thứ tự document/evidence đã rank và dedup trong cùng parent; giới hạn riêng `max_chunks`/`chunks_per_doc`. Tăng chunks_per_doc không giảm document membership khi mode independent. Giữ nguyên text ở hai official parents khác nhau.
5. Validator nội bộ kiểm source/identity/offset/text/finite-score cho cả document_evidence lẫn chunks. Payload validator independent cho phép documents không có exported chunks; vẫn kiểm chunk parents, caps, duplicate IDs/text, query coverage, UTF-8/JSON/ZIP. `with_chunks` giữ yêu cầu cũ. Đây là thay đổi policy nội bộ; phần scoring người dùng cung cấp đánh giá hai danh sách riêng, không xác lập mọi doc phải kèm chunk. Chưa tự khẳng định đã có toàn bộ hidden scorer rules.
6. Factory, disk pipeline và CPU replay đọc mode mới thống nhất. Full-run lock phải thêm hash selector/serializer contracts liên quan để resume không trộn checkpoint có semantics khác. Không nối checkpoint inference cũ bằng sửa lock.
7. Tạo hai cấu hình mới dựa trên A0 GPU3: parent_first, budget 200 pairs, document caps 120 và 200, chunk cap 60, 1 chunk/doc. Không tăng số pairs, không sửa A0 GPU3 đang phục vụ dense. Hai cấu hình có chung retrieval/model input nên có thể dùng chung scored trace khi replay; trace A0/document_order không giả làm trace parent_first.

## Trình tự thực hiện

| Bước | Thực hiện | Điều kiện hoàn tất |
|---|---|---|
| P0 | Chụp SHA code dense/config A0; đọc callers/tests | Có protected hashes, xác định không sửa file dense contract |
| P1 | Selector mode independent, document_evidence và diagnostics | 200 eligible parents + 60 chunk budget vẫn chọn đủ 200 docs; input không bị mutate |
| P2 | Validator/source grounding, factory/disk/replay và locks | Document-only export vẫn có căn cứ nội bộ; payload đúng 3 keys; parent/score/identity sai bị chặn |
| P3 | Config d120/d200 parent_first | 200 pairs tối đa; cả hai giữ dense/model/tokenization contract A0 |
| P4 | Regression tích hợp CPU | Test scored/unscored, caps, duplicates, invalid scores, corrupted anchors/chunks, ZIP roundtrip, allocation và cache replay |
| P5 | Replay trace model thật có sẵn | Artifacts mới, hashes/source trace được ghi; báo coverage/counts, không suy F2 |
| P6 | Tổng kết và cập nhật plan dùng config mới | Tests PASS, protected hashes không đổi, số liệu tái hiện được |

## Kiểm thử bắt buộc

- 200 documents có score hợp lệ: mode cũ <=60 docs/60 chunks; independent d120 =120 docs/60 chunks; d200 =200/60. Tăng chunks_per_doc không giảm docs.
- Document không có eligible/scored evidence không được xuất; NaN/Inf/bool hoặc sai trạng thái phải fail. Không có exported chunk vẫn cần document_evidence hợp lệ.
- Parent của document_evidence/chunk phải đúng; text/offset/source verify như trước. Không cho thêm raw official ID rồi bỏ qua grounding.
- Payload cho phép relevant_docs nhiều hơn parents trong relevant_chunks chỉ với independent; chunk parent ngoài danh sách vẫn fail. Kiểm query IDs, duplicates, caps, bool IDs và ZIP.
- Pipeline với parent_first và 200 pairs chấm được 200 distinct parents khi pool đủ; allocation document_order làm đối chứng, không đổi thuật toán của baseline.
- CPU selection replay lặp lại cho canonical payload/hash giống nhau, không gọi model. Chỉ thay selection trên cùng trace; không giả định allocation mới đã được rerank.
- So hash protected files trước/sau. Các artifacts/report lịch sử không ghi đè.

## Tiêu chí chất lượng và bước sau

Fix hoàn tất khi loại được giới hạn ngầm, validators vẫn chặn dữ liệu sai và cache/replay/export đúng. Đây không phải chứng minh tăng F2. Không chọn d120/d200 bằng output counts. Sau khi dense hoàn tất và người dùng mở bước inference, cần pilot quality trên full corpus rồi feedback/scorer đúng artifact để chọn policy; không chạy tự động trong lượt sửa này.

Rollback bằng config `document_selection=with_chunks` (hoặc config A0 cũ không có field này) trên một run mới phù hợp lock; giữ output/code snapshot lịch sử. Không sửa lock để resume checkpoints khác implementation.

## Kết quả thực thi

**Đã hoàn tất triển khai và kiểm chứng chức năng; F2 mới vẫn UNMEASURED.**

- Đã thêm mode, document support nội bộ, diagnostics, factory/disk/replay loading, mode-aware validator và full-run serializer hash lock.
- Đã tạo hai config independent d120/d200, parent_first, cùng 200 pairs/query. Config A0 GPU3 và ba file trong dense implementation contract giữ nguyên SHA trước/sau.
- `pytest tests/v4`: **57/57 PASS**, gồm regression CLI/API mới, disk configuration, 200 parents/200 pairs, grounded document support, bad scores, wrong parent/text/offset, separate caps, legacy behavior và ZIP roundtrip. TestClient treo trong sandbox ở bước thread portal; cùng test và toàn bộ suite chạy ngoài sandbox PASS trong 2,53 giây, không dùng GPU/model. Có một deprecation warning từ dependency TestClient, không có test failure.
- CPU replay dùng trace model thật `reference_20q_20261016_parent_first_chunks2`, khóa checksum diagnostics và snapshot 1.000 docs/2.743 passages. Trace chứa ranked candidates nên có thể chọn lại ngoài result limit 10 ban đầu; replay explicit `--result-limit 200`, không tạo thêm scores hoặc giả trace full corpus.
- D120: 2.400 documents/1.200 chunks trên 20 queries = 120 docs/60 chunks mỗi câu. D200: 3.944 documents/1.200 chunks = 188–200 docs và 60 chunks mỗi câu. Không ép đủ 200 khi không đủ parents có score.
- Kiểm grounding cho toàn bộ document_evidence và exported chunks trên snapshot khóa; payload đúng schema và ZIP PASS. ZIP tạo lại cùng hash. Documents d120 là subset d200 và exported chunks hai arm bằng nhau.
- Replay legacy giữ chính xác canonical payload SHA `825aaaa652a725ebeb5f0913bacf8c2ef64f0e207296a27027b860b4ca863512` của run nguồn.
- Các ZIP replay chỉ có 20 queries dùng cho kiểm thử; không phải submission 1.200 queries và không được coi là artifact full.

Bằng chứng nằm ở `outputs/v4/independent_selection_fix_20261021/`: `tests_v4_final.xml`, `replay_validation.json`, `protected_hashes_before.json`, `protected_hashes_after.json`, `implementation_result.json`, các thư mục `replay_d120`, `replay_d200`, `replay_legacy`.

Lệnh replay đã dùng (đổi d120 sang d200 cho arm còn lại; mỗi output phải là thư mục mới):

```bash
PYTHONPATH=. .venv-phase2-batch/bin/python scripts/v4/replay_subset_selection.py \
  --source outputs/v4/reference_20q_20261016_parent_first_chunks2 \
  --config configs/v4_phase2_full_independent_d120_gpu3.yaml \
  --result-limit 200 \
  --output outputs/v4/independent_selection_fix_20261021/replay_d120
```

Để dùng bản sửa trong lượt query tiếp theo, truyền explicit một trong hai config independent mới. Config A0 cũ tiếp tục có semantics `with_chunks` để làm đối chứng; không dùng nó rồi kỳ vọng đã tách ngân sách. Chưa mở quality pilot/full inference, chưa chọn d120 hay d200 là arm thắng, chưa có feedback/score mới. Cần đo quality trên full corpus trước khi chốt policy contest; sửa correctness này không bảo đảm tăng precision/F2.

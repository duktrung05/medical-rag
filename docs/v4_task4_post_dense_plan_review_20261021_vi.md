# Review plan Task 4 sau dense — 21/10/2026

Phạm vi: đối chiếu kế hoạch một GPU và kế hoạch nghiệm thu với code hiện tại; không chạy bước pipeline mới. Quyết định thực thi hiện hành của người dùng: **để dense hoàn tất, sau đó dừng; review plan trước khi làm tiếp**. Review này không đề xuất chuyển sang hai GPU cho lượt dense đang chạy.

## 1. Xác nhận điểm dừng thực tế

Lúc kiểm tra khoảng 11:08, tmux socket `medrag-task4` chỉ có session `dense_wait_gpu3_20261020`. Worker dense PID `937235` vẫn chạy dưới waiter; không thấy process assemble/full inference/verifier Task 4 trong danh sách tiến trình đã kiểm.

`scripts/v4/wait_for_gpu3_dense_resume.sh` gọi launcher → builder dense rồi `exit "$result"`; chỉ retry khi launcher trả 75 do GPU busy. Không có lệnh assemble hoặc inference ở phía sau. `build_release_dense.py` hoàn tất vector/mapping, tính hashes, publish manifest, rename output rồi return. Đây là phần kết thúc có sẵn của job đang được phép chạy, không phải tự mở một bước nghiệm thu mới.

Do đó chưa cần sửa script hoặc dừng worker để đáp ứng chỉ đạo. Tại điểm dừng được phép có dense manifest `complete`; **chưa được suy ra `DENSE_FULL_PASS` độc lập, `FULL_INDEX_PASS`, `FULL_INFERENCE_PASS` hoặc `FULL_V4_EXPORT_PASS`**. Không xếp verifier/queries/export chạy nền sau dense.

## 2. Những điểm cần sửa hoặc bổ sung trong plan trước khi chạy tiếp

### R1 — Cao: phân biệt dense publish với dense nghiệm thu độc lập

Plan P3 yêu cầu audit toàn mapping, input hashes, finite/norm và integrity. Builder kiểm input khi xử lý từng batch, kiểm identity cuối phần skip khi resume, kiểm count/sequence khi complete rồi tính hashes. `DiskDense.complete()` không quét lại toàn ma trận hoặc join toàn mapping với sparse; builder cũng chưa tạo `dense_full_integrity.json` như tên output trong plan.

Bằng chứng: `scripts/v4/build_release_dense.py:114`, `scripts/v4/build_release_dense.py:128`, `src/medrag/retrieval/disk_dense.py` hàm `complete()`.

Điều chỉnh: sau khi người dùng cho làm tiếp, thêm verifier CPU chỉ đọc riêng, không sửa các file đã khóa của builder đang chạy. Verifier cần nối đúng contract/fingerprint/sparse hash; kiểm shape `(2951224, 1024)`, dtype FP32, finite và norm theo block; SQLite integrity/count/sequence/uniqueness; toàn bộ `(row_id, official_id, passage_id, input_sha256)` so với eligible input theo đúng ordering; artifact hashes. Chỉ có report này PASS mới ghi `DENSE_FULL_PASS`. Publish trước audit phải được mô tả rõ trong plan vì đó là thứ tự code hiện tại.

### R2 — Cao: assemble thành công chưa tự chứng minh FULL_INDEX_PASS

Assembler đối chiếu manifest identities/model contracts và SHA-256, tạo symlinks, ghi combined manifest. Nó không chạy toàn bộ coverage join của dense mapping, canary hoặc smoke query, và chưa sinh `full_integrity.json`/`full_coverage.json`/`full_query_preflight.json` như P4 liệt kê.

Bằng chứng: `scripts/v4/assemble_disk_index.py:20`, `scripts/v4/assemble_disk_index.py:44`.

Điều chỉnh: chỉ assemble từ dense đã nghiệm thu; verifier combined kiểm resolved symlink targets và hash nối đúng hai source manifests, counts/coverage, mapping và các report sparse còn hợp lệ. Khóa trước tập smoke query IDs và tiêu chí pass: không lỗi, đủ trace, scores hữu hạn, grounding/selection hợp lệ, CPU replay khớp. Smoke là kiểm vận hành trên corpus đầy đủ, không phải bằng chứng relevance hoặc F2. Dùng một status chính `FULL_INDEX_PASS`; nếu giữ alias `G5_FULL_INDEX_PASS` thì ghi mapping rõ.

### R3 — Cao: runner hiện gộp P5 và P6, không có điểm dừng inference riêng

Plan chia inference và export thành hai bước. Trong code, `run_full_baseline.py` chạy 1.200 query rồi tự replay, validate grounding, ghi predictions/ZIP, query status và manifest, sau đó publish. CLI chưa có `--inference-only` hoặc `--export-only`.

Bằng chứng: `scripts/v4/run_full_baseline.py:149`, `scripts/v4/run_full_baseline.py:173`, `scripts/v4/run_full_baseline.py:184`.

Điều chỉnh đề xuất trước khi mở inference: tách entrypoint/mode inference và replay/validate/export. Inference ghi query results/cache/status và dừng; export CPU chỉ đọc checkpoint đã khóa, không gọi model. Khóa code/config trước khi bắt đầu lượt inference mới để không làm resume mismatch. Nếu giữ runner liền mạch, plan phải nói rõ một lệnh mở cả inference lẫn export; không mô tả hai điểm dừng chưa tồn tại. Hiện chưa thực hiện thay đổi này.

### R4 — Trung bình: trạng thái, retry và resume inference chưa đủ bằng chứng

Runner lưu checkpoint sau mỗi query complete; query thất bại nằm trong list RAM và `last_failed_attempt.json` được ghi sau vòng lặp. `query_status.parquet` chỉ xuất ở cuối sau export. Không có ledger bền vững theo attempt cho running/failed/pending ngay trong lượt chạy. Resume skip complete checkpoints, nhưng chưa kiểm số pairs/cache coverage của checkpoint đó trước khi skip.

Tests V4 đã có report 40/40 PASS, nhưng `test_full_runner_artifacts.py` hiện kiểm atomic output helper và từ chối pilot index; report resume GPU đã có chỉ áp dụng dense pilot 12.347 rows. Hai bằng chứng này không tự chứng minh inference resume và finalization recovery.

Bằng chứng: `scripts/v4/run_full_baseline.py:122`, `scripts/v4/run_full_baseline.py:143`, `scripts/v4/run_full_baseline.py:199`, `tests/v4/test_full_runner_artifacts.py`, `outputs/v4/task4/completion_20261020/reports/dense_resume_test_20261020.json`.

Điều chỉnh: ledger atomic theo query/attempt, timestamp/error, phân biệt pending/running/failed/complete; retry có giới hạn, lỗi model/CUDA không âm thầm thành prediction rỗng. Thử crash sau cache commit trước query checkpoint, sau query checkpoint, và giữa export/publish trên fixtures nhỏ. Resume phải không mất/trùng query, không gọi lại query complete hợp lệ và final canonical payload phải khớp lượt liên tục. Không cần chạy thử 1.200 queries để kiểm cơ chế này.

### R5 — Trung bình: selection replay chưa phải đối chiếu toàn score cache

Replay hiện gọi selector trên `ranked_candidates` đã lưu và so với selected checkpoint. Điều này kiểm selection logic, không chứng minh mọi reranked pair có cache entry đúng exact query/payload/input hash và score trùng trace. Cache đã khóa contract và kiểm finite cho scores mới; pipeline cũng kiểm finite cho scores trả về, nên không kết luận cache hoàn toàn thiếu bảo vệ.

Bằng chứng: `scripts/v4/run_full_baseline.py:157`, `src/medrag/ranking/score_cache.py:48`, `src/medrag/core/pipeline.py` phần score cache và kiểm `math.isfinite`.

Điều chỉnh: audit mỗi pair `reranked` ở trace với exact cache key/query hash/input hash/identity và finite score; đối chiếu pair counts, không chỉ selected evidence. Sau đó replay selection và canonical payload từ các dữ liệu đã xác minh. Ghi hai report/two checks riêng: score-cache audit và selection replay; không gọi đây là chạy lại model.

### R6 — Trung bình: report gates chưa được entrypoint inference cưỡng chế

`run_full_baseline.py` kiểm scope/counts của manifest, input/config locks và khởi tạo `DiskPipeline` để kiểm file hashes. Nó chưa yêu cầu report `DENSE_FULL_PASS`/`FULL_INDEX_PASS` hay hash của các report đó trước khi bắt đầu. Vì vậy một index assembled nhưng chưa qua smoke/coverage có thể đi vào runner nếu gọi trực tiếp.

Điều chỉnh: thêm preflight rõ ràng cho run mới, khóa report hashes vào run lock hoặc có orchestrator kiểm bắt buộc; report FAIL/missing/mismatch phải chặn inference. Chỉ bổ sung sau review, không sửa code đang chạy.

## 3. Phần của plan nên giữ

- A0, revisions, max lengths, input format, candidate budgets, selection và eligible release được khóa; không đưa tuning vào cùng lượt nghiệm thu.
- Giữ sparse reports đã PASS và kiểm lại chuỗi hashes khi tiêu thụ; không rebuild sparse hoặc rerun pilot không có lý do.
- Grounding validator hiện đã kiểm source/passage identity, offsets, text slice, URL/hash qua `verify()`; ZIP validator đã kiểm schema/query coverage/caps/CRC và equality. Giữ cơ chế này và bổ sung reports/tests còn thiếu, không thay bằng kiểm thủ công sơ sài.
- P4 benchmark trên full corpus là cần thiết: exact dense search hiện quét memmap bằng CPU, nên không dùng pilot nhỏ để hứa ETA cho 1.200 queries.
- `FULL_V4_EXPORT_PASS / SCORE_PENDING` và việc nộp BTC tách riêng vẫn đúng; F2 chỉ nhận từ feedback/score đúng artifact hash.

## 4. Thứ tự đề xuất sau khi người dùng cho làm tiếp

| Bước | Đầu ra/điều kiện |
|---|---|
| A. Review và chuẩn bị | Chốt findings; bổ sung verifier, preflight, mode/ledger và kiểm thử liên quan trước khi mở full inference |
| B. Dense nghiệm thu CPU | Dense published + report `DENSE_FULL_PASS`; không đổi output đã publish |
| C. Combined index | Assemble, integrity/coverage, smoke và benchmark representative full queries; report `FULL_INDEX_PASS` |
| D. Inference | Đủ exact 1.200 official IDs, complete=1.200, failed/pending/running=0; status/cache/trace hợp lệ; `FULL_INFERENCE_PASS` |
| E. Replay và export | Score audit + selection replay + grounding/schema/ZIP/hash PASS; `FULL_V4_EXPORT_PASS / SCORE_PENDING` |
| F. Score | Chỉ nộp khi có yêu cầu riêng; receipt/feedback nối đúng ZIP SHA |

**Hiện chưa mở bước A triển khai code hoặc các bước B–F.** Trong lượt review này chỉ đọc code/process/artifact nhỏ và cập nhật tài liệu; không chạy verifier full, assemble, query/model, export, upload hoặc tests pipeline. Các findings về thiếu checks là kết quả đọc code, không phải báo cáo một lượt full chạy FAIL.

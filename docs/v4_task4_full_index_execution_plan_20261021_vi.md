# Plan nghiệm thu dense và full index Task 4

> **Đã thực hiện ngày 21/10/2026 đến `FULL_INDEX_PASS`.** Xem [báo cáo thực thi](v4_task4_full_index_execution_result_20261021_vi.md). Nội dung dưới đây giữ nguyên thiết kế trước lượt chạy để đối chiếu; các câu “chưa thực thi” phản ánh thời điểm lập plan.

Ngày lập: 21/10/2026, giờ Việt Nam. Trạng thái **PLAN_ONLY — chưa thực thi**. Phạm vi kết thúc tại `FULL_INDEX_PASS`; không chạy inference 1.200 queries, export hoặc nộp BTC. Yêu cầu hiện hành vẫn là dừng sau dense để review; việc tạo tài liệu này không mở chạy verifier, assemble hoặc GPU smoke.

## 1. Mục tiêu và hai mốc bàn giao

1. **Mốc CPU:** nghiệm thu dense độc lập, ghép sparse + dense và kiểm toàn dữ liệu full index. Report tổng ghi `FULL_INDEX_ARTIFACTS_PASS` (tên gate bổ sung của plan này), không cần GPU.
2. **Mốc GPU:** chạy một tập nhỏ official queries trên full corpus với A0 và GPU 3, kiểm trace/cache/replay/grounding cùng resource report. Khi cả mốc CPU và smoke đạt mới ghi `FULL_INDEX_PASS`.

Manifest index có `status=complete` mô tả artifact đã publish; không thay thế hai gate trên. `G5_FULL_INDEX_PASS` trong tài liệu cũ là alias của `FULL_INDEX_PASS`, không phải gate thứ ba.

## 2. Điểm xuất phát và input cố định

Lúc đọc metadata khoảng 15:43:

| Thành phần | Hiện trạng | Đường dẫn từ project root |
|---|---|---|
| Sparse full | Manifest complete, 922 parts; reports integrity/coverage `SPARSE_FULL_PASS` | `outputs/v4/task4/sparse_full_20261016/` |
| Dense full | Publish manifest lúc 14:32:47; 2.951.224 rows, 922 parts, chiều 1.024 | `outputs/v4/task4/dense_full_gpu3_20261019/` |
| Full index | Chưa tồn tại | `outputs/v4/task4/full_index_gpu3_20261019/` |
| Full inference | Chưa tồn tại | `outputs/v4/task4/full_a0_gpu3_20261019/` |
| Config | A0, dense/reranker `cuda:0` | `configs/v4_phase2_full_a0_gpu3.yaml` |
| Báo cáo sparse | Có hai reports đã PASS | `outputs/v4/task4/completion_20261019/reports/sparse_integrity.json`, `sparse_coverage.json` |

Các SHA-256 **đã ghi nhận từ metadata/file nhỏ**, chưa phải kết quả rehash file dữ liệu lớn trong lượt lập plan:

| File | SHA-256 |
|---|---|
| Sparse manifest | `788c908cf8811730ae9085cb27712480362d0a56b5fe6d0cf2d1bbf22bcf4a03` |
| Dense manifest | `869c44e491cb15caf4e22e5ef3be08bd18a6f414bc75f2d8edcba2196b92edc9` |
| Config A0 GPU 3 | `42bb623dbb3c29ca14b0a2dab6e71aff6d78fde6e594425c073bbaa54e75933e` |
| Dense vectors, theo manifest | `d487c2bd9cf1a8402e0192940917b54e9fae4ab8cb501e91f8faf5435c7f63a6` |
| Dense mapping, theo manifest | `11581fb90f0175676e122e971c9b9a6cdf2ffc7e7d4d9b882990710040b8989f` |
| Dense contract, theo manifest | `48259438cdcde68ed723cbd6a01b1b9671ab19f85cd0ee1a98c89587589e17be` |

Expected release: **1.073.525 sources, 2.951.224 passages, 922 parts, 6 excluded docs/49 excluded passages**. Corpus version `official-71be0e1239a19dfd7a717e2bc13e3bec21bd0334`. Dense fingerprint `66ad1de7ea8aa90ad2626188fe9565c5f71f65dfadd8de79d34a472955c6a7e5`.

Ba implementation hashes của builder/encoder/disk dense hiện vẫn khớp dense contract khi đọc file nhỏ. Điều này không chứng minh mọi module runtime khác cũng đã khóa; bước I0 phải inventory cả code runtime và worktree dirty/untracked.

## 3. Bảng thực thi và điều kiện chuyển bước

| Bước | Tài nguyên | Điều kiện vào | Kết quả bắt buộc |
|---|---|---|---|
| I0. Chốt inputs, code, tài nguyên | CPU nhẹ | Được yêu cầu triển khai phạm vi tương ứng | Run lock, inventory và tests của tooling mới PASS |
| I1. Dense nghiệm thu độc lập | CPU + I/O | I0 PASS, không còn writer | `DENSE_FULL_PASS` |
| I2. Assemble và kiểm combined | CPU + I/O | I1 và sparse evidence hợp lệ | `FULL_INDEX_ARTIFACTS_PASS` |
| I3. Full-corpus smoke A0 | CPU + đúng một GPU | I2 PASS, UUID GPU 3 rảnh | `FULL_QUERY_SMOKE_PASS` |
| I4. Khóa report bàn giao | CPU nhẹ | I2 + I3 PASS, hashes còn khớp | `FULL_INDEX_PASS`, sau đó dừng |

Nếu chỉ thực hiện phạm vi CPU thì kết thúc ở I2, báo **smoke chưa chạy**, không ghi `FULL_INDEX_PASS`. Không có thao tác tự động mở inference sau I4.

## 4. I0 — Chuẩn bị không làm thay đổi release

Report root mới: `outputs/v4/task4/full_index_acceptance_20261021/`. Mỗi lần thử có `attempt_<UTC timestamp>/` riêng; không ghi đè reports PASS/FAIL cũ. Report JSON ghi atomically, flush/fsync rồi rename; chỉ publish report cuối khi đủ checks. Có progress nhỏ cập nhật mỗi 60 giây hoặc xong một shard/block group, gồm stage, PID, heartbeat, rows, bytes và elapsed.

Run lock ghi paths đã resolve, hashes manifests/config/exclusions/Phase 1–2 release và official query file; code mới cùng modules dùng thực tế; dependency versions, model revisions/cache identity; effective A0 policy, expected counts và resources. Cấu hình `document_selection` hiện lấy mặc định `with_chunks`: phải ghi rõ giá trị resolve này, không tự chuyển sang `independent` hoặc arm A1/A2 đang có trong code.

Khóa độc quyền theo run/output trong suốt thao tác. Kiểm không có writer trên source/target; mở source SQLite `mode=ro`, không dùng `immutable=1` để bỏ qua journal/locking. Nếu phát hiện hot journal hoặc source đổi hash thì dừng, giữ chẩn đoán; recovery/rebuild là việc riêng, không sửa release trong verifier.

Tooling cần tạo khi triển khai, hiện **chưa tồn tại theo plan này**:

- `scripts/v4/verify_release_dense.py`: verifier CPU, không khởi tạo encoder/model/CUDA.
- `scripts/v4/verify_full_index.py`: verifier combined và report gate CPU.
- `scripts/v4/smoke_full_index.py`: runner giới hạn đúng tập smoke, tách hoàn toàn `run_full_baseline.py`.
- Các tests tương ứng cho dữ liệu lỗi, lock, bounded-memory và stop conditions.

Giữ nguyên file artifacts đã publish. Tooling mới có thể gọi modules hiện có nhưng phải khóa hashes và tránh constructor có side effect model loading trong phần CPU.

Tests fixtures nhỏ cần bao phủ: hash/contract sai; thiếu/đảo row; duplicate passage; parent/input hash sai; NaN/Inf/norm/dtype/shape sai; exclusion xuất hiện; SQLite hỏng; symlink trỏ nhầm; report stale; smoke fail hoặc GPU busy không được PASS; dừng sau smoke không tạo full-run/export. Chạy tests liên quan sau khi có code; không yêu cầu chạy lại toàn bộ pipeline hoặc pilot cũ không liên quan.

## 5. I1 — Nghiệm thu dense bằng CPU

### 5.1. Chuỗi contract và file integrity

1. Đối chiếu đúng formats/status, 922 parts và expected counts; `contract.json` phải bằng object contract trong dense manifest.
2. Tái tính dense fingerprint bằng đúng canonicalization của builder hiện tại: SHA-256 của `json.dumps(contract, sort_keys=True).encode()`; so với dense manifest và mapping metadata. Không đổi cách serialize khi kiểm fingerprint lịch sử.
3. Đối chiếu sparse fingerprint/manifest hash, corpus version, Phase 2 release hash trong sparse lock, semantic exclusions hash, model/revision/lengths/input format/dtype/normalization với các inputs đã khóa.
4. Rehash ba files của dense theo manifest bằng streaming; trước/sau đọc ghi nhận stat và manifest hash. Khác biệt nào cũng FAIL; không tự cập nhật expected hash từ file thực tế để hợp thức hóa lỗi.

### 5.2. Mapping và coverage đầy đủ

Mở một SQLite read transaction cho snapshot mapping. Kiểm schema, metadata complete, rows/expected_rows/dimension/fingerprint; `PRAGMA integrity_check` phải `ok`, count chính xác, row IDs liên tiếp `0..2951223`, passage ID duy nhất và parent hợp lệ.

Tạo ordered expected stream từ đúng 922 Phase 2 shard theo builder: verify SHA từng shard, loại đúng exclusions, sort `(doc_id, passage_index)` trong shard. Đọc source title và passage content từ sparse store bằng batch queries; không 2,9 triệu lần mở DB/lookup riêng lẻ.

So từng row mapping với tuple `(row_id, official_id, passage_id, SHA256(payload))`, trong đó payload là `title + '\n' + text` khi title có nội dung, còn title rỗng thì text. Đồng thời đối chiếu passage text với shard, source representation hash, parent, offsets, text slice, URL/hash/provenance theo hợp đồng đã khóa. Expected stream phải hết đúng N rows cùng lúc với mapping; không chấp nhận chỉ so row counts hoặc sample cuối batch.

Đối chiếu toàn eligible sparse passage set với mapping, không thiếu/dư/duplicate; nguồn đúng 1.073.525 eligible parents và exclusions không xuất hiện. Dùng SQLite joins/temp DB trên report directory hoặc ordered external merge, không giữ toàn identities/text trong Python dict/set. Temp store có giới hạn dung lượng và cleanup chỉ trong attempt hiện tại.

### 5.3. Vectors theo block

- `np.load(..., mmap_mode='r', allow_pickle=False)`; shape đúng `(2951224, 1024)`, dtype FP32.
- Scan tối đa 32.768 rows/block; check finite, norm và thống kê min/max/max deviation. Giữ tolerance của writer hiện tại: norm trong `[0.99, 1.01]`; không thắt/nới ngưỡng sau khi xem kết quả.
- Không `matrix.tolist()`, không load cả ma trận vào RAM, không encode lại corpus và không đổi vectors để normalize lại.

Gate `DENSE_FULL_PASS` chỉ khi tất cả mục trên PASS. Report ghi hashes, counts, coverage discrepancies bằng 0, vector stats, nguồn đối chiếu và code hashes; đây là kiểm identity/serialization/structure, không phải bằng chứng mô hình encode đúng ngữ nghĩa cho từng row hoặc relevance/F2.

## 6. I2 — Assemble và nghiệm thu combined index bằng CPU

### 6.1. Sparse evidence được tái sử dụng có điều kiện

Đọc hai sparse reports đã PASS, kiểm manifest/release-audit hashes nối đúng run lock và rehash bốn sparse artifacts khi assembler tiêu thụ. Chỉ tái sử dụng coverage/BM25 integrity lịch sử khi những bytes đã kiểm vẫn khớp hash hiện tại. Không chạy lại quét hàng trăm triệu postings nếu không có thay đổi hoặc lỗi mới.

Nếu report thiếu hoặc chuỗi hash lệch: FAIL gate, báo rõ cần xác minh sparse riêng; không dùng tên thư mục hoặc số rows thay thế bằng chứng. Report mới ghi checks nào làm mới, checks nào tái sử dụng và report hash gốc.

### 6.2. Ghép bằng assembler hiện có

Chỉ sau I1 PASS và kiểm target chưa tồn tại mới chạy assembler CPU. Lệnh minh họa dưới đây **chưa được thực thi**, không đi qua launcher GPU:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  taskset -c 4-7 nice -n 10 ionice -c 3 \
  .venv-phase2-batch/bin/python -u scripts/v4/assemble_disk_index.py \
  --sparse outputs/v4/task4/sparse_full_20261016 \
  --dense outputs/v4/task4/dense_full_gpu3_20261019 \
  --config configs/v4_phase2_full_a0_gpu3.yaml \
  --output outputs/v4/task4/full_index_gpu3_20261019
```

Affinity 4–7 phải được kiểm lại theo điều kiện CPU thực tế lúc chạy; không gọi đây là hard resource reservation. `CUDA_VISIBLE_DEVICES=''` bổ sung bảo vệ, nhưng CPU tooling còn phải tránh import/khởi tạo model và xác nhận không có CUDA context.

Assembler tạo symlinks tới sparse/dense, không copy hoặc encode lại ma trận. Nguồn phải được giữ nguyên sau assemble; full index không phải artifact độc lập có thể xóa sources. Script hiện rehash trước assemble và khi tạo combined manifest, có thể đọc dữ liệu lớn nhiều lượt; không hứa thao tác nhanh chỉ vì không cần GPU.

### 6.3. Verify combined

Verifier mới kiểm đủ required file list, exact symlink targets dưới đúng sparse/dense roots đã khóa, không dangling link; manifest source hashes, counts, parts, corpus version, dense fingerprint, model contract và six artifact hashes. Nối kết quả I1 và sparse reports bằng exact hashes để tránh lặp lại cùng phép scan toàn dữ liệu mà không có lý do.

Assembler có thể publish manifest complete trước khi verifier combined PASS. Vì vậy downstream phải yêu cầu report gate riêng; thiếu report hoặc report FAIL thì không chạy smoke/inference. Nếu verify fail, giữ output để chẩn đoán, không xóa sparse/dense và không overwrite manifest để đổi expected values.

Kết thúc I2: `full_integrity.json`, `full_coverage.json`, `cpu_acceptance.json` ghi `FULL_INDEX_ARTIFACTS_PASS` cùng exact combined manifest hash. Nếu phạm vi chỉ CPU thì dừng tại đây.

## 7. I3 — Smoke A0 giới hạn trên full corpus, cần một GPU

### 7.1. Dữ liệu và tài nguyên khóa trước khi chạy

GPU vật lý 3, UUID `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7`. Qua `run_single_gpu.py` để giữ lock, occupancy checks trước/sau CUDA init, mask đúng UUID và cả hai models trên local `cuda:0`; GPU bận thì `WAIT_GPU_BUSY`, không tự dùng GPU khác hoặc dừng foreign job.

Dùng đúng 20 query IDs reference đã tồn tại, đọc text từ official query parquet đã kiểm SHA:

```text
46, 53, 81, 92, 130, 207, 242, 380, 494, 656,
754, 771, 789, 815, 927, 934, 959, 1144, 1185, 1200
```

Nguồn danh sách: `outputs/v4/reference_20q_20261016_selection_fix/manifest.json`, SHA `78f538e1d1acb2e5608671fa6bfa4fb74909cff706bfca9e085d304a7ad51618`. Đây là sample regression cố định, không đại diện thống kê chắc chắn cho toàn bộ 1.200 queries.

Khóa IDs/text hashes/config/index/report hashes trong `smoke_lock.json`; runner từ chối IDs ngoài tập này hoặc thiếu query. `result_limit=200`, 200 rerank pairs/query tối đa; A0 `max_docs=200`, `max_chunks=60`, `chunks_per_doc=1`, `document_selection=with_chunks`. Không yêu cầu output đủ 200 docs; cap chunks khiến output docs hiệu dụng không quá 60.

### 7.2. Runner và checks

Runner smoke mới dùng `DiskPipeline` với score cache riêng trong report attempt, include ranked candidates; load models một lần, tuần tự 20 queries, không gọi `run_full_baseline.py`. Lưu checkpoint atomic sau mỗi query, diagnostics và runtime; query lỗi ghi failed và chặn gate. CUDA/OOM lỗi thì dừng, không retry vô hạn hoặc fallback CPU.

Với mỗi query, kiểm branch hits, candidate identity/score hữu hạn, rerank budget, mọi selected evidence đã reranked, exact source/offset/provenance grounding và selection caps. Audit mọi reranked pair với cache/query/payload hash/score, rồi replay selector bằng CPU và so selected/canonical representation. Query rỗng phải có diagnostics và được giải thích trước khi PASS, không tự biến lỗi thành output rỗng.

Full corpus có thêm candidates nên **không yêu cầu selected/top-200 bằng snapshot 1.000 docs**. Canary lịch sử chỉ chứng minh semantics trong phạm vi của nó; cosine tolerance `1e-5`/near-tie report giữ nguyên, không chỉnh tie contract để ép kết quả full giống subset. Các kiểm structural/grounding phải exact, không dùng tolerance cho IDs hoặc text.

Để kiểm repeatability, replay lại query 92 và 927 bằng cùng runtime/cache sau lượt chính; lưu cả trace/selected/cache-audit hashes. Sai khác chưa giải thích chặn smoke. Đây là tối đa **22 lượt retrieval trên 20 unique queries**, không phải full inference; lựa chọn 2 IDs này được khóa trước vì có lịch sử near-tie, không đổi sau khi nhìn kết quả.

Không ghi `predictions.json`/`submission.zip` ở namespace full run. Intermediate smoke JSON/hashes phục vụ kiểm tra, không được gọi submission.

### 7.3. Bẫy tooling cần tránh

- `benchmark_full_pilot.py` cố định 2 shards/12.347 passages và 50 query pilot: không chạy nguyên script này trên full index hoặc sửa expected count để bỏ qua guard.
- `compare_disk_dense_same_vectors.py` gọi `matrix.tolist()` và tạo reference RAM: chỉ dùng cho subset, không đưa full matrix 11,26 GiB vào Python lists.
- `DiskPipeline` hiện tự rehash index khi load; giữ integrity check, tách startup/hash latency khỏi query latency. Exact dense search chạy CPU/memmap, GPU chỉ phục vụ encode query và rerank trong config hiện tại.

### 7.4. Tiêu chí và đo thời gian

`FULL_QUERY_SMOKE_PASS` yêu cầu 20/20 unique queries complete, failed=0, 2 repeat checks đạt, cache/selection replay/grounding/budgets PASS và process chỉ dùng đúng UUID. Không coi GPU utilization thấp là lỗi khi worker đang search trên CPU/I/O.

Ghi load/hash time, first-query latency, từng query latency, mean/P50/P95/max theo phương pháp percentile được ghi rõ; RSS, GPU allocated/reserved/process VRAM, cache hits/misses, pair counts, truncation và I/O counters. Dùng số này làm dự toán sơ bộ cho lượt 1.200 queries nếu được yêu cầu sau này; sample 20 queries có thể thiên lệch, chưa chốt deadline hoặc tuyên bố cải thiện F2.

## 8. I4 — Bàn giao, tính bất biến và điểm dừng

`full_index_acceptance.json` ghi `FULL_INDEX_PASS` khi run lock, dense report, CPU report và smoke report cùng trỏ exact index/config/code/input hashes. Nếu manifest/config/code liên quan thay đổi trong lúc chạy, report hiện có bị stale và gate phải fail thay vì tự nối kết quả từ hai runtime.

Bàn giao paths full index, reports, run lock, checksums, smoke checkpoints/cache và resource report. Có bảng reusable/stale/failed; lịch sử sparse/pilot/near-tie được tham chiếu, không viết lại như kết quả full mới.

**Dừng sau I4.** Không gọi runner 1.200 queries, không export hoặc upload. Trước khi chạy inference sau này cần xử lý các findings R3–R6 của [review sau dense](v4_task4_post_dense_plan_review_20261021_vi.md): tách inference/export hoặc mô tả rõ phạm vi, ledger/retry/resume, score-cache audit và enforcement các report gates.

## 9. Budget và xử lý gián đoạn

- Một tác vụ quét I/O nặng tại một thời điểm; CPU affinity tối đa 4 cores theo plan hiện hành, BLAS/OMP một thread cho verifier. Tổng RSS budget 32 GiB, artifacts Task 4 budget 128 GiB; là budget giám sát nếu chưa có cgroup.
- Vector file hiện 12.088.213.632 bytes gồm NPY header. Memmap/scan block, không full-RAM copy. Trước execution inventory dung lượng toàn Task 4, temp SQLite và reports; đủ dung lượng filesystem không tự đồng nghĩa trong budget.
- Verifier bị ngắt: giữ attempt/progress là interrupted, không PASS; restart lại các checks của snapshot đã khóa hoặc resume chỉ nếu thiết kế checkpoint audit có tests. Không lấy một báo cáo nửa chừng để đóng gate.
- Assembler bị ngắt: có thể còn `.<output>.incomplete-<pid>`. Giữ để chẩn đoán, không rename thủ công thành complete. Chỉ cleanup đúng staging của run đã xác nhận không còn writer, không dùng wildcard xóa source.
- Combined đã tồn tại: nếu report/hashes hợp lệ thì reuse; nếu chưa đủ evidence thì verify tại chỗ, không overwrite. Mismatch thì dừng và báo discrepancy trước khi quyết định namespace mới.
- GPU smoke bị ngắt: giữ checkpoint/cache attempt; resume chỉ khi lock hashes khớp, chẩn đoán lỗi trước retry. Không thay model/config để tiếp tục cache cũ.
- Mọi FAIL giữ bằng chứng và ngăn gate sau; không tự rebuild/re-encode sparse/dense để che discrepancy.

## 10. Phạm vi của lần lập plan này

Chỉ đọc code, manifests/reports nhỏ, hashes các file nhỏ và metadata filesystem; chưa rehash dữ liệu lớn, chạy verifier/assemble, mở CUDA/model, smoke queries hoặc tests mới. Tên scripts/reports/gates bổ sung ở trên là thiết kế cần triển khai, không phải artifacts hay kết quả PASS đã có.

# Task 4 V4 — trạng thái 19/10/2026, 14:14 giờ Việt Nam

## GPU đang chờ

Cả bốn GPU đang có compute process. GPU vật lý 0/1 có PID 1690683/1690684 của dịch vụ diffusion; GPU 2/3 có PID 3564028/3564029, hai job `vibiomir.l5.embed` của tài khoản `quybui`. Không có worker GPU V4 Task 4 trong snapshot này.

Task 4 giữ duy nhất GPU UUID `GPU-3eb95da2-57af-94b3-2ae9-8e5d46d9c079` (vật lý 2). Launcher mới `scripts/v4/run_single_gpu.py` kiểm GPU occupancy trước khi import torch; preflight thực tế trả `WAIT_GPU_BUSY`, exit 75. Report: `outputs/v4/task4/completion_20261019/reports/gpu_preflight.json`. Không có cơ chế tự chiếm GPU khác hoặc tự khởi chạy khi card rảnh.

Config `configs/v4_phase2_full_a0_single_gpu.yaml` giữ nguyên toàn bộ A0 so với config GPU hiện tại, chỉ đổi `index_namespace`; cả dense và reranker đều `cuda:0` sau khi launcher mask đúng một UUID. Tám tests tại `tests/v4/test_single_gpu_launcher.py` PASS: GPU đang có process dù utilization bằng 0, config sai device/list, busy không import torch, lock trùng, CUDA thấy nhiều card và UUID lệch. Nhánh CUDA thực khi card rảnh còn phải kiểm tiếp.

## Sparse CPU được phục hồi và resume

Worker cũ đã kết thúc tại checkpoint 914/922: 1.072.027 sources, 2.945.999 passages, excluded passages 49. Lần resume phát hiện `sqlite3.OperationalError: attempt to write a readonly database`: `passages.sqlite` còn hot rollback journal từ transaction bị ngắt, trong khi constructor mở metadata bằng mode read-only.

Đã thực hiện khi không còn sparse writer:

1. Sao lưu nguyên `passages.sqlite` (~7,9 GiB), `passages.sqlite-journal` (~208 MiB) và checkpoint vào `outputs/v4/task4/recovery_20261019/`; kiểm kích thước bản sao khớp trước phục hồi.
2. Mở database đúng target bằng SQLite `mode=rw` để SQLite tự rollback hot journal; sau đó mở lại `mode=ro` đọc metadata thành công. Metadata trở về đúng 2.945.999 rows và fingerprint gốc `8f8245e8357137b53a1a4a927d2e6532c65aa2ed60caf495997b74ca97603eb8`. Không xóa journal thủ công hoặc sửa code/fingerprint đã khóa.
3. Resume bằng `scripts/v4/run_sparse_resume.sh` trong tmux riêng: socket `medrag-task4`, session `sparse_full_20261019`. Giữ lock sparse cũ, affinity CPU 0–3, nice 10, I/O idle, BLAS/OMP một thread. PID Python tại lúc ghi báo cáo: 3754680. Log: `outputs/v4/task4/sparse_full_20261016.resume_20261019.log`.

Sparse đang xử lý lại phần dang dở và còn 8 shard chưa checkpoint. Chưa có manifest sparse full `complete`; chưa chạy dense full hoặc full queries. Sau khi sparse publish mới chạy integrity/coverage đầy đủ.

Cập nhật 14:39: worker CPU vẫn ở PID 3754680, checkpoint 914/922. `read_bytes` tăng từ khoảng 3,27 GB lên 3,35 GB trong thời gian quan sát; tiến trình phần lớn chờ I/O trên `corpus.sqlite`, chưa ghi checkpoint shard 915. Giữ nguyên mức ưu tiên I/O idle để tránh tranh tài nguyên với các job khác trên máy.

Đã xếp thêm session tmux `sparse_verify_20261019` chạy `scripts/v4/verify_release_sparse.py` sau khi worker publish manifest. Verifier chạy trên CPU 0–3, nice 10, I/O idle; kiểm hash bốn file theo manifest, SQLite quick check/metadata/rows, coverage parent, BM25 stats và multiset IDs giữa corpus và BM25. Báo cáo sẽ nằm ở `outputs/v4/task4/completion_20261019/reports/sparse_integrity.json` và `sparse_coverage.json`; log ở cùng thư mục. Chỉ coi P1 PASS khi report ghi `SPARSE_FULL_PASS`.

## Việc tiếp theo

- P1: để sparse CPU hoàn tất và verify index đã publish.
- P2: khi GPU 2 được trả lại, preflight đúng UUID rồi hoàn tất canary near-tie, pilot A0 và resource gate trên một GPU.
- P3–P6: dense full → assemble/verify full index → 1.200 queries/cache/replay → validate/export ZIP theo `docs/v4_task4_completion_single_gpu_plan_20261019_vi.md`.

Hiện chưa đạt `FULL_INDEX_PASS`, `FULL_INFERENCE_PASS` hoặc `FULL_V4_EXPORT_PASS`; F2 vẫn `UNMEASURED`.

## Cập nhật GPU 3, 15:45

Theo yêu cầu mới, Task 4 chuyển sang **GPU vật lý 3**, UUID `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7`. GPU 2 hiện có job của `quybui`; canary GPU 2 của Task 4 đã được dừng trước khi cấp phát VRAM. Launcher được khóa sang UUID GPU 3, có thêm kiểm occupancy sau khi khởi tạo CUDA để giảm khoảng trống giữa preflight và lúc chạy worker. Tám tests launcher PASS. `configs/v4_phase2_full_a0_gpu3.yaml` giữ nguyên model revisions, retrieval/selection A0 và `cuda:0` bên trong mask; chỉ namespace output đổi sang GPU 3.

Canary same-vectors đã khởi chạy trong tmux `gpu3_canary_20261019`, PID 513798 tại lúc ghi báo cáo, chỉ xuất hiện trên GPU 3. Log: `outputs/v4/task4/completion_20261019/reports/gpu3_canary.log`; report dự kiến: `canary_same_vectors_gpu3.json`. Đây chưa phải canary PASS. Sparse full vẫn checkpoint 914/922 và chưa có manifest complete, nên dense toàn corpus chưa thể bắt đầu.

Cập nhật 15:50: canary **same FP32 vectors/query vectors** đã PASS 20/20 query trên GPU 3, không đổi thứ tự top-200, score delta tối đa `2.886579864025407e-15` (`canary_same_vectors_gpu3.json`). Canary encode độc lập cũng hoàn tất: top-200 set vẫn giống nhau nhưng query 92 đổi thứ tự hạng 176–177 và query 927 đổi hạng 195–196; component delta tối đa `4.8335641622543335e-06` (`gpu3_neartie.log`). Chưa đóng gate near-tie chỉ dựa trên set giống nhau.

Dense pilot 2 shard/12.347 passages đã khởi chạy trên GPU 3 với `scripts/v4/build_release_dense.py`, output `outputs/v4/task4/dense_pilot_gpu3_20261019`, tmux `gpu3_dense_pilot_20261019`; GPU 3 có PID 568313 lúc kiểm. Đây là pilot để đo throughput/VRAM và khả năng resume, chưa phải encode full release. Sparse full vẫn ở 914/922.

Cập nhật 15:57: dense pilot GPU 3 đã publish manifest `complete` với 12.347 rows, 1.024 dimensions, 2/2 shard; combined pilot `outputs/v4/task4/combined_pilot_gpu3_20261019` đã ghép và kiểm hash thành công. Benchmark A0 đủ dense + reranker đang chạy 50 official query IDs 1–50 trong tmux `gpu3_a0_pilot_20261019`; report đích `outputs/v4/task4/completion_20261019/reports/pilot_gpu3_a0_50q.json`. Đã thêm và PASS test thứ 9 cho trường hợp job GPU khác đến trong lúc launcher khởi tạo CUDA. Chưa có report benchmark hoàn tất và sparse full vẫn checkpoint 914/922.

Cập nhật 16:13: benchmark A0 GPU 3 đã complete 50/50 queries: mean 11,41 s, P50 1,18 s, P95 53,31 s, tối đa 102,24 s; 9.999 rerank pairs, 93 pairs vượt 512 tokens trước truncation; torch peak allocated khoảng 3,53 GiB, reserved khoảng 3,76 GiB. Đây là pilot 12.347 passages, không dùng làm ETA cho full search 2.951.224 passages.

Sparse vẫn ở checkpoint 914/922 vì đang rollback phần ghi dở của shard 915. Corpus SQLite metadata đã về đúng 1.072.027 sources/2.945.999 passages; passage BM25 cũng đúng 2.945.999 rows; document BM25 còn 1.072.226 rows, tức 199 dòng vượt checkpoint. `truncate_to_rows` phải tìm/xóa postings cho 199 dòng này. `EXPLAIN QUERY PLAN` với `row_id>=1072027` cho cả GROUP BY và DELETE đều là `SCAN postings` trên DB khoảng 5,3 GiB. Worker PID 3754680 vẫn sống, `read_bytes` tăng lên khoảng 5,09 GB lúc 16:13, nhưng phần lớn ở trạng thái chờ I/O; log chưa có `part 915/922`. Đã thử duy nhất PID này ở I/O best-effort level 7 trong 45 giây, tốc độ đọc không cải thiện rõ (~29 MB), nên trả về `idle`. Không chạy worker thứ hai, không sửa file nằm trong sparse implementation hash hoặc xóa staging.

Kiểm riêng 199 dòng dư lúc sau: row IDs liên tiếp `1072027..1072225`, 199 official IDs không trùng, **đúng bằng toàn bộ 199 PASS IDs** của `part-00914` (shard thứ 915). Không ID nào nằm ngoài shard hoặc còn trong corpus store đã rollback. Đây là trạng thái ghi dở do lần worker trước bị ngắt giữa ba SQLite commits, không phải 199 document sai dữ liệu. Worker hiện tại vẫn đang đọc `documents.sqlite` để rollback BM25; xóa trực tiếp `docs` lúc worker hoạt động sẽ bỏ sót postings/DF/metadata và làm hỏng index. Chỉ chấp nhận sau khi worker ghi `part 915/922` và checkpoint tăng lên 915.

## Cập nhật phục hồi và giám sát, 16:41

Worker PID 3754680 vẫn chạy; checkpoint 914/922 là mốc commit cuối, không phải dấu hiệu dừng. Từ khoảng 16:35, `documents.sqlite-journal` xuất hiện và tăng kích thước, `write_bytes` của worker tăng: SQLite đã chuyển sang transaction ghi để rollback document BM25. Đến 16:41, `read_bytes` khoảng 6,31 GB, `write_bytes` khoảng 46,6 MB, journal khoảng 19,8 MB và vẫn tăng. Ổ `/data_hdd_16t` còn khoảng 2,9 TiB; quan sát `iostat` cho thấy `/dev/md128` bận I/O. Không có bằng chứng lỗi dữ liệu, hết dung lượng hoặc worker treo.

Đã thêm monitor chỉ đọc `scripts/v4/monitor_sparse_recovery.py` trong tmux socket `medrag-task4`, session `sparse_monitor_20261019`. Nó lấy mẫu `/proc`, checkpoint và metadata journal mỗi 60 giây; report JSONL ở `outputs/v4/task4/completion_20261019/reports/sparse_recovery_monitor.jsonl`. Sau 30 phút không có thay đổi đọc/ghi, CPU ticks, journal hoặc checkpoint, monitor chỉ ghi `suspect_stall_inspect_only`; **không** dừng worker, sửa SQLite hay khởi chạy job khác. Đã thử `--once` và xác nhận session nền hoạt động, hai mẫu đầu tiếp tục tăng read/write/journal. Session `sparse_verify_20261019` vẫn chờ sparse publish để chạy integrity/coverage.

Điều kiện chuyển bước: log `part 915/922` và `next_part=915` trước; sau đó `922/922`, manifest sparse `complete` và báo cáo `SPARSE_FULL_PASS`. Pilot A0 50 query đã hoàn tất, nhưng gate near-tie và các audit pilot còn thiếu phải được chốt trước khi mở dense full trên đúng GPU vật lý 3. Nếu worker thoát trước manifest, đọc log và checkpoint trước khi quyết định resume; không xóa 199 dòng thủ công hoặc sửa implementation lock giữa chừng.

## Cập nhật kiểm tra, 23:36

Sparse đã publish manifest `complete`, đủ 922/922 shard, 1.073.525 sources và 2.951.224 passages. Kiểm độc lập xác nhận SHA-256 của bốn file khớp manifest và `corpus.sqlite` qua metadata/count/parent/`quick_check`. Đối chiếu toàn bộ 1.073.525 source IDs và 2.951.224 cặp source–passage giữa corpus và BM25 PASS; tổng độ dài BM25 khớp metadata, row IDs liên tiếp đúng số lượng, postings và vocabulary đều có dữ liệu. Báo cáo: `outputs/v4/task4/completion_20261019/reports/sparse_coverage_fast_20261019.json` và `sparse_bm25_stats_20261019.json`.

Verifier nền `verify_release_sparse.py` đã hoàn tất sau lượt quét I/O dài: `quick_check`, số dòng/thống kê của hai BM25 và multiset ID giữa corpus/BM25 đều PASS. `sparse_integrity.json` và `sparse_coverage.json` cùng ghi `SPARSE_FULL_PASS`; P1 PASS. Lượt verifier độc lập trùng đã dừng sau khi corpus PASS để tránh tranh tài nguyên.

Phân tích near-tie GPU 3: cùng FP32 vectors/query vectors khớp thứ tự top-200 trên 20/20 query. Encode độc lập chỉ đảo hai cặp hạng kề nhau: query 92 tại 176–177 và query 927 tại 195–196; khoảng cách cosine của các cặp khoảng 1e-8 đến 1e-7, nằm trong tolerance 1e-5 đã ghi trước; tập top-200 giữ nguyên. So sánh end-to-end canary trước đó có selected payload giống nhau trên 20/20 query. Báo cáo `near_tie_analysis_20261019.json` ghi rõ strict independent-encode order vẫn mismatch; không sửa score hoặc tie contract để ép PASS.

## Cập nhật 20/10: P2 hoàn tất, dense full đã khởi chạy

- P1 giữ trạng thái `SPARSE_FULL_PASS`: sparse manifest 922/922, SHA-256, SQLite, BM25 rows/stats và coverage đã được xác minh.
- P2 đạt `G4_PILOT_PASS`. Bộ `tests/v4` có 40/40 pass; audit đúng hai tokenizer trên 12 trường hợp pilot kiểm tra passage cuối, title rỗng, source offsets/provenance, dense query, title+passage và reranker pair. Audit ghi nhận cấu hình baseline 512 tokens; benchmark A0 trước đó thấy 93/9.999 pairs bị cắt. Báo cáo không diễn giải điều này thành chỉ đọc hai passage đầu.
- Resume test trên GPU 3 bị ngắt tại 1.664 rows giữa các batch và tại mốc 6.066 rows giữa hai shard; sau resume đủ 12.347 rows. `vectors.npy`, `mapping.sqlite`, `contract.json` trùng SHA-256 với pilot chuẩn.
- Canary same vectors giữ thứ tự chính xác 20/20 query. Hai đảo hạng độc lập ở query 92/927 được giải thích là cặp gần hòa trong cosine tolerance `1e-5`; top-200 sets và selected outputs không đổi. Strict order mismatch vẫn được lưu trong báo cáo, không chỉnh tie contract.
- Pilot đo 12.347 rows trong 289,278 giây từ preflight đến manifest. Ước lượng tuyến tính full dense là 19,21 giờ; dự trù 23,05–26,89 giờ với thêm 20–40% cho I/O. Chỉ tính dense build/publish, không gồm 1.200 query; sẽ hiệu chỉnh sau 20 shard.

Lúc 16:55 giờ Việt Nam, builder dense full đã chạy trong tmux `dense_full_gpu3_20261020`, PID 363852. Launcher xác nhận duy nhất UUID GPU 3, một CUDA device và CPU affinity 4–7. Trạng thái ban đầu `RUNNING_INPUT_HASH_VALIDATION`: đang kiểm SHA của sparse full; encode vectors chưa bắt đầu ở lần kiểm lúc 16:56. Output đích `outputs/v4/task4/dense_full_gpu3_20261019/`; dự kiến 2.951.224 vectors FP32, chiều 1.024. Theo dõi bằng log `outputs/v4/task4/completion_20261020/reports/dense_full_gpu3.log` và status JSON cùng thư mục. P3 đang chạy; P4 ghép/preflight, P5 1.200 queries, P6 export ZIP còn chờ.

Cập nhật 17:02: sau khi input hashes hoàn tất, BGE-M3 đã bắt đầu encode. Một process khác của user `ntq` (PID 421069, `eval_kg_vqa_m5.py`) xuất hiện trên GPU 3 sau preflight và dùng khoảng 23,7 GiB VRAM. Theo quy tắc giữ nguyên job ngoài Task 4, đã gửi SIGINT cho đúng worker Task 4 PID 363852; checkpoint dense còn nguyên ở 1.920/2.951.224 rows, fingerprint `66ad1de7ea8aa90ad2626188fe9565c5f71f65dfadd8de79d34a472955c6a7e5`. Không xóa output hoặc đổi code/config. Waiter `dense_wait_gpu3_20261020` đang chạy preflight chỉ đọc mỗi 60 giây; khi UUID GPU 3 rảnh, nó sẽ gọi lại launcher và resume đúng `.building` này. Lượt preflight hiện ghi `WAIT_GPU_BUSY` cho PID 421069.

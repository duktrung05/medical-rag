# Plan chuyển dense build Task 4 từ một GPU sang hai GPU

Ngày lập: 21/10/2026, giờ Việt Nam. Trạng thái: **PLAN_ONLY**; chưa dừng worker, chạy pilot GPU, đổi code thực thi hoặc di chuyển checkpoint. Phạm vi là phần encode dense P3 còn lại; P4–P6 giữ các gate acceptance hiện có.

## 1. Quyết định đề xuất và bằng chứng

Chuẩn bị một builder mới, hai worker encode độc lập và **một writer duy nhất**, kế thừa vectors đã commit sang output mới bằng migration có provenance. Chỉ chuyển sau khi pilot, kiểm thử resume và phép tính lợi ích đều PASS. Worker một GPU tiếp tục chạy trong giai đoạn chuẩn bị.

Snapshot khoảng 10:56–10:58:

- Worker hiện tại PID `937235`, parent waiter `432993`, chạy khoảng 17 giờ; đọc lại PID và command trước mọi thao tác sau này.
- Mapping đọc qua SQLite `mode=ro` ghi nhận **2.450.661/2.951.224 rows (~83,0%)**, còn 500.563 rows. Đây là số rows đã commit, có thể vượt mốc shard trong log và sẽ tăng tiếp.
- GPU 3 RTX A6000 đang chạy worker, utilization 100%, process VRAM khoảng 5.894 MiB.
- GPU 2 RTX A6000 không có compute process, VRAM 1 MiB. Đây là snapshot, không phải tài nguyên được đặt chỗ.
- GPU 0/1 có workload khác. Ổ lưu trữ còn khoảng 2,9 TiB; vẫn phải kiểm budget artifact 128 GiB và RSS tổng 32 GiB của Task 4.

| Vai trò dự kiến | UUID vật lý | CUDA bên trong worker |
|---|---|---|
| Encode worker A | `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7` (GPU 3) | `cuda:0`, chỉ thấy GPU này |
| Encode worker B | `GPU-3eb95da2-57af-94b3-2ae9-8e5d46d9c079` (GPU 2) | `cuda:0`, chỉ thấy GPU này |
| Coordinator/writer | CPU | Không khởi tạo CUDA |

Plan này thay giới hạn một GPU của plan ngày 19/10 **cho P3 sau khi triển khai**, theo yêu cầu mới. Không suy rộng thành thay đổi inference, scoring hoặc quyền dừng workload khác.

## 2. Ràng buộc rút ra từ code hiện tại

1. `scripts/v4/run_single_gpu.py` khóa UUID GPU 3 và bắt đúng một CUDA device. Chỉ đổi `CUDA_VISIBLE_DEVICES` không làm builder tự phân phối việc.
2. `scripts/v4/build_release_dense.py` khóa hash của chính nó, `disk_dense.py`, `bge_encoder.py` trong `contract.json`. Sửa các file này làm resume hiện tại bị từ chối. Giữ nguyên cả ba file và config cũ.
3. Builder gom input theo thứ tự shard → `(doc_id, passage_index)` → batch tối đa 128 rows. Encoder hiện gọi SentenceTransformer với inference batch mặc định 32; hai ranh giới này phải giữ nguyên khi chia việc để hạn chế thay đổi do batching.
4. `DiskDense.append_batch()` flush vector trước, commit mapping và `meta.rows` sau. Checkpoint thực nằm trong `dense/mapping.sqlite`; không có checkpoint JSON riêng của dense build này.
5. Hai process gọi `append_batch()` cùng một output có thể lấy cùng `start=rows` rồi ghi đè vectors trước khi SQLite báo lỗi. SQLite lock không bảo vệ toàn bộ vector + mapping transaction.
6. `DiskDense` xác minh fingerprint, nên không thể đổi fingerprint của staging cũ bằng cách sửa metadata để ép resume. Cần output/contract migration mới.
7. P5 hiện chạy query tuần tự và cache riêng; exact dense search hiện đọc memmap và tính bằng CPU. Hai GPU encode không bảo đảm P5 nhanh gấp đôi.

Nguồn đối chiếu: `scripts/v4/build_release_dense.py`, `scripts/v4/run_single_gpu.py`, `scripts/v4/wait_for_gpu3_dense_resume.sh`, `src/medrag/retrieval/disk_dense.py`, `src/medrag/retrieval/bge_encoder.py`, `scripts/v4/assemble_disk_index.py`, `scripts/v4/run_full_baseline.py`, và source SentenceTransformer trong `.venv-phase2-batch` hiện có.

## 3. Chuẩn bị khi worker hiện tại vẫn chạy

Tạo các file mới, không sửa implementation đang khóa:

- `scripts/v4/build_release_dense_parallel.py`: coordinator, import prefix, schedule encode, ghi index, resume và publish.
- `scripts/v4/launch_dense_parallel.py`: kiểm UUID, occupancy, GPU locks và output lock; quản lý vòng đời child processes.
- `tests/v4/test_dense_parallel.py`: kiểm mapping, ordering, crash/resume và migration.
- Config/run plan riêng và report tại `outputs/v4/task4/migration_2gpu_20261021/`.
- Output mới dự kiến `outputs/v4/task4/dense_full_gpu23_20261021/`, staging cùng tên thêm `.building`.

Thiết kế thực thi:

1. Coordinator tạo work units theo đúng các lời gọi encode 128 rows của builder cũ. Mỗi unit có `start_row`, `end_row`, shard, thứ tự input và digest của ordered identities/input hashes.
2. Hai worker dùng process `spawn`, mỗi process nạp model cache offline, revision và tokenizer giống cũ; mask duy nhất UUID đã giao. Không kế thừa CUDA context qua `fork`.
3. Worker chỉ trả vectors + work ID; không mở mapping/vectors output ở chế độ ghi. Coordinator kiểm digest, shape, finite, norm rồi commit theo `start_row` liên tiếp.
4. Có backpressure: tối đa 8 work units mỗi worker và reorder buffer tối đa 16 units; dừng cấp việc khi buffer đầy. Không giữ toàn corpus/input/vectors trong RAM.
5. Worker GPU lỗi: ngừng cấp việc, xử lý các kết quả hợp lệ đến ranh giới commit, đóng child còn lại; giữ staging để resume. Không tự chuyển device/CPU hoặc biến batch lỗi thành dữ liệu rỗng.
6. Resume dựa trên rows đã commit; chỉ schedule phần chưa commit. Không tự tăng số rows từ work-result tồn tại hoặc phần vector đã ghi nhưng chưa có mapping commit.

Giữ nguyên model/revision, input `title + newline + passage` (title rỗng theo adapter cũ), max length 512, normalization, output FP32/1.024 chiều, exclusions và thứ tự identity. Ghi dtype model thực tế, package versions, tokenizer, inference batch, TF32/determinism settings vào runtime report; không đồng thời tối ưu precision hoặc tăng batch trong migration.

## 4. Gate trước khi dừng full worker

### G0 — Kiểm thử logic và recovery trên thư mục tạm

- Dùng encoder giả có kết quả xác định để kiểm kết quả đến ngược thứ tự, duplicate/missing unit, digest sai và worker exception.
- Crash sau khi ghi vector nhưng trước mapping commit; resume phải ghi lại phần chưa commit và không mất/nhân đôi identity.
- Crash giữa import prefix, giữa shard và trong tail encode; lượt resume phải bằng lượt liên tục về vectors/mapping logic.
- Prefix không bị thay byte; source checkpoint chỉ đọc; contract mismatch phải fail.
- Test lock chống hai coordinator trên cùng output, GPU busy, child UUID lệch và cleanup child khi coordinator lỗi.
- Chạy tests V4 có liên quan, ghi kết quả; chưa gọi PASS trước khi chạy thực tế.

### G1 — Pilot GPU và parity

- Khi GPU 3 còn chạy full, chỉ có thể smoke/pilot GPU 2 riêng và đo baseline từ log/committed rows của worker hiện tại. Không nạp model pilot thứ hai lên GPU 3 đang bận.
- Chọn trước sample ngắn/dài, title rỗng, batch cuối shard, nhiều source và các trường hợp near-tie canary đã khóa. Có thể tái dùng artifacts pilot GPU 3 làm đối chiếu khi input và batching trùng.
- Sau khi full worker dừng và checkpoint được giữ an toàn, chạy pilot hai GPU ngắn trên output riêng để đóng gate đồng thời; nếu FAIL, resume worker cũ ngay. Không gọi pilot GPU 2 riêng là bằng chứng throughput hai GPU.
- Identity/order/coverage phải khớp tuyệt đối. Prefix tái sử dụng phải byte-identical, không encode lại.
- Với encode độc lập trên GPU 2/3, báo cáo vector deltas và kiểm canary theo cosine tolerance `1e-5` đã có; top-200 sets và selected payload phải giữ nguyên. Đảo hạng near-tie phải được liệt kê như báo cáo cũ, không đổi tie contract để ép PASS.
- Cùng vectors/query vectors vẫn phải khớp retrieval order; sai khác ngoài gate đã khóa chặn migration.
- Đo throughput **đã commit**, thời gian load, import prefix, memory, I/O và final verification; không chỉ đo utilization GPU.

### G2 — Có lợi về thời gian

Tại thời điểm quyết định, đọc lại `R = expected_rows - committed_rows`. Đo `v1` từ chênh lệch committed rows trong cửa sổ 5–10 phút và `v2` từ pilot hai GPU trên input đại diện.

```text
T_giu_1_gpu = R / v1 + finalization_1
T_chuyen_2_gpu = stop + snapshot + load/pilot + import_prefix
                 + R / v2 + finalization_2 + migration_audit
```

Đây là phép tính bảo thủ: cộng các chi phí dù một phần có thể chồng lấp. Chỉ chuyển nếu dự báo tiết kiệm ít nhất **30 phút và 20% thời gian còn lại**, có tính biên I/O. Nếu không đạt, để worker hiện tại hoàn thành; mã hai GPU có thể dùng ở lượt sau. Khoảng 83% đã hoàn tất nên mức tăng tốc toàn job bị giới hạn; không hứa nhanh gấp đôi hoặc ETA chính xác từ GPU utilization.

## 5. Cutover có kiểm soát

1. Chỉ bắt đầu khi G0 và pilot sơ bộ PASS, GPU 2 còn rảnh, có dự báo lợi ích, đủ budget và khả năng rollback. Đọc lại PID, command line, UUID, parent waiter và trạng thái output; PID trong tài liệu không được dùng mù.
2. Gửi `SIGINT` tới **đúng worker dense của Task 4** và chờ thoát; không dùng `kill -9`, kill cả tmux server hoặc dừng job khác. Không bảo đảm signal rơi đúng ranh giới shard; lấy checkpoint committed sau khi process đã thoát.
3. Kiểm parent waiter không tự launch lại. Script hiện tại thoát theo exit code của worker sau lượt thực thi, ngoại trừ code 75; vẫn phải xác nhận không còn child và GPU lock đã nhả trước bước tiếp theo.
4. Khi không còn writer, tạo bản sao nhất quán của staging cũ tại recovery directory riêng, gồm contract, mapping, vectors và SQLite journal/WAL/SHM nếu có. Chỉ copy khi nguồn đã ngừng ghi; không hardlink vì rollback phải độc lập. Ghi hashes/kích thước và kiểm snapshot.
5. Nếu SQLite có hot journal, giữ bản sao raw trước; thực hiện recovery bằng SQLite trên bản sao làm việc rồi kiểm tra. Dùng bản sao đã recovery làm nguồn import; giữ staging gốc nguyên trạng. Chuẩn bị thêm đường dẫn `.building` riêng của bản sao này cho rollback bằng builder cũ: output path không nằm trong contract hiện tại nên có thể resume bản sao với nguyên contract/fingerprint và config cũ. Không xóa journal hoặc sửa `rows` thủ công. Lỗi integrity không giải thích được thì dừng migration để chẩn đoán.
6. Chốt **K = số rows đã commit** từ metadata và kiểm count/row sequence/last identity, SQLite integrity cùng prefix vectors finite/norm. Phần `[K, N)` trong file vector được preallocate không được xem là kết quả.
7. Chạy pilot đồng thời ở G1 sau khi hai GPU rảnh, cập nhật G2. FAIL hoặc không có lợi → quay về original staging và launcher cũ.
8. Khóa hai UUID theo thứ tự cố định để tránh deadlock; all-or-nothing nếu một card bận. Sau CUDA init kiểm lại foreign process và UUID ở từng worker. Khóa Task 4 chỉ phối hợp các job của mình, không đặt chỗ GPU với user khác.
9. Tạo output mới với fingerprint/contract migration mới. Nhập `[0, K)` qua API writer từ vectors/mapping đã verify, theo đúng thứ tự và giữ nguyên FP32 bytes; **không encode lại prefix**. Chỉ resume encode tail khi import hoàn tất và prefix audit PASS.
10. Cấp work units của `[K, N)` cho GPU 3 và GPU 2. Nếu K nằm giữa batch, unit đầu chính là phần dư batch mà builder cũ sẽ encode khi resume; các unit sau giữ ranh giới cũ.

Không sửa staging, contract, fingerprint hoặc metadata gốc để nhận code mới. Không chạy hai builder gốc trỏ chung output. Không publish symlink/output chuẩn khi index migration còn `.building`.

## 6. Contract migration và publish

Contract mới cần ghi rõ nguồn prefix và implementation mới, không giả nhận toàn bộ vectors được sinh bởi builder cũ:

- Parent contract hash/fingerprint; snapshot mapping hash và hash vùng vector committed; K và expected rows.
- Sparse manifest/release hash, model/revision, input contract, lengths, normalization, dimension/dtype như cũ.
- Hash của coordinator/launcher/worker mới và modules nền tảng được dùng; runtime/settings, GPU UUIDs.
- Range `[0, K)` là `reused_committed_prefix`; `[K, N)` là `parallel_encoded_tail`.
- Row-plan digest và report hashes cho parity, resume, coverage, resources.

Tạo mapping mới bằng `DiskDense(..., create=True, fingerprint=new_fingerprint)` rồi import prefix bằng API; không sửa trực tiếp `meta.fingerprint`. Giữ các trường manifest mà `assemble_disk_index.py` đang kiểm, đồng thời ghi rõ version contract migration; kiểm tương thích assemble/reader trên pilot trước full cutover.

Trước publish yêu cầu:

- Đủ **2.951.224 rows**, chiều 1.024, FP32, finite/norm hợp lệ; không thiếu/trùng passage identity.
- Toàn mapping nối đúng sparse eligible passage IDs/parents/input hashes theo row plan; prefix `[0, K)` byte-identical với snapshot.
- SQLite integrity PASS; hashes artifacts và provenance PASS; canary/retrieval/selected gates PASS.
- Writer finalize thành công, ghi manifest rồi rename staging thành output complete; chỉ sau đó assemble sang combined index mới và verify trước P5.

Giữ output gốc làm rollback đến khi dense migration và combined index đều PASS. P5 dùng config A0 đã khóa, có thể tiếp tục một GPU; muốn chia 1.200 queries cần plan riêng về partition/cache/merge, không tự đưa vào migration này.

## 7. Rollback và tài nguyên

- Lỗi trước khi dừng: worker cũ tiếp tục bình thường.
- Lỗi sau khi dừng: dừng/đợi toàn bộ child mới, xác nhận không writer và nhả GPU locks; giữ artifacts/report lỗi riêng, rồi resume staging gốc bằng launcher một GPU, sau occupancy check. Nếu staging gốc có hot journal, dùng bản sao recovery đã kiểm tra ở bước 5 thay vì mở read-only trên database chưa recovery; truyền output path riêng phù hợp tên `.building`, giữ nguyên contract/fingerprint/config cũ.
- Lỗi sau một phần tail: có thể resume builder hai GPU theo contract mới; nếu quay lại builder cũ thì bắt đầu từ K, không nối tail mới vào fingerprint cũ.
- Nếu foreign job xuất hiện trên một card: dừng có kiểm soát phần Task 4, giữ checkpoint; không dừng foreign job hoặc âm thầm đổi GPU.
- CPU affinity/budget được chia trong tập 0–7 theo khả năng máy tại cutover, không để mỗi worker mặc định dùng toàn bộ cores. Giữ nice/I/O priority theo budget cũ, cache offline và theo dõi RSS/queue/disk throughput.
- Tính cả staging cũ, snapshot, output mới, pilot và report vào disk budget. Dung lượng vector đầy đủ mỗi bản khoảng 11,26 GiB, chưa gồm mapping; nếu vượt 128 GiB thì chưa cutover.

Deliverables khi triển khai: code/tests mới, `preflight_gpu23.json`, `migration_snapshot.json`, `migration_contract.json`, `pilot_gpu23.json`, `migration_gate.json`, `prefix_integrity.json`, `dense_full_integrity.json`, log/progress phản ánh rows committed và báo cáo rollback nếu xảy ra. Tài liệu này không phải bằng chứng các deliverables đã tồn tại hoặc các gate đã PASS.

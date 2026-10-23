# Kế hoạch hoàn tất Task 4 V4 trên một GPU của server

> **Cập nhật selection ngày 21/10:** đã triển khai arm `independent` tách document/chunk budgets, cấu hình `configs/v4_phase2_full_independent_d120_gpu3.yaml` và `configs/v4_phase2_full_independent_d200_gpu3.yaml`, cùng parent-first/200 pairs. A0 bên dưới giữ làm đối chứng lịch sử. Chưa chọn arm thắng hoặc tự mở inference; cần quality pilot/feedback. Xem [plan và nghiệm thu sửa selection](v4_independent_selection_plan_20261021_vi.md). Dense dùng nguyên config/contract cũ, không cần rebuild.

> **Cập nhật 21/10/2026:** sau khi dense hoàn tất, người dùng đã yêu cầu thực hiện [plan nghiệm thu full index](v4_task4_full_index_execution_plan_20261021_vi.md). P3/P4 đã đạt `FULL_INDEX_PASS` theo [báo cáo thực thi](v4_task4_full_index_execution_result_20261021_vi.md). Điểm dừng hiện hành: trước P5 inference 1.200 queries, export và nộp BTC. Các chỉ đạo dừng ngay sau dense trong bản review trước đây là lịch sử, đã được yêu cầu mới thay thế.

Dense đã publish ngày 21/10 lúc 14:32. Phần nghiệm thu dense và full index tách gate CPU với smoke GPU và đã hoàn tất; không tự mở P5/P6.

Ngày lập: 19/10/2026, giờ Việt Nam. Phạm vi: hoàn tất baseline A0 toàn corpus đến `FULL_V4_EXPORT_PASS / SCORE_PENDING`. Tiêu chí dữ liệu và output kế thừa `docs/v4_task4_output_acceptance_plan_20261016_vi.md`; tài liệu này chốt cách thực thi phần còn lại với **đúng một GPU vật lý**. Đây là plan, chưa phải báo cáo các gate đã PASS.

> **Cập nhật 22/10/2026:** rule một GPU cho P5 full inference đã được thay bằng [rule hai GPU cho A0 full inference](v4_task4_two_gpu_inference_rules_20261022_vi.md): hai worker theo shard, GPU3/GPU2, output/cache riêng và merge/export trên CPU. Runner hiện tại chưa hỗ trợ worker shards; `full_inference_allowed=false` vẫn giữ cho đến khi worker/coordinator và preflight mới được triển khai. Thay đổi này chỉ áp dụng cho A0 inference, không thay đổi báo cáo `FULL_INDEX_PASS` đã nghiệm thu.

Cập nhật 15:45 cùng ngày theo yêu cầu người dùng: GPU được chọn cho mọi lượt Task 4 tiếp theo là **GPU vật lý 3**, UUID `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7`; config hiện hành `configs/v4_phase2_full_a0_gpu3.yaml`, các output full mới mang hậu tố `gpu3_20261019`. Các kết quả smoke GPU 2 trước đây chỉ là bằng chứng lịch sử.

## 1. Điểm xuất phát đã kiểm tra

| Hạng mục | Bằng chứng hiện tại | Phần còn lại |
|---|---|---|
| Sparse full | Worker còn chạy; checkpoint 914/922, 1.072.027 sources, 2.945.999 passages, loại 49 passages | 8 shard, finalize SQLite, checksum và publish manifest |
| Coverage input | Audit 922 shard đã PASS; eligible 1.073.525 sources / 2.951.224 passages | Reconcile lại index đã publish với release |
| Pilot sparse/dense | Pilot 2 shard và resume đã có artifact; dense CPU 12.347 rows complete | Pilot GPU A0 đầy đủ và phép đo runtime |
| Canary | Cùng vectors/query vectors: top-200 khớp 20/20; independent encode còn đổi thứ tự ở query 92 và 927; selected payload cũ bằng nhau | Giải thích và chốt gate near-tie, kiểm lại trên runtime GPU đã khóa |
| GPU | Smoke 8 passages/pairs trên GPU 2 RTX A6000 đã PASS cho BGE-M3 và reranker | Đo khi cả hai model cùng nằm trên một GPU và dùng đủ budget A0 |
| Full dense / inference / export | Chưa có manifest full dense, full combined index hoặc submission full | Thực hiện sau các gate bên dưới |
| F2 | Chưa đo | Chỉ điền khi có feedback/scorer đúng artifact |

Snapshot tài nguyên lúc lập plan: GPU 0/1 đang có khoảng 37,7 GB VRAM sử dụng mỗi card; GPU 2/3 có khoảng 4 MiB mỗi card. Đây là quan sát tại một thời điểm, không phải đặt chỗ GPU. Sparse staging khoảng 28 GiB; filesystem còn khoảng 3 TiB.

## 2. Hợp đồng dùng một GPU

- Chọn **GPU vật lý 3**, UUID `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7`, NVIDIA RTX A6000. Dùng UUID để tránh nhầm nếu số thứ tự GPU thay đổi.
- Tất cả tiến trình GPU của Task 4 nhận `CUDA_VISIBLE_DEVICES=GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7`. Trong tiến trình, dense và reranker đều dùng **`cuda:0`**; số 0 ở đây là GPU duy nhất được cho thấy, không phải GPU vật lý 0.
- Một worker GPU tại một thời điểm: canary → pilot → encode full → inference full. Không DDP, DataParallel, nhiều device, nhiều GPU worker hoặc tự chuyển sang GPU 3.
- Bổ sung launcher dùng chung trước các lượt GPU: kiểm `torch.cuda.is_available()`, `torch.cuda.device_count() == 1`, UUID thực tế và device của hai model; fail nếu lệch. Khóa `flock` theo UUID phải được giữ suốt vòng đời worker. Khóa này chỉ phối hợp các job Task 4; vẫn phải kiểm tra tiến trình ngoài Task 4 trước khi chạy.
- Khi GPU đã có workload khác, chờ cấp lại đúng GPU này; không tự dừng workload khác. Khi mất CUDA, dừng worker và giữ checkpoint; resume trên cùng UUID sau preflight, không tự fallback CPU.
- Ghi UUID, PID/tiến trình con, model device, CUDA mask, package versions và VRAM vào resource report. Quan sát `nvidia-smi` phải xác nhận PID Task 4 chỉ dùng UUID đã chọn.

Môi trường chạy GPU dự kiến:

```text
Python: .venv-phase2-batch/bin/python
PYTHONPATH=.
HF_HOME=<project>/.cache/huggingface
HF_HUB_OFFLINE=1
CUDA_DEVICE_ORDER=PCI_BUS_ID
CUDA_VISIBLE_DEVICES=GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7
OMP_NUM_THREADS=4
MKL_NUM_THREADS=4
OPENBLAS_NUM_THREADS=4
```

Trong lúc sparse còn chạy trên affinity 0–3, worker GPU dùng affinity 4–7, `nice 10`, I/O priority idle; tổng affinity hai job tối đa 8 CPU. Theo dõi tổng RSS hai job trong budget 32 GiB và toàn artifact Task 4 trong budget 128 GiB. Các giới hạn bộ nhớ này là budget giám sát cho đến khi có cgroup thực thi; không gọi chúng là hard cap. Chừa ít nhất 20% VRAM khả dụng được ghi trước pilot. Điều chỉnh batch ở pilot rồi khóa lại; không đổi dtype hoặc thuật toán giữa full run.

Config riêng `configs/v4_phase2_full_a0_gpu3.yaml` đã được tạo: sao A0 hiện tại, dense/reranker đều `cuda:0`, index namespace GPU 3. Giữ model revisions, 512-token baseline, candidate budgets và selector A0. Dense output FP32; reranker FP16 theo adapter hiện tại. Lưu ý 128 rows/batch của builder là ranh giới commit; batch inference thực của SentenceTransformer mặc định là 32, cần ghi/đo riêng.

## 3. Các bước thực hiện và điều kiện chuyển bước

### P0 — Chuẩn bị launcher, run lock và quan sát tiến trình

1. Tạo config single-GPU và launcher theo mục 2; thêm kiểm tra từ chối mask nhiều GPU/device sai/worker trùng. Không chạy full model để kiểm guard âm tính.
2. Chụp hashes code/config/model cache và input releases. Tạo report directory `outputs/v4/task4/completion_20261019/reports/`.
3. Tiếp tục quan sát worker sparse hiện có. Nếu checkpoint không tiến, kiểm PID, trạng thái I/O, thời gian sửa SQLite, dung lượng và log; chỉ kết luận treo khi có bằng chứng. Không start worker thứ hai, không đổi ưu tiên/kill job ngoài Task 4.
4. Giữ nguyên mọi file trong `implementation_sha256` của sparse lock khi worker còn hoạt động; đặc biệt cả `retrieval/core.py`, tokenizer và các hàm grounding. Nếu sửa canary cần đụng các file này, đợi sparse publish xong. Không sửa fingerprint để ép checkpoint cũ được nạp.

Output: `run_lock.json`, `gpu_preflight.json`, `resource_budget.json`, environment/package snapshot. Pass khi một worker chỉ thấy đúng UUID và paths/hash đều xác định.

### P1 — Hoàn tất và verify sparse full (CPU)

1. Để worker hiện tại hoàn thành 922/922 shard và finalize. Nếu worker thoát, kiểm lỗi rồi resume cùng output và code/input lock; không rebuild từ đầu khi checkpoint hợp lệ.
2. Chỉ tiếp nhận `outputs/v4/task4/sparse_full_20261016/manifest.json` có `status=complete`, `scope=full_release`, `parts=922`, đúng 1.073.525 sources / 2.951.224 passages và 49 excluded passages.
3. Verify SHA-256 theo manifest sau khi publish; kiểm SQLite read-only, metadata/row counts, source–passage identities/coverage và global BM25 statistics. Không chạy integrity scan nặng trên SQLite trong khi worker đang ghi.

Output: `sparse_integrity.json`, `sparse_coverage.json`, snapshot manifest/hash. Gate `SPARSE_FULL_PASS`; đây chưa phải `FULL_INDEX_PASS` vì dense còn thiếu.

### P2 — Đóng canary và benchmark pilot A0 trên một GPU

1. Dùng locked canary 1.000 docs / 2.743 passages / 20 queries. Kiểm exact search với cùng FP32 vectors và cùng query vector trước; IDs/order và payload phải khớp, cosine theo tolerance đã khóa.
2. Với independent encode, xuất các IDs/ranks/scores đổi thứ tự, top-200 set và biên top-K; phân tích riêng query 92/927. Không dùng delta vector nhỏ hoặc selected bằng nhau để tự PASS thứ tự retrieval. Nếu cần thay tie/order contract, ghi thay đổi, khóa trước lượt kiểm mới và regenerate reference có version; giữ nguyên bằng chứng FAIL cũ. Sai khác ngoài vùng tie đã định nghĩa hoặc làm đổi selected/export vẫn chặn full.
3. Kiểm tail passage, title-only, offsets/provenance và input thực sau truncation 512 tokens cho dense/reranker. Baseline có thể truncate token; phải công bố, không được đánh đồng với bỏ mọi passage sau hai chunk đầu.
4. Dùng sparse pilot 10.000 docs / 12.347 passages, **encode lại trên cùng GPU được chọn** cho benchmark GPU; không trộn row CPU vào lượt đo này. Assemble pilot, chạy đủ 50 official queries với IDs đã khóa trong `benchmark_50q.json` và `result_limit=200`, budget A0 đầy đủ.
5. Đo khi BGE-M3 và reranker cùng được nạp: load/hash time, encode passages/s và tokens/s, stage latency mean/P50/P95, số pairs, RSS, VRAM allocated/reserved/process, disk và tốc độ đọc. Smoke cũ đo hai model riêng, chưa đáp ứng bước này.
6. Thử dừng/resume dense giữa batch và ở shard boundary, cùng một GPU/contract; so coverage, mapping và logical output với lượt liên tục. Chạy tests V4 liên quan, kiểm các trường hợp missing input/hash/device/cache và lỗi không trở thành prediction rỗng. Xử lý và ghi rõ trạng thái kiểm thử API đã từng treo.
7. Lập ETA từ pilot GPU; không ngoại suy GPU từ pilot CPU hoặc smoke 8 passages. Hiện exact dense search quét vectors bằng CPU/memmap, nên benchmark cả I/O và search, không chỉ encode/rerank GPU.

Output: `canary_parity.json`, `near_tie_analysis.json`, `tokenization_audit.json`, `pilot_gpu_a0.json`, `resume_test.json`, `tests_report.json`, `full_run_estimate.json`. Gate `G4_PILOT_PASS` khi không còn mismatch chưa giải thích, resume đúng và resource budget đạt. P0/P2 có thể tiến hành khi sparse còn chạy nếu không sửa file khóa hoặc vượt budget; P3 phải đợi cả P1 và P2 PASS.

### P3 — Encode dense toàn corpus trên GPU đã khóa

1. Dùng builder hiện có `scripts/v4/build_release_dense.py`, sparse full đã PASS và config single-GPU. Khóa code/config/precision/batch trước khi tạo output mới `outputs/v4/task4/dense_full_gpu3_20261019/`.
2. Đọc đủ 922 Phase 2 shards, verify SHA và loại đúng exclusions; input title + newline + passage, mapping giữ nguyên official ID/passage ID/input hash. Encode đủ **2.951.224 vectors**, không gate theo document-first hay lấy hai passage đầu.
3. Ghi vector trước, commit mapping sau; log rows/shard/throughput và thời gian. Resume chỉ dưới cùng contract, kiểm rows cũ và identity; không coi `.building` là index dùng để query.
4. Kiểm toàn mapping, count/sequence, unique identities, input hashes, dimension 1.024 thực đo, finite/norm và file hashes trước publish. Riêng vector FP32 khoảng 11,26 GiB, chưa tính SQLite/mapping/cache.

Output: dense `manifest.json`, `contract.json`, `dense/vectors.npy`, `dense/mapping.sqlite`, `dense_full_integrity.json`, `dense_runtime.json`. Gate `DENSE_FULL_PASS` khi complete và count khớp sparse/release.

### P4 — Ghép index và preflight full corpus

1. Dùng `scripts/v4/assemble_disk_index.py`, output mới `outputs/v4/task4/full_index_gpu3_20261019/`; sparse/dense fingerprint, corpus version, rows, parts và model/input config phải nối đúng.
2. Resolve mọi symlink trong combined index, verify checksums và counts read-only; hoàn tất coverage/grounding, global sparse stats và dense mapping audit.
3. Chạy smoke cấu hình A0 trên full index với một vài official query IDs khóa trước; benchmark representative queries trên **full** index để cập nhật ETA 1.200 queries. Không suy tốc độ full exact search từ pilot 12.347 vectors.

Output: combined manifest, `full_integrity.json`, `full_coverage.json`, `full_query_preflight.json`, ETA cập nhật. Chỉ mở P5 khi `G5_FULL_INDEX_PASS` và mọi gate trước đó PASS.

### P5 — Inference đủ 1.200 queries với cache và resume

1. Chạy `scripts/v4/run_full_baseline.py` qua cùng launcher một GPU, output `outputs/v4/task4/full_a0_gpu3_20261019/`. Cùng một tiến trình nạp dense query encoder và reranker trên `cuda:0`.
2. Khóa exact official query IDs/text/hash, config A0 và result limit 200. Trần 240.000 rerank pairs là giới hạn theo budget, không phải dự đoán số pairs thực.
3. Giữ query checkpoint và score cache bền vững; bổ sung attempts/timestamps/error/progress có cấu trúc ở những chỗ runner hiện chỉ có log cuối lượt. Query complete được skip khi resume đúng lock; query lỗi phải hiện failed, được retry có giới hạn sau chẩn đoán, không được tính complete với output rỗng.
4. Lưu branch candidates, ranked candidates/pair scores, selection và elapsed time. Thiếu score/cache sai hash/NaN hoặc model lỗi phải chặn query và export. Ghi runtime, stage loss và phân bố output.
5. Sau inference, yêu cầu đúng 1.200 unique IDs, complete=1.200, failed/pending/running=0 và mọi selected evidence có score hữu hạn. CPU selection replay từ ranked candidates/scores đã khóa phải khớp selected/canonical payload. Replay hiện có không đồng nghĩa chạy lại model; nếu audit cache toàn pairs còn thiếu thì bổ sung trước khi đóng gate.

Output: `queries/`, `scores.sqlite`, `query_status.parquet`, `selected/selected_results.jsonl`, diagnostics và runtime/replay reports. Gate `G6_FULL_INFERENCE_PASS`.

### P6 — Validate, xuất ZIP và bàn giao (CPU)

1. Kiểm toàn selected evidence với nguồn/offsets/ID/URL/provenance, coverage từ official query set và eligible store. Kiểm selected–export equality.
2. Runner tạo canonical `predictions.json` và deterministic `submission.zip` có duy nhất entry `predictions.json` ở root. Mở lại ZIP, kiểm CRC/schema/payload equality và SHA-256; dùng size cap đã khóa trong config và ghi rõ đây là cap baseline, không tự khẳng định luật BTC hiện hành.
3. Bàn giao `export/SHA256SUMS`, `validation_report.json`, `run_manifest.json`, report replay/runtime/coverage và `FULL_V4_REPORT.md`. Paths thực tế của runner được giữ; report tổng tham chiếu chúng, không cần copy lại mọi index lớn.
4. Trạng thái kết thúc kỹ thuật: **`FULL_V4_EXPORT_PASS / SCORE_PENDING`**. DOCS_F2/CHUNKS_F2/FINAL giữ `null / UNMEASURED` đến khi có receipt/scorer đúng ZIP SHA. Bước nộp BTC thực hiện theo yêu cầu nộp riêng.

## 4. Các lệnh xử lý chính sau khi đã qua launcher/preflight

Launcher `scripts/v4/run_single_gpu.py` và config GPU 3 đã được tạo. Chín tests cho cấu hình/device, busy GPU, job đến giữa lúc khởi tạo CUDA, nhánh không import torch khi busy, lock chống chạy trùng, số CUDA devices sai và UUID sai đã PASS. Preflight GPU 3 lúc 15:45 đã xác nhận đúng một CUDA device/UUID và canary đã khởi chạy; trạng thái các gate theo báo cáo thực thi riêng.

Kiểm trạng thái bằng lệnh sau; chế độ này không import torch hay tạo CUDA context:

```bash
PYTHONPATH=. .venv-phase2-batch/bin/python scripts/v4/run_single_gpu.py --preflight-only
```

Các lệnh dưới đây là phần arguments cho Python trong launcher một GPU. Khi triển khai, đặt trước chúng `PYTHONPATH=. .venv-phase2-batch/bin/python scripts/v4/run_single_gpu.py --`. Riêng assemble index là công việc CPU và có thể gọi trực tiếp bằng Python sau khi đủ artifacts.

```bash
scripts/v4/build_release_dense.py \
  --root . \
  --sparse-index outputs/v4/task4/sparse_full_20261016 \
  --config configs/v4_phase2_full_a0_gpu3.yaml \
  --output outputs/v4/task4/dense_full_gpu3_20261019

scripts/v4/assemble_disk_index.py \
  --sparse outputs/v4/task4/sparse_full_20261016 \
  --dense outputs/v4/task4/dense_full_gpu3_20261019 \
  --config configs/v4_phase2_full_a0_gpu3.yaml \
  --output outputs/v4/task4/full_index_gpu3_20261019

scripts/v4/run_full_baseline.py \
  --root . \
  --index outputs/v4/task4/full_index_gpu3_20261019 \
  --config configs/v4_phase2_full_a0_gpu3.yaml \
  --output outputs/v4/task4/full_a0_gpu3_20261019
```

## 5. Xử lý gián đoạn và tiêu chí done

- Sparse vẫn ghi SQLite: theo dõi, không restart trùng. Worker đã thoát: chẩn đoán, resume checkpoint với fingerprint gốc.
- GPU busy/mất CUDA: giữ checkpoint và chờ đúng GPU; không tăng số card hay chuyển sang full CPU.
- OOM hoặc vượt budget: dừng tại checkpoint an toàn. Tối ưu batch ở pilot và khóa contract trước full; thay precision/input/scoring runtime tạo run/contract mới, không nối vào dữ liệu cũ một cách im lặng.
- Canary hoặc integrity FAIL: sửa nguyên nhân, giữ report FAIL và chạy lại gate; không mở full inference để bỏ qua gate.
- Không đưa ra ETA giờ hoàn tất full lúc lập plan: còn I/O sparse, canary và pilot GPU cần đo. Cập nhật ETA sau P2 và sau benchmark full index P4.

Task 4 được coi là done kỹ thuật khi có full index đã verify, 1.200/1.200 queries complete, replay/grounding/export/ZIP PASS, hashes truy nguyên được và toàn bộ PID GPU của Task 4 chỉ sử dụng UUID đã khóa. Một smoke GPU PASS, một pilot complete hoặc sparse 922/922 riêng lẻ không đủ đóng Task 4.

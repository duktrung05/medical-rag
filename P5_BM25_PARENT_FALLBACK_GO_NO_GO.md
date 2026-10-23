# P5 CPU Retrieval Acceleration — BM25 Parent Fallback

Ngày: **23/10/2026, Asia/Ho_Chi_Minh (UTC+07)**.

**Quyết định: NO-GO cho migration; tiếp tục A0 hiện tại.** SQL rewrite đã pass exact old/new parity trong cùng process, nhưng strict raw-score replay với frozen fixtures lịch sử FAIL. Backend cũ cũng lệch raw scores lịch sử như backend mới. Theo stop condition của task, không chạy performance benchmark, không triển khai hoặc chuyển A0. Chưa có speedup được đo.

## Deliverables

| Artifact | Nội dung |
|---|---|
| `src/medrag/retrieval/parent_first_bm25.py` | Backend opt-in `ParentFirstDiskBM25`, version `p5-parent-first-bm25-v1` |
| `tests/v4/test_parent_first_bm25.py` | Exact float/rank/pair parity, ties, Unicode, absent/empty parents, global stats, read-only checks và benchmark gate |
| `scripts/v4/benchmark_parent_first_bm25.py` | Bounded projection/replay và synthetic benchmark, không load model |
| `outputs/v4/p5_parent_fallback_fixtures_20261023_v1/report.json` | Kết quả parity, ordered raw scores dạng `float.hex()`, query plans, provenance và giới hạn tài nguyên |
| `outputs/v4/p5_parent_fallback_fixtures_20261023_v1/parent_projection.sqlite` | SQLite fixture riêng, chỉ dùng parent-query semantics; không dùng global search |
| Báo cáo này | Gate decision, giới hạn bằng chứng, norms-sidecar design và migration proposal |

Không sửa `disk_bm25.py`, `DiskPipeline`, runner A0, tokenizer, retrieval budgets, frozen config hoặc original index. Backend mới không được tự import vào đường chạy production.

## 1. Implementation

Global `search(query, limit)` vẫn delegate nguyên `DiskBM25.search()`. Chỉ search có `parent_id` đổi SQL sang:

```sql
SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id
FROM docs d INDEXED BY docs_parent
CROSS JOIN postings p
WHERE d.official_id=? AND p.term=? AND p.row_id=d.row_id
```

`CROSS JOIN` khóa hướng traversal; `INDEXED BY docs_parent` buộc bắt đầu từ parent index. Không tạo index mới hoặc chạy `ANALYZE` trên corpus gốc.

Plan kiểm tra trên projection có schema/index giống backend gốc, SQLite 3.53.1:

```text
Old:
  SEARCH p USING PRIMARY KEY (term=?)
  SEARCH d USING INTEGER PRIMARY KEY (rowid=?)
New:
  SEARCH d USING INDEX docs_parent (official_id=?)
  SEARCH p USING PRIMARY KEY (term=? AND row_id=?)
```

Giữ nguyên global N, total_length/avgdl, DF/IDF; giữ đúng biểu thức `idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * length / max(avgdl, 1)))`, thứ tự `set(tokens(query))`, cách cộng vào từng row, sorting `(-score, official_id, str(passage_id))` và rank bắt đầu từ 1. Không sort token set, không dùng local-parent statistics và không vectorize/reassociate arithmetic.

## 2. Bounded fixture validation

Đã chạy:

```bash
PYTHONHASHSEED=0 .venv-phase2-batch/bin/python \
  -m scripts.v4.benchmark_parent_first_bm25 fixtures \
  --output outputs/v4/p5_parent_fallback_fixtures_20261023_v1 \
  --queries 53 130 --parents 4
```

`PYTHONHASHSEED=0` chỉ áp dụng cho process thử nghiệm mới; không đổi environment của A0. Đó không phải seed đã biết của frozen historical run.

Giới hạn process thử nghiệm: **một CPU (31), nice 19, 384 MiB address space, 30 giây CPU**, tối đa 25 giây theo deadline kiểm tra và file 64 MiB. SQLite nguồn luôn `mode=ro`, `query_only=ON`; cache size chỉ là setting connection-local. Wall deadline được kiểm tra qua SQLite progress handler và giữa các benchmark calls; hard CPU limit độc lập, không tuyên bố mọi Python operation có hard wall timeout.

Projection lấy **4 fallback parents đầu tiên** của mỗi query, không chọn theo kết quả. Chỉ đọc docs qua parent index, global DF bằng term key và TF bằng exact `(term,row_id)` point lookup. Có giới hạn tối đa 128 passages/parent và 128 unique terms. Không gọi global BM25 hoặc dense retrieval trên full index.

Fixture cụ thể:

- Queries: **53, 130**.
- Parents: **8**; passages: **67**; unique terms tổng: **88**.
- Postings được chép: **2.393**; source posting point lookups: **5.896**.
- N trong projected metadata vẫn là **2.951.224**; total_length và DF giữ nguyên nguồn.
- Wall time toàn harness: **3,826 giây**, gồm extraction/replay/validation. Đây không phải old/new latency comparison.

Projection lược bỏ unrelated postings nhưng giữ đầy đủ các rows và query terms của target parents, nên đủ để đối chiếu parent-filtered results với cùng global statistics. Nó không đại diện cho chi phí term-first scan trên full corpus. Không dùng projection để benchmark full-index speedup.

### Kết quả

| Kiểm tra | Kết quả | Phạm vi |
|---|---|---|
| Passage IDs, raw score bits, toàn bộ ordered candidate ranks old/new | **PASS** | 67 matched passages thuộc 8 parents, limit 129 để không chỉ kiểm top-1 |
| Selected fallback passage so với frozen fixtures | **PASS** | 8 parents |
| Full candidate objects/order old/new | **PASS** | Queries 53, 130; 248 và 272 candidates |
| Candidate objects/order so với frozen live pair shards | **PASS** | Cùng 2 queries |
| Final pair IDs/order và input SHA-256 | **PASS** | 200/query, tổng 400; so cả old/new lẫn frozen pair trace |
| Raw selected-passage scores so với frozen historical trace | **FAIL** | 8/8 scores khác 1–3 ULP; backend cũ và mới cho cùng giá trị khác historical |
| Strict frozen acceptance gate | **FAIL — STOP** | Chặn benchmark và migration |

Global branch hits và những fallback parents ngoài 8 parents đã chọn được **replay từ frozen trace**. Vì vậy kiểm tra downstream 400 pairs là kiểm tra integration của subset này, không phải xác minh SQL mới cho mọi fallback parent hoặc chạy lại full retrieval. Frozen trace chỉ lưu selected top-1 fallback score; không có historical all-passage score ranking để so toàn bộ 67 historical scores.

### Phân biệt SQL parity với historical drift

Ví dụ query 53, parent 741409:

```text
old recomputed: 0x1.d5ab6310fd71ap+4
new recomputed: 0x1.d5ab6310fd71ap+4
frozen trace:   0x1.d5ab6310fd71bp+4
```

SQL rewrite không tạo thêm sai lệch trong phép thử này: old và new có raw-score bits giống nhau. Tuy nhiên frozen raw-score reproducibility hiện chưa đạt. Một nguyên nhân phù hợp là `set(tokens(query))` có thứ tự phụ thuộc Python hash seed/process và floating-point addition không kết hợp; historical term order/seed không có trong trace. Audit này chưa chứng minh đó là nguyên nhân duy nhất của drift.

Không đổi sang sorted terms, không thử nhiều seed để chọn một seed có kết quả thuận lợi, không nới tolerance, không sửa fixture, và không bỏ raw-score gate. Những việc đó không đáp ứng yêu cầu exact frozen parity.

## 3. Regression tests

Lần kiểm tra cuối:

```bash
nice -n 19 taskset -c 31 .venv-phase2-batch/bin/python -m pytest -q \
  tests/v4/test_parent_first_bm25.py \
  tests/v4/test_disk_bm25.py \
  tests/v4/test_p5_2_split.py
```

**120 passed in 16,00s.** Không load model. Suite mới kiểm tra float hex equality thay vì `pytest.approx`, repeated/multiple terms, Unicode/NFC, score ties và passage-ID tie-break, missing parent/term, empty corpus, incomplete/wrong fingerprint, global-search delegation, index hash không đổi, parent fallback→candidate→pair/payload hash parity và benchmark từ chối gate FAIL trước khi tạo synthetic index. Suite có 32-term cases để bảo vệ accumulation semantics.

Toàn bộ locked A0 implementation hashes và frozen config vẫn khớp `a0_t4_v1_full_20261022/lock.json` sau kiểm tra. Hash backend cũ vẫn là:

```text
a96c443ed81a04515bc81048bedcf612073461c49288a67eca9b3c6007bf40d0
```

Backend thử nghiệm:

```text
7a009fd97d4a7b9948b681c02a1393a275c4256f3df88f9a41a54b09ffda6026
```

Sau lần fixture run, harness được siết thêm: exit code 2 khi strict gate FAIL, chỉ cho tạo namespace trực tiếp dưới `outputs/v4/`, kiểm tra đầy đủ provenance trước benchmark. Không thay backend/scoring. Report đã ghi được giữ nguyên; harness SHA trong report là bản chạy trước các guard này, nên bản harness cuối **cố ý không nhận report cũ làm provenance PASS**. Dù vậy report cũ đã FAIL ở raw-score gate. Không chạy lại fixtures hoặc benchmark để tìm cách vượt gate.

## 4. Benchmark harness và trạng thái phép đo

**Chưa chạy benchmark old/new; speedup parent fallback và end-to-end đều chưa đo.** Lý do là strict parity gate FAIL, không phải thiếu một slot CPU. Không tiếp tục tối ưu BM25 sau kết quả này.

Harness có mode `benchmark` dành cho cửa sổ thử nghiệm sau khi giải quyết gate:

```bash
# Chỉ là mẫu lệnh cho tương lai; report hiện tại FAIL nên sẽ bị từ chối.
PYTHONHASHSEED=0 .venv-phase2-batch/bin/python \
  -m scripts.v4.benchmark_parent_first_bm25 benchmark \
  --output outputs/v4/p5_parent_fallback_benchmark_FUTURE \
  --parity-report /path/to/new_strict_PASS_report.json \
  --rows 20000 --repeats 3
```

Nó tạo synthetic SQLite trong namespace mới, tối đa 20.000 rows, 4 query terms, 4 passages/parent, tối đa 5 repeats; warm-up cả old/new và đổi thứ tự đo giữa repeats. Không dùng original full index cho retrieval. Report lưu:

- Per-call wall latency old/new, medians và median ratio.
- SQL execute/DF/posting-join statement counts bằng trace callback.
- SQLite VM-step counts bằng progress handler, đo riêng ngoài latency vì instrumentation có overhead.
- Internal B-tree probes mới và số term-postings cũ phải duyệt, suy ra từ synthetic shape với nhãn tách biệt.
- Hash parity report, source/config/index/dependency/hash-seed provenance và resource caps.

SQL statement count có thể **không giảm**: cả hai backend vẫn có metadata SELECT, DF SELECT/term và posting join/term. Thay đổi chủ yếu là lượng rows/internal probes mỗi join; không đánh đồng SQL statement count với số B-tree lookups hoặc công việc thực tế.

Synthetic benchmark sẽ chỉ chứng minh mechanism trên fixture có kích thước cụ thể. Không dùng ratio đó làm production speedup hay end-to-end speedup. Muốn đo end-to-end cần cửa sổ riêng với stage timing và frozen-query replay phù hợp, tránh cạnh tranh A0; chưa có phép đo đó trong task này.

## 5. Optional FP64 norms sidecar — thiết kế, chưa build

Không thay `DiskDense`, không build sidecar, không scan full matrix trong task.

Sidecar tương lai có thể là read-only `norms.npy`, shape `(2951224,)`, dtype float64, khoảng **22,516 MiB** payload, ở namespace derivative mới. Dùng vectors FP32 hiện có; không re-embed, không normalize lại hoặc đổi metric.

Quy trình build dự kiến, chỉ khi có cửa sổ được phép:

```python
# Cùng block_rows/layout/NumPy với reference.
block = np.asarray(matrix[start:end], dtype=np.float64)
norms[start:end] = np.linalg.norm(block, axis=1)
```

Kernel sử dụng vẫn giữ đúng:

```python
scores = block @ query_vector / (norms[start:end] * query_norm)
```

Không chuyển thành `(block / norm) @ query`, nhân reciprocal, dùng FP32 norm hoặc gộp query thành GEMM nếu chưa chứng minh exact numerical parity; các cách đó thay thứ tự arithmetic.

Manifest derivative cần vector SHA-256/fingerprint/shape/dtype/dimension, norm payload checksum, source backend/code hash, block_rows, NumPy/BLAS version, layout và build provenance. Không chỉ tin mtime; cần xác nhận vector content identity trong cửa sổ cho phép đọc full matrix. Kiểm tra norm finite/positive, đủ rows, publish file tạm bằng fsync+rename và checksum; không sửa original index manifest. Reader fail closed nếu sidecar không khớp hoặc incomplete.

Trước khi sử dụng, cần query-vector fixtures và exact checks cho norms, toàn score arrays, local threshold/ties, ordered dense top-200/identity mapping, fused candidates, final pairs/input hashes. Dùng equality/float bits, không chỉ `allclose` hoặc overlap. Include near ties, tail block và invalid sidecar. Query embeddings chưa được lưu trong full A0 shards nên không thể chứng minh dense parity chỉ bằng downstream replay hiện có.

Sidecar bỏ repeated norm reduction nhưng vẫn phải full-scan vectors và cast FP64 để dot product. Không có benchmark nên không hứa mức tăng tốc, nhất là khi BM25 chưa có stage timing.

## 6. A0 safety và migration proposal

Runtime read-only ở đầu validation xác nhận PID **197011** vẫn chạy, 118 threads, khoảng 163% lifetime CPU; output đã có **244 shards** tại snapshot. Đó là snapshot, không phải trạng thái live sau khi báo cáo được mở.

Task không dừng/signal/restart tmux, không thay affinity/environment của A0, không load model, không chạy full 1.200-query inference hoặc full-index retrieval thứ hai. Chỉ tạo backend/test/harness/report mới và namespace fixture riêng. Không ghi đè hoặc re-envelope original shards.

**Hiện tại:** để A0 tiếp tục; không checkpoint rồi chuyển implementation theo kết quả này.

**Nếu có quyết định migration sau này:**

1. Giải quyết strict historical score reproducibility theo một contract được xác định rõ trước. Không tự nới yêu cầu parity. Đo old/new trong cùng controlled environment là cần thiết nhưng không thay thế yêu cầu khớp frozen historical scores hiện tại.
2. Sau gate PASS, benchmark có giới hạn rồi mở rộng coverage: mọi fallback parents của pinned queries, common/long queries, ties và hash/process reproducibility. So đầy đủ raw scores/ranks, selected passages, candidate ordering, final pair/input hashes; ghi rõ phạm vi đã chứng minh. Đo end-to-end mới biết có đáng chuyển run.
3. Tạo checkout/version và namespace mới, khóa backend mới, backend reference, tokenizer/core/pipeline, dependencies, seed/term-order policy và frozen config/index identities. Config frozen không cần thay. Không sửa code mà process đang dùng, không sửa lock cũ để hợp thức hóa code mới.
4. Nếu A0 kết thúc hoặc có cửa sổ chuyển đã được người dùng cho phép, snapshot danh sách existing shard IDs/hashes, bảo toàn nguyên byte originals và lineage; validate toàn bộ trước khi kế thừa. Không tự dừng A0 trong task này.
5. Stock split runner yêu cầu `lock_sha256` của shard bằng lock đang chạy và implementation hashes khớp. Vì vậy không copy lẫn shard từ lock cũ và mới rồi gọi đó là resume hợp lệ. Cần migration adapter/validator riêng có source-lock/source-shard lineage, hoặc tạo derived envelopes trong **namespace mới** sau khi xác nhận mọi query/pair/candidate/hash fields không đổi; originals vẫn nguyên vẹn.
6. Merge phải bảo đảm đủ đúng query IDs 1–1200, không trùng/mất shard, đúng 200 pairs/query và exact passage/input/cache identities; kiểm tra consistency của score contract, generate manifest mới cùng mapping mỗi derived shard về source hash/version. GPU/finalizer chỉ được nhận artifact theo một contract được validate đầy đủ. Không dùng mtime để chọn shard.
7. Giữ đường rollback về originals và runner cũ. Không chạy hai writers cho cùng output. Chỉ chuyển khi measured remaining-time saving vượt chi phí validation/migration và parity đạt yêu cầu đã khóa.

## 7. Trạng thái cuối

| Hạng mục | Trạng thái |
|---|---|
| Backend thử nghiệm | Hoàn thành, opt-in |
| Regression tests | 120 PASS |
| Exact old/new trên bounded frozen-parent projection | PASS |
| Historical frozen raw-score exact parity | FAIL ở baseline và optimization |
| Frozen selected passages/candidates/400 pair IDs/input hashes trong phạm vi đã test | PASS |
| BM25 measured speedup | Chưa đo; benchmark bị gate chặn |
| End-to-end measured speedup | Chưa đo |
| Dense norms sidecar | Chỉ thiết kế, chưa build/use |
| Migration | **NO-GO** |
| A0 hiện tại | Tiếp tục chạy, không can thiệp |

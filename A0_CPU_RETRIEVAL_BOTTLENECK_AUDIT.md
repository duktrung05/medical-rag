# A0 CPU Retrieval Bottleneck Audit

Ngày audit: **23/10/2026, khoảng 09:32–09:41, Asia/Ho_Chi_Minh (UTC+07)**.

**Quyết định đề xuất: để A0 hiện tại chạy tiếp.** Có các thao tác dư thừa đã xác nhận, đặc biệt BM25 parent fallback dùng query plan không hiệu quả và dense tính lại corpus norms. Tuy nhiên chưa có stage timing của full A0 hoặc benchmark optimization để kết luận tỷ trọng latency, cam kết speedup hay biện minh cho việc thay implementation giữa lần chạy.

## 1. Phạm vi và cách thu thập

- Audit code, frozen config, manifest/lock, metadata SQLite, NPY header, các immutable pair shards và báo cáo đã tồn tại.
- Runtime chỉ đọc: `ps`, `/proc/197011/{status,stat,io,maps,cgroup,environ}`, counters/affinity của threads, cgroup metadata; `tmux list-panes` và `capture-pane`. Chỉ đọc whitelist biến môi trường liên quan threading/device, không dump toàn bộ environment.
- Hai mẫu CPU/I/O dài khoảng 8 giây; không attach profiler, debugger hoặc tracer.
- `EXPLAIN QUERY PLAN` và `SELECT ... FROM meta` dùng SQLite URI `mode=ro`. Không thực thi truy vấn retrieval trong audit. Query plan được kiểm tra lại bằng chính interpreter `.venv-phase2-batch/bin/python`, SQLite **3.53.1**.
- Sandbox ban đầu không thấy host PID và không truy cập được tmux socket. Các lệnh runtime sau đó chạy ngoài sandbox theo cơ chế approval của công cụ, vẫn chỉ đọc. Không diễn giải việc PID không hiện trong sandbox là process đã dừng.
- Không signal/dừng/đổi affinity, environment hoặc thread settings của A0. Không chạy full-index retrieval thứ hai, không load model mới, không đọc toàn bộ vector matrix để benchmark/hash, không sửa frozen config, code, checkpoint, SQLite hoặc output shards. Artifact duy nhất được tạo là báo cáo này.

## 2. Frozen contract và trạng thái thực tế

Process được xác nhận:

```text
PID:       197011, PPID: 196900
Started:   2026-10-22 15:12:08 +07
Command:   .venv-phase2-batch/bin/python -m scripts.v4.run_p5_2_split prepare
           --mode full --config configs/v4_submission_p2_a0.yaml
           --output outputs/v4/a0_t4_v1_full_20261022
```

Tại snapshot khoảng 09:32:38: elapsed **18:20:29**, cumulative CPU time **29:52:14**, lifetime CPU **162%**, trạng thái `Rl+`, **118 threads**, main-thread affinity **0–31**. Một mẫu sau xác nhận cả **118 threads** đều có affinity `0–31`.

Frozen config hash khớp lock:

```text
bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03
```

Index manifest hash khớp lock:

```text
efdad2cf0e1db51730f7eaf0f931242e9bba9d5b0998910f6cb168f8dede6941
```

Index có **1.073.525 sources**, **2.951.224 passages**, **1.024 chiều**, dense FP32, `status=complete`. Contract A0: document BM25 400, passage BM25 200, dense 200, fused 400, RRF k=60, parent-first 200 rerank pairs, tối đa 3 passages/document khi allocation. Selection/context chạy ở finalize.

Config trên đĩa đặt dense/reranker `cuda:0`, nhưng `prepare()` tạo bản copy và đặt dense device thành `cpu` tại `scripts/v4/run_p5_2_split.py:198–200`. Reranker được truyền bằng `object()` để tạo pairs, không load reranker và không chạy reranker inference/token diagnostics ở prepare. Đây là hành vi có sẵn, không phải thay đổi của audit.

Cả **7 file implementation** liệt kê trong `lock.json` đều khớp SHA-256 hiện tại. Backend `disk_dense.py`, `disk_bm25.py`, encoder và disk store **không nằm trong danh sách implementation hashes của lock**. Mtime hiện tại của các file này đều trước khi PID khởi động; điều này hỗ trợ audit source hiện tại nhưng không phải chứng minh bytecode đã load trong process. Không được lợi dụng thiếu sót của lock để thay backend âm thầm.

### Checkpoints

Tại lần đọc khoảng 09:33:41 có **240 shards, query IDs 1–240**, tổng khoảng **248,6 MB**; chưa có `pair_manifest.json`. Runner chỉ tạo manifest sau khi hoàn thành mọi query, nên trạng thái này phù hợp với prepare đang chạy.

| Query | Shard mtime (+07) |
|---|---|
| 236 | 23/10 09:11:09.876 |
| 237 | 23/10 09:13:35.655 |
| 238 | 23/10 09:16:20.435 |
| 239 | 23/10 09:30:38.947 |
| 240 | 23/10 09:33:23.606 |

| Cửa sổ khoảng cách mtime | Mean | Median | Min–max |
|---|---:|---:|---:|
| 20 khoảng gần nhất, q220→240 | 257,34 s | 169,44 s | 88,16–858,51 s |
| 50 khoảng gần nhất, q190→240 | 258,33 s | 146,17 s | 55,74–2.139,75 s |

Shards 1–3 có từ trước PID hiện tại. Với 237 khoảng q3→240, mean là **281,33 s**, median **174,10 s**. Khoảng đầu này có thể bao gồm chi phí khởi động/resume, vì vậy cửa sổ gần nhất phù hợp hơn để tham khảo throughput. Chỉ tính 236 khoảng q4→240 thì mean là **277,19 s**, median **173,60 s**. Mtime là thời điểm publish sau fsync; khoảng cách bao gồm toàn bộ query kế tiếp, serialization/I/O và scheduling, không phải stage latency.

Nếu giữ mean 50 khoảng gần nhất, 960 queries còn lại tương ứng khoảng **68,9 giờ**. Đây chỉ là ngoại suy throughput tại snapshot, không phải ETA chắc chắn; query variability rất lớn. Không dùng median để bỏ qua long-tail queries.

## 3. Bằng chứng implementation

### 3.1 Dense: full scan và corpus norms lặp lại đã xác nhận

Nguồn: `src/medrag/retrieval/disk_dense.py:117–157`.

Mỗi `search()` thực hiện tuần tự:

1. Đọc metadata mapping, encode đúng một query trên CPU; đổi query vector sang FP64 và tính query norm.
2. Mở lại `vectors.npy` bằng read-only memmap.
3. Duyệt **toàn bộ 2.951.224 rows**, mặc định block 32.768 rows: **91 blocks/query**.
4. Mỗi block chuyển FP32 sang một array FP64 mới, rồi tính:

   ```python
   block @ query_vector / (np.linalg.norm(block, axis=1) * query_norm)
   ```

5. Giữ local top-200 có bao gồm ties, lookup identities rồi sort để lấy global top-200 theo score, official ID và passage ID.

**Corpus norm thực sự được tính lại cho mỗi query.** Index đã yêu cầu norm của embeddings lúc append trong khoảng 0,99–1,01; điều này không cho phép thay mọi norm bằng 1 nếu cần parity. Search hiện dùng actual norm của stored FP32 vectors sau khi cast FP64.

Chi phí dữ liệu suy ra từ shape/dtype, **không phải benchmark**:

- Matrix FP32: **11,258 GiB** (file 12.088.213.632 bytes gồm header).
- Mỗi query chuyển tổng cộng **22,516 GiB** dữ liệu array FP64 qua các blocks; đây là lượng dữ liệu tạo qua cả scan, không phải allocation đồng thời.
- Full block FP64: **256 MiB**. NumPy cài trong venv, `numpy/linalg/_linalg.py:2804–2805`, dùng `(x.conj() * x).real` rồi `add.reduce` để tính row norm; có thêm temporary bình phương tương đương một full block. Đây là các phép array/reduction riêng với BLAS matrix-vector multiply.
- Norm cache FP64 cho cả corpus chỉ cần **22,516 MiB** payload.
- Tối đa thông thường **18.200 local candidates** trước identity lookup/sort, có thể nhiều hơn nếu ties. Mỗi block gọi `np.partition` hai lần với cùng input/k; đây là thao tác dư thừa nhỏ hơn, chưa đo mức ảnh hưởng.

Memmap đọc lại pages của toàn matrix về mặt truy cập bộ nhớ mỗi query; **không đồng nghĩa đọc vật lý 11,258 GiB từ ổ đĩa mỗi query**. Page cache có thể phục vụ phần lớn dữ liệu. Việc mở lại memmap cũng làm mất khả năng tái sử dụng mapping trực tiếp giữa queries.

### 3.2 BM25: global scoring và parent fallback

Backend thực tế của `DiskPipeline` là `src/medrag/retrieval/disk_bm25.py`, không phải BM25 cũ trong `src/retrieval/` hay BM25 in-memory trong `retrieval/core.py`.

Global BM25 (`disk_bm25.py:113–149`) tra `df`, duyệt postings của từng unique query term, join docs bằng row ID; tạo `sqlite3.Row`, cộng score và cập nhật dict bằng Python. Sau đó **sort toàn bộ rows có score**, rồi mới lấy limit 400/200. Vì vậy không phải dense-style scan mọi vector/document, nhưng common terms có thể chạm tập postings lớn. Code có phần serial đáng kể; chưa đo riêng scoring, SQLite và sort.

**Phát hiện ưu tiên: parent fallback dùng global term-first query plan.** `Pipeline.prepare_rerank()` gọi `search_parent()` riêng cho mọi fused parent chưa có passage evidence (`pipeline.py:93–104`). Backend chạy lại tokenization, metadata/DF queries và SQL join cho mỗi parent.

Query plan hiện tại trên `passages.sqlite`, bằng interpreter của A0:

```text
SEARCH p USING PRIMARY KEY (term=?)
SEARCH d USING INTEGER PRIMARY KEY (rowid=?)
```

Điều kiện `d.official_id=?` được lọc sau lookup docs. Mặc dù có index `docs_parent(official_id)`, plan hiện tại **không dùng nó để bắt đầu từ passages của parent**. Với mỗi term có trong corpus, kế hoạch này duyệt postings toàn corpus cho term đó cho mỗi lần fallback. Đây là amplification có bằng chứng query plan, không phải khẳng định đã đo latency của fallback.

Số parents chỉ có channel `document_bm25` trong candidate shards là các trường hợp phải qua fallback theo control flow hiện tại. Đây là **số tối thiểu** số fallback calls; nếu một passage hit không qua verification, có thể còn fallback khác.

| Query | Fused parents | Parents chỉ có document channel |
|---|---:|---:|
| 1 | 316 | 75 |
| 190 | 310 | 79 |
| 220 | 246 | 60 |
| 236 | 246 | 50 |
| 237 | 293 | 69 |
| 238 | 253 | 85 |
| 239 | 306 | 98 |
| 240 | 247 | 57 |

Ví dụ query 240 có ít nhất **57 lần** parent search. Không suy từ số calls ra tỷ lệ thời gian.

Chỉ dùng `EXPLAIN`, một SQL có thể thử ở phiên bản tương lai:

```sql
SELECT p.row_id,p.tf,d.length,d.official_id,d.passage_id
FROM docs d INDEXED BY docs_parent
CROSS JOIN postings p
WHERE d.official_id=? AND p.term=? AND p.row_id=d.row_id
```

Plan thu được:

```text
SEARCH d USING INDEX docs_parent (official_id=?)
SEARCH p USING PRIMARY KEY (term=? AND row_id=?)
```

Không thực thi SQL retrieval này, không đổi SQLite, không chạy `ANALYZE` hoặc tạo index. Hướng parent-first dùng index hiện có có thể tránh amplification mà giữ cùng tập `(term,row_id,tf,length)` và BM25 global statistics. Cần kiểm tra actual outputs/latency trước khi triển khai.

### 3.3 Corpus lookups, fusion và pair preparation

- `DiskCorpusStore.source()`/`passage()` mở connection để `_check()` rồi connection lookup; `SourceLookup`/`PassageLookup` không cache (`src/medrag/data/disk_store.py`). Pipeline lặp lại source lookup để verify passage và lấy titles. Có thể đọc/deserialize full source text của cùng parent nhiều lần trong một query.
- `Source.text_sha256` là property tính SHA-256 lại mỗi lần (`core/contracts.py:18–20`); verifier gọi property đó. Đây là chi phí lặp lại có bằng chứng code, chưa đo latency.
- `fuse()` chỉ xử lý tối đa 400+200+200 input hits và sort các identities đó (`retrieval/core.py`); không có full-corpus fusion. Chưa có timing để gán tỷ trọng nhỏ/lớn chính xác.
- Sau fallback, pipeline sort parents, allocate parent-first, tạo title-newline-passage payload. Runner tính input/cache hashes, serialize khoảng 1 MB/shard và publish bằng fsync + atomic rename. Chưa có dữ liệu để kết luận fsync hoặc hashing là bottleneck.
- Reranker inference, selection và adjacent context expansion không nằm trong CPU prepare đang audit.
- Constructor `DiskPipeline` hash toàn bộ index files một lần khi khởi tạo, không lặp mỗi query. Resume lại runner phải chịu startup validation/model load; không quy chi phí này cho mỗi shard.

## 4. Runtime: có serial phases, BLAS vẫn có burst parallelism

Process đang map NumPy OpenBLAS, SciPy OpenBLAS, Torch CPU và OpenMP libraries. Không có biến threading trong whitelist được đặt trong launch environment: `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, các affinity/dynamic settings tương ứng, v.v. Điều này **không xác định** threadpool size thực tế hoặc Torch intra/inter-op settings; không query nội bộ process.

| Quan sát | Mẫu kết thúc 09:33:49 | Mẫu kết thúc 09:35:23 |
|---|---:|---:|
| Thời gian sample | 8,015 s | 8,004 s |
| Tổng CPU từ per-thread counter deltas | 1.585,57% | 100,08% |
| Thread hoạt động | Main + nhiều workers | Main; workers 0 CPU trong sample |
| Logical read bytes (`rchar`) tăng | 16.500 | 363.574.112 |
| Read syscalls (`syscr`) tăng | 6 | 88.817 |
| Physical read bytes tăng | 0 | 0 |

Mẫu đầu: host idle **34,06%**, iowait **0,99%**; available RAM khoảng **157,75 GiB**. Mẫu sau cho thấy một đoạn serial có nhiều logical reads, phù hợp với SQLite activity; **không đủ để gắn đoạn đó chắc chắn vào global BM25 hay parent fallback**. Mẫu đầu cũng không đủ để phân biệt encoder, BLAS hay các native operations khác.

Cumulative I/O tại snapshot đầu: `rchar=75.830.394.253.630` bytes, `syscr=18.504.823.101`, nhưng `read_bytes=16.384` bytes. Đây là bằng chứng logical reads/calls lớn và physical reads được hạch toán cho process rất thấp. Nó không chứng minh toàn bộ host/storage luôn không có contention, cũng không chia được thời gian theo stage. Không diễn giải `rchar` như bytes đọc từ ổ đĩa hay như số bytes dense mmap scan.

RSS snapshot khoảng **3,50 GiB**, HWM **15,44 GiB**, `VmSwap=0`. RSS có thể biến động do memmap/transient arrays; không dùng một snapshot để kết luận matrix không có trong page cache.

Trong cgroup metadata đọc được, `user.slice` có `cpu.max=max 100000`, `nr_throttled=0`; affinity cho phép 32 logical CPUs. Không thấy bằng chứng quota CPU tại cấp đọc được gây mức 162%. Chưa có NUMA/bandwidth/thread-efficiency benchmark.

**Kết luận threading:** 118 threads không tương đương 118 threads đang làm việc. Lifetime 162% che khuất các native bursts dùng nhiều lõi xen kẽ serial work. Tăng số BLAS threads không tự làm vòng lặp Python/SQLite nhanh hơn; cũng chưa có bằng chứng để kết luận thêm threads sẽ giúp dense. `np.linalg.norm(axis=1)` không trở thành parallel GEMV chỉ bằng cách tăng BLAS threads.

## 5. Stage timing: dữ liệu có và thiếu

| Stage cần phân biệt | Full A0 hiện tại |
|---|---|
| Query embedding | Không có timing riêng; CPU encode một query/call trong dense search |
| Dense retrieval không gồm embedding | Không có timing riêng |
| Document BM25 | Không có timing riêng |
| Passage BM25 global | Không có timing riêng |
| Parent BM25 fallback | Không có tổng timing, số calls/tổng postings runtime riêng |
| Fusion | Không có timing riêng |
| Corpus lookup/verify, allocation/payload | Không có timing riêng |
| Pair hashing/serialization/fsync | Không có timing riêng |
| End-to-end preparation | Có checkpoint intervals; không có per-query monotonic duration |

`prepare()` chỉ print `prepared {qid}: 200 immutable pairs`. Output directory hiện chỉ có `pairs/`, `lock.json`, `key_contract.sqlite`; không tìm thấy stage latency log trong directory đó. Diagnostics của shard 240 có hit/candidate/pair counts và `reranker_tokens=null`, không có durations. Capture tmux cũng chỉ có progress lines, không có timestamps theo stage.

Có **benchmark cũ** tại `outputs/v4/p4_b0_full_index_retrieval_probe_20261022/results.json`, cùng index manifest:

| Query | Passage BM25 | `dense_seconds` |
|---|---:|---:|
| 39 | 13,370 s | 7,519 s |
| 74 | 23,048 s | 6,954 s |
| 119 | 48,036 s | 6,868 s |

Theo probe source hiện tại (`scripts/v4/probe_p4_b0_full_index_retrieval.py`), `dense_seconds` bọc toàn bộ `dense.search()`, tức gồm query encoding. Artifact cũ thiếu `device`, config hash, thread settings và runtime metadata; không khẳng định đó là CPU setting giống A0 hiện tại. Đây là B0 passage-only probe trên 3 queries, không có document BM25/parent fallback/full A0 prepare. **Không dùng số liệu này để phân rã latency A0 hoặc dự đoán speedup.** Nó chỉ cho thấy BM25 đã từng mất thời gian đáng kể trong một phép đo có sẵn.

## 6. Optimization đề xuất và parity

Không triển khai bất kỳ mục nào dưới đây trong audit.

| Ưu tiên | Hướng triển khai tương lai | Parity cần giữ | Speedup đã đo |
|---|---|---|---|
| 1 | Parent-first BM25 SQL dựa trên `docs_parent` và point lookup `(term,row_id)`; hoặc batch parent fallback | Giữ global N/DF/avgdl, token set và thứ tự cộng term, cùng công thức FP64 và tie-break; không chuyển sang local-parent IDF | Chưa có; query plan tốt hơn là bằng chứng cấu trúc, không phải benchmark |
| 2 | Precompute actual corpus norms một lần thành sidecar FP64, khóa bằng vector hash/fingerprint/dtype/shape/version | Tính bằng đúng FP64 conversion + `np.linalg.norm(..., axis=1)` như reference; giữ phép `dot / (norm * query_norm)` | Chưa có; chỉ định lượng cache 22,516 MiB và bỏ được repeated norm pass |
| 3 | Cache source/passage theo query hoặc dùng batch lookup; cache source text hash với immutable identity | Không bỏ verification, không sửa text/offset/title, tránh mutation của cached objects; giữ evidence order | Chưa có |
| 4 | Giữ read-only memmap suốt vòng đời backend; reuse FP64 block buffer; chỉ gọi partition một lần | Giữ layout/shape/dtype, cùng scores, threshold/ties và identity order | Chưa có |
| 5 | Tune OpenBLAS và Torch threads độc lập trong process thử nghiệm | Query encoder lẫn BLAS có thể đổi floating-point reduction khi đổi threads; kiểm chứng scores/ranks/pairs | Chưa có; mẫu burst không phải efficiency benchmark |
| 6 | Cache query-level BM25 metadata/postings/scores hoặc completed candidates/pairs | Key phải bao gồm exact query, corpus/index/implementation contract, limits/tokenization; cache đủ dữ liệu fallback | Chưa có; full score/posting cache cần đo memory |

Safe caching có hai mức khác nhau:

- **Đã có:** immutable pair shards được runner validate rồi skip khi resume. Đây là cách tái sử dụng kết quả đã làm an toàn nhất; không cần retrieval lại cho query hoàn tất.
- **Chưa có:** global passage BM25 hiện chỉ trả top-200. Không được dùng riêng top-200 đó làm parent fallback: nhiều parent cần passage ngoài top-200. Nếu tái sử dụng full per-query scores/postings thì cần giữ tất cả matched rows liên quan, global statistics và thứ tự cộng điểm giống reference; đo memory trước.

### Có cần re-embed corpus không?

**Không cần** để precompute norms, giữ matrix hiện tại, tối ưu lookup/BM25 hoặc tạo derivative index/cache từ vectors. Full FP64 matrix derivative tốn khoảng **22,516 GiB** payload, có thể bỏ repeated casts nhưng có chi phí tạo/cache/bandwidth và chưa benchmark. Tạo derivative phải để ở namespace mới; không rewrite vectors/index hiện tại.

Pre-normalized vectors hoặc exact inner-product index cũng có thể tạo từ embeddings đã có, không cần encoder corpus. Tuy nhiên `dot(v/norm(v), q)` có thứ tự rounding khác `dot(v,q)/(norm(v)*norm(q))`; FP32/FAISS/GEMM/khác BLAS không bảo đảm exact ranking ở sát ranh giới. ANN còn có rủi ro bỏ sót candidates. Vì yêu cầu giữ A0 results, **ưu tiên norms sidecar giữ nguyên arithmetic** hơn bỏ norm, đổi precision, đổi metric hoặc ANN.

Ngay cả norm sidecar vẫn phải kiểm tra bitwise/ULP và ranking dưới cùng NumPy/layout; không tự coi mọi cách tính norm tương đương số học là exact parity. Không hứa tăng tốc N lần chỉ từ số vòng scan được loại bỏ. Với phần stage chiếm tỷ lệ f và speedup s, end-to-end speedup là `1 / ((1-f) + f/s)`; cả f và s hiện chưa được đo cho A0.

## 7. Bottlenecks chưa xác minh

Chưa xác minh bằng timing:

- Parent fallback có chiếm phần lớn tổng wall time không; mức amplification thực tế theo term/parent của mọi query.
- Tỷ trọng query encoder CPU, dense FP64 cast/norm/GEMV/top-k/identity lookup.
- Mức độ oversubscription giữa OpenBLAS/Torch/OpenMP, threadpool sizes thực tế, bandwidth/NUMA effects.
- Tỷ trọng global BM25 SQLite join, Python score accumulation và sorting.
- Tỷ trọng corpus connection churn/source hashing hoặc shard fsync.

Bằng chứng mạnh nhất cho ưu tiên nghiên cứu là **term-first parent fallback lặp hàng chục lần/query**; dense redundant norms là chi phí chắc chắn tiếp theo. Chưa có cơ sở nói dense scan là nguyên nhân chính duy nhất, hoặc RAM/ổ đĩa/affinity là nguyên nhân chính.

## 8. Phương án triển khai và resume an toàn

### Trong lúc A0 hiện tại chạy

1. Giữ nguyên PID/session, frozen config, implementation và output namespace. Tiếp tục quan sát checkpoint timestamps/progress bằng read-only metadata.
2. Không hot-patch backend dù backend chưa được lock hash. Không chạy thêm retrieval/model/benchmark cạnh A0.
3. Không tạo norms/FP64 derivative ngay lúc này: đó sẽ là full matrix scan thêm và nằm ngoài phạm vi audit. Tiếp tục dùng checkpoints hiện có.

### Khi có cửa sổ thử nghiệm riêng sau này

1. Tạo version/checkout và output namespace mới, giữ nguyên bản A0 để đối chiếu. Mở rộng implementation provenance để khóa cả dense/BM25/encoder/store và dependency/thread settings. Không rewrite lock A0 hiện có cho code mới.
2. Bổ sung monotonic timing cho query encode; dense cast/norm/dot/top-k/map; doc/pass BM25; tổng parent fallback + số calls; fusion; lookup/verify; allocation/payload; hash/serialize/fsync. Timing nằm ở audit sidecar của run mới, không chèn vào immutable A0 shards.
3. Thử từng optimization một. Ưu tiên BM25 parent-first vì không cần corpus embedding mới hoặc model mới để kiểm tra SQL semantics. Đo old/new parent results trên fixture/corpus copy được phép; giữ global statistics và stable tie-breaks. Đây là công việc đề xuất, chưa chạy trong audit này.
4. Với dense, dùng query-vector fixtures đã lưu ở run thử nghiệm để tách encoder và numerical retrieval parity. Full A0 shards hiện không lưu query embeddings/full branch traces, nên chỉ replay downstream từ shard không chứng minh dense parity được. Mọi reference retrieval mới cần ở cửa sổ được phép riêng.
5. So sánh ordered hits + scores theo branch, fallback hits, fused candidates, toàn bộ ordered 200 pairs và input/cache hashes; cuối cùng so canonical candidates/pair shard bytes. Top-k overlap đơn thuần không đủ. Phải giữ threshold ties và tie-break `(official_id, passage_id)`.
6. Pilot 6 queries/pair fixtures có sẵn chỉ là regression coverage; cần thêm query dài/common terms/ties và corpus coverage đủ rộng trước khi tuyên bố parity. Không coi một pilot pass là bằng chứng mọi query sẽ giữ ranking.
7. Chỉ cân nhắc chuyển run sau khi có measured end-to-end gain và parity. Nếu vẫn cho phép dùng cùng frozen contract, quy trình migration phải khai báo rõ implementation mới và validate mọi shard kế thừa bằng provenance riêng; không chỉnh lock cũ hoặc trộn shard khác contract.

Nếu process hiện tại tự kết thúc ngoài ý muốn và cần resume **cùng implementation**, runner đã hỗ trợ:

```bash
.venv-phase2-batch/bin/python -m scripts.v4.run_p5_2_split prepare \
  --mode full \
  --config configs/v4_submission_p2_a0.yaml \
  --output outputs/v4/a0_t4_v1_full_20261022
```

**Đây chỉ là lệnh tham khảo, audit không thực thi. Không chạy song song với PID hiện tại.** `ensure_lock()` so sánh lock; mỗi existing shard được `validate_pair_shard()` kiểm tra query/lock/pairs/input hashes rồi skip; query chưa publish được làm lại. `_write_bytes_checked()` fsync file tạm rồi rename, từ chối overwrite artifact khác. Runner vẫn khởi tạo pipeline/encoder và kiểm tra index trước khi skip shards, nên resume có startup cost. Không tự xóa hoặc sửa file `.incomplete-*` nếu xuất hiện.

## 9. Quyết định vận hành

**Để A0 tiếp tục.** Khoảng 20% queries đã có shard tại snapshot; throughput hiện tại gợi ý còn gần 3 ngày nếu workload tương tự. Con số đó khiến optimization đáng nghiên cứu, nhất là BM25 fallback, nhưng chưa có benchmark/parity để chứng minh chuyển implementation ngay sẽ rút ngắn thời gian hoàn thành an toàn.

Không nên chỉ tăng threads hoặc chuẩn hóa lại vectors với kỳ vọng bảo toàn ranking mặc định. Bước đầu cho lần thử nghiệm kế tiếp là đo stage timings và kiểm tra parent-first SQL; sau đó dùng actual FP64 norm cache và giảm lookup/allocation dư thừa. Audit này không thay đổi A0 đang chạy.

## 10. Fingerprints bổ sung để tái lập audit

Các source ngoài implementation lock, SHA-256 tại thời điểm audit:

```text
src/medrag/retrieval/disk_dense.py
8ad4a99ec8d52d20e9c7fed853b3bfb6574748f83f208c311d0b531d2d027b62
src/medrag/retrieval/disk_bm25.py
a96c443ed81a04515bc81048bedcf612073461c49288a67eca9b3c6007bf40d0
src/medrag/retrieval/bge_encoder.py
c14b0f565171172c9f477af184c13a16b16cb80bf8460045a27ad2e6c2ee7445
src/medrag/data/disk_store.py
6188d90994de2d7306d9085db9ec071908360b23d97190ab7200bf1d64c87f68
```

Vectors SHA-256 **lấy từ manifest, không hash lại file 11,258 GiB trong audit**:
`d487c2bd9cf1a8402e0192940917b54e9fae4ab8cb501e91f8faf5435c7f63a6`.

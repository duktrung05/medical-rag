# R11 — kế hoạch đo high-recall parent retrieval

Trạng thái: **PLAN, chưa chạy retrieval mới**. R11 là Stage A của
[roadmap FINAL >0,4](v4_over_040_architecture_roadmap_20261026_vi.md).
Champion hiện tại là cap-10 evidence-window: BTC FINAL `0,2324`, DOCS_F2
`0,2788`, CHUNKS_F2 `0,1860`. R11 chỉ đo khả năng đưa đúng parent vào pool;
không đo official F2, không chạy cross-encoder và không tạo submission.

## 1. Câu hỏi và mẫu đo

Câu hỏi: tăng depth, gom điểm ở **parent trước khi cắt**, và chuẩn hóa query có
thu hồi được các verified parents còn thiếu ở một pool đủ nhỏ để ranker xử lý?

- Dùng đúng 87 cặp `(query_id, doc_id)` manual trên 87 official queries trong
  `positive_registry.parquet` và `first_loss_attribution.csv` của R6. Cả 87
  đều có source và evidence được xác minh trong eligible index. Khóa tập này
  trước mọi retrieval; xác minh lại source SHA-256 và evidence containment bằng
  hàm mapping offset đã sửa, không dùng offset R6 cũ.
- Cap-10 giữ nguyên document lists của S2. Baseline manual document coverage là
  **76/87**; còn **11** parent manual ngoài champion docs. Báo cáo cả raw
  union, fused top-400, fused top-2.000 và champion docs; không đồng nhất
  retrieval pool với document output.
- Giữ 10 retained controls đã khóa trong R7 `cohort_lock.json`. Ba control
  queries trùng với 87 manual queries, nên cohort retrieval có **94 unique
  queries**. Đo riêng 76 manual parents đang có trong champion. Không chèn
  champion parents hoặc labels vào candidate pool để làm đẹp gate.
- Historical positives chỉ là preservation diagnostic, không cộng vào 87 manual
  hay dùng để chọn arm. Tập manual đã từng được nhìn trong các vòng trước;
  kết quả R11 là *known-positive coverage*, chưa phải đánh giá độc lập chất
  lượng document ranking.

Đầu vào khóa bằng SHA-256: official query parquet, query split registry,
manual registry/attribution, R7 control lock, cap-10 predictions, effective
competition config, index manifest và model revision. Namespace mới:
`outputs/v4/r11_high_recall_coverage_20261026/`; tuyệt đối không sửa artifacts
S2/cap-10 hoặc index/backing stores.

## 2. Arms và phép chạy

Khóa implementation, query rewrite rules và tie-break trước canary. Bắt đầu
với cùng BGE-M3 query encoder/index và ba branch S2: document BM25, passage
BM25, dense passage. `k_RRF=60` và corpus fingerprint giữ nguyên. Chỉ query
encoding mới được tính; không encode lại documents/passages.

1. **Raw:** dùng nguyên official query text. Với mỗi branch, lấy tối đa 4.000
   hits một lần/query rồi cắt prefix ở `400, 800, 2.000, 4.000`. Nhờ vậy bốn
   depth không tạo bốn lần full-index scan. Ghi cả item rank và parent ID.
2. **Normalized:** dùng hàm `src.medrag.evidence.identity.normalize` đã có:
   Unicode NFKC, HTML entity decode, lowercase và gộp whitespace. Repo hiện
   không có medical synonym/abbreviation lexicon đủ provenance, nên không mở
   rộng từ vựng trong R11 và không tạo synonym theo 87 labels. Cùng bốn depth
   như raw; query không đổi sau normalization dùng lại raw hits, không quét
   index lần nữa.
3. **Parent fusion:** trong từng branch, gộp các passage/document hits cùng
   `official_id` bằng **best branch rank**. Tính RRF trên tối đa ba parent ranks
   (mỗi branch đóng góp một lần), sắp xếp theo `(-score, parent_id)`. So ba arm
   khóa sẵn: raw, normalized, và raw∪normalized. Với union, lấy best parent rank
   của hai query variants *trong cùng branch* rồi mới RRF; không cho một parent
   hưởng gấp đôi điểm chỉ vì xuất hiện ở cả hai variants. Cắt đúng 400 và 2.000
   **unique parents** sau fusion.

Replay S2 mixed-item top-400 trên raw branch prefixes làm reference; đối chiếu
ID/thứ tự với frozen R7 traces ở phần cohort trùng nhau. R7 parent-floor-400
chỉ có frozen results trên 30 R7 queries. Không gộp item trước cut ở các
references; đây là đối chứng cho lợi ích của parent-first fusion. Arm được
chọn trên development theo manual gains ở fused
top-400, sau đó controls và cost; tie-break cố định `raw → normalized → union`,
rồi depth nhỏ hơn. Validation/held-out chỉ xác nhận arm đã chọn, không dùng để
chỉnh rules/weights. Báo cáo mọi depth/arm để chẩn đoán, không chọn cấu hình
mới sau khi mở validation.

## 3. Canaries, budget và kiểm tra

- Preflight dùng R7 query `53, 130, 380`: top-400 doc BM25, top-200 passage
  BM25/dense và S2 mixed-item top-400 phải khớp ID/thứ tự với frozen traces.
  Nếu không khớp, dừng để điều tra runtime/index drift.
- Chạy một canary 3 query ở depth 4.000 cho cả hai variants, ghi CPU seconds,
  wall time, peak RSS, số query được normalize thật và count unique parents.
  Chỉ mở cả cohort nếu ngoại suy nằm trong **4 CPU-hours** và không vượt RAM
  khả dụng. Nếu không, thử raw-only cohort trước và ghi normalized arm là
  `NOT_RUN_BUDGET`; không coi thiếu arm là kết quả âm.
- Tối đa 94 queries, CPU budget 14.400 giây tính cả canary; checkpoint per query
  với input/config hash để resume an toàn. R7 từng mất 1.922 giây/30 raw queries
  ở depth gốc; raw+normalized 94 queries có thể tiến sát budget, nên phải đo
  canary depth 4.000 thay vì suy từ con số đó.
- Mỗi query ghi branch hit IDs/ranks, parent ranks, fused ranks ở mọi depth,
  latency và hash. Validator kiểm prefix nesting, uniqueness, deterministic
  tie-break, source/index identity và không thiếu query. Không cần GPU.

## 4. Metrics và gate quyết định

Đếm **unique query–parent pairs**, không đếm evidence rows trùng parent. Với
87 manual pairs, báo `covered/87` ở raw union, fused top-400 và top-2.000 cho
từng arm/depth; thêm bảng 11 missing parents và first retrieval stage/rank.
Gain = manual parent ngoài champion docs nhưng xuất hiện trong arm; manual loss
= champion manual parent bị loại khỏi fused top-2.000. Báo riêng retained
control losses và per-query gains/losses, không lấy
historical net gain làm proxy official F2.

GO sang thiết kế document ranker khi **một arm đã khóa** đạt đồng thời:

1. Thu hồi ít nhất **10/11** verified manual parents ngoài champion docs ở
   fused top-2.000, **hoặc** tăng *net* manual coverage ít nhất 10 điểm phần
   trăm so với 76/87, tức đạt tối thiểu **85/87**. Nếu giữ đủ 76 manual
   parents cũ, vế thứ hai tương đương thu hồi ít nhất **9/11**.
2. Giữ **10/10** R7 retained controls và **76/76** manual parents champion
   trong fused top-2.000; báo riêng controls vì một số là historical positives.
3. Ít nhất **80% số gains ở top-2.000** cũng nằm trong fused top-400
   (ví dụ 9 gains thì tối thiểu 8 ở top-400). Đây là khả năng đưa gain qua
   cheap fusion, chưa chứng minh ranker cuối giữ được chúng.
4. Validation và held-out cùng chiều với development, không có control loss.
   Với số nhãn nhỏ, báo count theo split; không biến percentage của split thành
   quyết định độc lập nếu mẫu số quá nhỏ.

Ngưỡng 9 hoặc 10 trên 11 misses là **rất chặt**. Nếu chỉ đạt 2–8 gains, báo
`PARTIAL_COVERAGE`, không gọi GO chỉ vì R7 đã cho 2 gains. Nếu gains có trong
top-2.000 nhưng không đạt top-400, ưu tiên sửa cheap parent fusion/pre-ranker.
Nếu cả depth 4.000 vẫn không thêm ít nhất 2 manual parents so với depth 2.000,
dừng tăng depth và chuyển sang query reformulation hoặc kiểm corpus eligibility.
Nếu raw union ở depth 4.000 vẫn bỏ sót nhiều verified parents, báo rõ đó là
retrieval miss; nếu raw union có nhưng fused top-400 không có, báo fusion/rank
miss. Không chuyển 4.000 candidates/query sang cross-encoder.

## 5. Deliverables và handoff

- `cohort_lock.json`: 87 manual pairs, 10 controls, 94 unique queries, split,
  hashes và source checks.
- `runtime_canary.json`: frozen parity, depth-4.000 latency/RAM, budget decision.
- `traces/<qid>.json` và `coverage_by_arm_depth.csv`: raw ranks, parent ranks,
  top-400/top-2.000 membership, gain/loss attribution.
- `decision.json` và execution report: arm/depth đã chọn, metric theo split,
  controls, CPU time, `GO`/`PARTIAL_COVERAGE`/`STOP`, giới hạn suy luận.

Chỉ sau R11 GO mới mở Stage B: xây direct query–document judgments, semantic
pre-ranker rút khoảng 2.000 parents xuống 300–400, rồi train/test document
reranker theo query-disjoint split. R11 tự nó không đổi champion.

# Backlog trước khi có đề kỹ thuật chính thức

**Trạng thái:** Working backlog  
**Phạm vi:** Các hạng mục có thể triển khai an toàn trước khi BTC công bố đầy
đủ dữ liệu, schema, metric và ràng buộc runtime.  
**Nguyên tắc:** Xây retrieval engine tổng quát; chưa khóa quyết định phụ thuộc
đề chính thức.

## 1. Mục tiêu

Trước khi có đề đầy đủ, repository cần đạt trạng thái:

```text
canonical data adapter
  -> BM25 + multilingual dense retrieval
  -> RRF fusion
  -> optional cross-encoder reranking
  -> configurable document/chunk selection
  -> validation + evaluation + manifest
```

Khi BTC công bố dữ liệu, phần việc còn lại chủ yếu là viết adapter, xác nhận
evaluator, build index, calibrate trên dev và khóa cấu hình.

## 2. Quy ước trạng thái

- `[ ]`: chưa làm.
- `[~]`: đang làm.
- `[x]`: hoàn thành và đã qua acceptance gate.
- `BLOCKED_SPEC`: phải chờ đề hoặc dữ liệu chính thức.

## 3. P0 — Retrieval engine chạy thật

### PRE-001 — Mở rộng runtime config

**Trạng thái:** `[ ]`  
**Mục tiêu:** Command chính đọc được cấu hình `demo`, `bm25`, `dense`, `hybrid`
và `hybrid_rerank`.

Tasks:

- [X] Thiết kế config models strict cho retrieval, fusion, reranking, scoring và
  selection.
- [ ] Giữ đường chạy `demo` làm fixture không cần model.
- [ ] Báo lỗi rõ khi index, model hoặc dependency đang bật nhưng bị thiếu.
- [ ] Thêm test từ chối key sai và cấu hình không hợp lệ.

Acceptance gate:

- Bốn config baseline hiện có được load thành công.
- Config sai bị từ chối với thông báo có thể hành động.
- Không còn fallback âm thầm sang demo backend.

### PRE-002 — Composition/factory cho pipeline

**Trạng thái:** `[ ]`  
**Phụ thuộc:** PRE-001

Tasks:

- [ ] Tạo factory cho sparse retriever, dense retriever, hybrid retriever và
  reranker.
- [ ] Sửa `src/service.py` để dựng component theo config.
- [ ] Tách input adapter khỏi retrieval core.
- [ ] Thêm end-to-end fixture cho từng backend.

Acceptance gate:

- `scripts.retrieve` chạy đúng backend được chọn.
- Test chứng minh mỗi config dựng đúng loại component.

### PRE-003 — Implement BM25 production baseline

**Trạng thái:** `[ ]`

Tasks:

- [ ] Build/load sparse index cùng `chunk_id`, `doc_id`, `language` và text.
- [ ] Hỗ trợ Unicode và deterministic tie-breaking bằng ID.
- [ ] Lưu corpus hash và từ chối index không khớp corpus.
- [ ] Hỗ trợ batch query và configurable `top_k`.
- [ ] Thử word tokenization và character n-gram như hai experiment tách biệt.

Acceptance gate:

- `BM25Retriever.search()` không còn stub trả `[]`.
- Cùng input/config tạo cùng ranking.
- Có Recall@K, first-positive rank và Macro F2 trên smoke benchmark.

### PRE-004 — Chuyển exact dense smoke thành retriever production

**Trạng thái:** `[x]`

Tasks:

- [x] Tái sử dụng encode, pooling và normalization từ dense smoke script.
- [x] Cache passage embeddings và ordered chunk IDs.
- [x] Pin model/tokenizer revision trong manifest.
- [x] Hỗ trợ exact search trước; FAISS là adapter tùy chọn cho corpus lớn.
- [x] Có CPU/GPU fallback và batch-size cấu hình được.
- [x] Phát hiện NaN, dimension mismatch và corpus/index mismatch.

Acceptance gate:

- `DenseRetriever.search()` chạy thật từ config chính.
- Ranking có thể tái lập.
- Có báo cáo VI query -> VI/EN/ZH positive riêng biệt.

**Bằng chứng:** `outputs/baseline_dense/dev/report.md` ghi kết quả trên 178
query XQuAD dev và xác nhận hai lượt chạy có ranking/score giống nhau.

### PRE-005 — Hoàn thiện RRF fusion

**Trạng thái:** `[ ]`  
**Phụ thuộc:** PRE-003, PRE-004

Tasks:

- [ ] Deduplicate candidate theo chunk ID.
- [ ] Giữ rank và provenance từ từng retriever.
- [ ] Cấu hình được sparse/dense depth, `rrf_k` và fusion depth.
- [ ] Deterministic tie-breaking.
- [ ] Unit test bằng các ranking nhỏ có expected result biết trước.

Acceptance gate:

- Hybrid ranking chạy end-to-end.
- Báo cáo được positive chỉ do BM25, chỉ do dense hoặc cả hai tìm thấy.

### PRE-006 — Implement cross-encoder reranker

**Trạng thái:** `[ ]`  
**Phụ thuộc:** PRE-005

Tasks:

- [ ] Load model một lần và batch inference.
- [ ] Rerank cặp `(query, chunk_text)`; title/context là field tùy chọn.
- [ ] Cấu hình max length, batch size và candidate depth.
- [ ] Giữ raw retrieval score, rerank score và provenance riêng biệt.
- [ ] Profile latency, RAM và VRAM.

Acceptance gate:

- `BGEReranker.rerank()` không còn stub trả `[]`.
- Có ablation `no-rerank` so với rerank top 50/100/150.
- Reranker không làm mất candidate trước khi selection.

## 4. P0 — Benchmark và evaluator nội bộ

### PRE-007 — Tạo medical multilingual smoke benchmark

**Trạng thái:** `[ ]`

Mục tiêu ban đầu:

- 50–100 query tiếng Việt.
- 200–500 document và 1.000–3.000 chunk.
- Positive bằng tiếng Việt, Anh và Trung.
- Có hard negatives và nhiều loại clinical intent.

Tasks:

- [ ] Lập source registry gồm URL, version, license và review status.
- [ ] Bao phủ diagnosis, symptom, treatment, drug, contraindication, adverse
  effect, monitoring và prevention.
- [ ] Thêm query chứa viết tắt, synonym, negation và qualifier.
- [ ] Tách dev/test theo intent/source group để giảm leakage.
- [ ] Gọi artifact là software/synthetic smoke benchmark cho đến khi có clinical
  review.

Acceptance gate:

- Không có duplicate hoặc parent violation.
- Mỗi query có positive và hard negative.
- Có coverage VI -> VI, VI -> EN và VI -> ZH.
- Source/license report không còn blocker.

### PRE-008 — Evaluator linh hoạt theo contract

**Trạng thái:** `[ ]`

Tasks:

- [ ] Hỗ trợ per-query Precision, Recall và F-beta.
- [ ] Hỗ trợ macro-per-query và micro aggregation.
- [ ] Tính document và chunk metric riêng.
- [ ] Composite weight nằm trong config, không hard-code.
- [ ] Test empty prediction, query không có positive, duplicate và unknown ID.
- [ ] Thêm known-answer tests tính tay.

Acceptance gate:

- Evaluator vượt toàn bộ known-answer tests.
- Thay beta/averaging/weights chỉ cần sửa config.

### PRE-009 — Candidate diagnostics và error taxonomy

**Trạng thái:** `[ ]`  
**Phụ thuộc:** PRE-007, ít nhất một retriever thật

Tasks:

- [ ] Ghi Recall@10/50/100/200, MRR và first-positive rank.
- [ ] Phân tích theo ngôn ngữ positive và clinical intent.
- [ ] Ghi candidate provenance và top false positives.
- [ ] Gắn error tags: corpus, cross-lingual, synonym, abbreviation, negation,
  candidate miss, rerank failure, selection failure và parent aggregation.

Acceptance gate:

- Với mỗi query lỗi có thể xác định lỗi ở retrieval, reranking hay selection.
- Có `error_analysis.jsonl` trong mỗi experiment.

## 5. P1 — Selection và document/chunk consistency

### PRE-010 — Hoàn thiện calibration engine và CLI

**Trạng thái:** `[ ]`

Tasks:

- [ ] Nối `scripts/calibrate_threshold.py` với calibration engine hiện có.
- [ ] So sánh fixed K, absolute threshold, relative threshold, score gap và
  dynamic K.
- [ ] Grid search chỉ trên dev.
- [ ] Xuất toàn bộ kết quả, không chỉ cấu hình thắng.

Acceptance gate:

- CLI tạo `calibration_results.csv` và locked config.
- Không đọc test qrels trong quá trình calibration.

### PRE-011 — Joint document/chunk selection

**Trạng thái:** `[ ]`

Tasks:

- [ ] So sánh document aggregation bằng max, mean top-N và hybrid score.
- [ ] Chọn chunk trước rồi suy ra supporting documents như một strategy riêng.
- [ ] Giữ strategy chọn doc/chunk độc lập để ablation.
- [ ] Enforce parent consistency trước khi serialize.

Acceptance gate:

- Không còn parent violation.
- Có ablation chứng minh strategy cuối tốt hơn fixed-K baseline trên dev.

## 6. P1 — Reproducibility và release discipline

### PRE-012 — Manifest cho mọi experiment

**Trạng thái:** `[ ]`

Manifest tối thiểu:

- Git commit và trạng thái worktree.
- Config hash.
- Corpus/query/qrel hashes.
- Model và tokenizer revisions.
- Dependency versions, seed và device.
- Candidate depths và selection parameters.
- Artifact hashes và elapsed time.

Acceptance gate:

- Mỗi output directory có config, manifest, rankings, predictions, metrics và
  error analysis.

### PRE-013 — Một command end-to-end

**Trạng thái:** `[ ]`  
**Phụ thuộc:** PRE-001 đến PRE-012 theo backend được chọn

Command phải thực hiện:

```text
validate input
  -> verify/build index
  -> retrieve
  -> fuse/rerank
  -> aggregate/select
  -> validate prediction
  -> evaluate
  -> write manifest/report
```

Acceptance gate:

- Chạy được trong clean environment theo README.
- Hai lần chạy cùng config/input tạo ranking giống nhau trong tolerance đã định.
- Không có thao tác thủ công giữa các bước.

### PRE-014 — Củng cố test suite và dependency boundaries

**Trạng thái:** `[ ]`

Tasks:

- [ ] Unit tests không yêu cầu tải model hoặc internet.
- [ ] Đánh dấu tests cần Torch/model bằng pytest marker riêng.
- [ ] Mock encoder cho dense unit tests.
- [ ] Thêm index mismatch, Unicode, tie, empty input và ID invariance tests.
- [ ] Thêm test cấm routing theo query ID.

Acceptance gate:

- Core tests chạy với dependencies mặc định.
- Retrieval integration tests chạy khi cài extra `retrieval`.

## 7. P2 — Chuẩn bị query understanding, chưa bật mặc định

### PRE-015 — Medical terminology registry

**Trạng thái:** `[ ]`

Tasks:

- [ ] Thiết kế registry Việt–Anh–Trung cho disease, symptom, drug, test và
  anatomy.
- [ ] Lưu source/review status cho mỗi synonym.
- [ ] Hỗ trợ abbreviation và canonical term.
- [ ] Log chính xác term nào được thêm vào query.
- [ ] Đặt expansion sau feature flag, mặc định `false`.

Acceptance gate:

- Có ablation original query so với expanded query.
- Chỉ bật khi tăng hoặc giữ Macro F2 và cải thiện cross-lingual recall.

## 8. Các quyết định phải chờ đề chính thức

Đánh dấu `BLOCKED_SPEC`; không khóa trước:

- [ ] Exact input/output schema và field names.
- [ ] Ranking được chấm như list hay set.
- [ ] Min/max document và chunk trong output.
- [ ] Empty prediction có hợp lệ không.
- [ ] Công thức kết hợp document/chunk F2.
- [ ] Corpus đã chunk sẵn hay phải tự chunk.
- [ ] Final chunk size/overlap.
- [ ] Cho phép external corpus, translation API hoặc LLM hay không.
- [ ] Runtime, internet, RAM, VRAM và latency limits.
- [ ] Final model, ANN index, threshold, top K và aggregation strategy.
- [ ] Fine-tuning và training data policy.

## 9. Những việc không làm trước khi có bằng chứng

- Không fine-tune trên XQuAD rồi coi đó là medical improvement.
- Không tối ưu threshold quá kỹ trên smoke benchmark nhỏ.
- Không viết logic theo query ID hoặc manual override từng câu.
- Không đưa LLM planner/agentic retrieval vào critical path lúc này.
- Không xây vector database phức tạp trước khi biết corpus size.
- Không hard-filter theo ngôn ngữ trong bài toán cross-lingual.
- Không dùng test set để chọn model, K hoặc threshold.

## 10. Thứ tự thực hiện đề xuất

1. PRE-001, PRE-002 — runtime/config thật.
2. PRE-003, PRE-004 — BM25 và dense baselines.
3. PRE-007, PRE-008 — medical smoke benchmark và evaluator.
4. PRE-005, PRE-009 — hybrid và candidate diagnostics.
5. PRE-006 — reranker.
6. PRE-010, PRE-011 — calibration và joint selection.
7. PRE-012, PRE-013, PRE-014 — release/reproducibility/tests.
8. PRE-015 — terminology expansion có kiểm soát.

## 11. Definition of Done trước khi BTC công bố đề

- [ ] Không còn production retriever/reranker stub trả `[]`.
- [ ] BM25, dense, hybrid và hybrid-rerank chạy từ config chính.
- [ ] Có medical multilingual smoke benchmark có provenance.
- [ ] Có candidate recall diagnostics và error analysis.
- [ ] Evaluator thay đổi contract qua config.
- [ ] Selection/calibration không đọc test qrels.
- [ ] Test suite được tách core/integration và đều có command rõ ràng.
- [ ] Một command tái tạo experiment end-to-end.
- [ ] Mọi experiment có manifest và hashes.
- [ ] Các quyết định phụ thuộc đề vẫn được để mở, không biến thành assumption ẩn.

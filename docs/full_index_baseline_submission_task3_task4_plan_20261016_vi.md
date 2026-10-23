# Plan Task 3–4: full-release indexes → baseline 1.200 queries → BTC score

Ngày: 16/10/2026. Trạng thái: kế hoạch, chưa triển khai Task 3–4.

**Thay thế thiết kế Task 4:** dùng [kế hoạch sửa V4 trước Task 4](v4_evidence_fixes_before_task4_plan_20261016_vi.md) làm căn cứ triển khai hiện hành. Workflow document-first `2/6 passages đầu`, parity với batch `r2ai` Task 2, và selector/threshold legacy trong các mục 2–4 dưới đây là thiết kế migration lịch sử, không áp dụng cho V4 Task 4 mới. Reference hiện hành là V4 Task 3 all-passage qua `load_prepared → build_pipeline → Pipeline`. Các yêu cầu coverage, provenance, checkpoint và strict validation vẫn được kế thừa theo kế hoạch mới.

**Cập nhật phạm vi:** sau khi thống nhất vai trò workflow, Task 3 được thực hiện là V4 core integration và được ghi ở `docs/task3_v4_workflow_integration_result_20261016_vi.md`. Phần full-release index bên dưới được chuyển sang Task 4 cùng với baseline 1.200 query và BTC scoring; trạng thái chạy của Task 4 vẫn là chưa bắt đầu.

Task 3–4 ở tài liệu này tiếp nối Task 1–2 trong `phase2_batch_integration_subset_verification_plan_20261016_vi.md`; không phải Task 3/4 của Phase 3 dedup, cũng không phải T3/T4 của kế hoạch ablation C/R/S.

## 1. Căn cứ đã kiểm tra và mục tiêu

| Căn cứ | Kết quả đã có | Hệ quả cho bước tiếp theo |
|---|---|---|
| Task 2 real-model subset | 1.000 docs, 2.743 passages, 20 official queries; 2 run PASS | Có thể chuyển sang triển khai đường chạy toàn release; chưa có bằng chứng F2 |
| Repeatability | 4.000 candidate scores trùng chính xác; cùng document/passage order và canonical predictions | Dùng làm regression contract cho implementation mới trên cùng dataset nhỏ |
| Official snapshot | `links_corpus_vi_en_zh.parquet`: 1.220.598 rows; 1.200 official queries | Phải phân biệt official URL inventory với tập documents có nội dung đạt chuẩn |
| Phase 1/2 release | 1.073.531 PASS docs, 2.951.273 passages; 6 semantic exclusions tương ứng 49 passages | Mục tiêu full-release index: 1.073.525 docs và 2.951.224 eligible passages; kiểm lại bằng join/count thực tế |
| Runner hiện tại | `scope=explicit_subset_only`, `max_documents=1000`; materialize text/vectors, build index trong run | Cần loader/index/query runner cho full release, không chỉ bỏ giới hạn 1.000 |
| Feedback lịch sử | DOCS_F2 0,280164632426; CHUNKS_F2 0,055969257307; FINAL 0,168066944867, trên 593 scored queries | Là mốc lịch sử; so delta chỉ khi nối được ZIP/hash và cùng phase/query set/scorer |

Mục tiêu Task 3: tạo index và lookup có thể tái sử dụng, đầy đủ theo eligible release, có kiểm tra identity, resume và tài nguyên.

Mục tiêu Task 4: chạy một baseline cố định trên 1.200 official queries, xuất ZIP hợp lệ và ghi nhận score BTC của chính ZIP đó. Sau đó mới dùng baseline này làm đối chứng tối ưu Candidate Retrieval/Evidence Ranking.

Không có mục tiêu điểm số dự đoán hoặc cam kết runtime từ subset. Tỷ lệ query rỗng 6/20 ở Task 2 không được dùng để ước lượng kết quả toàn release.

## 2. Workflow được chốt cho hai task

```text
Phase 1/2 release + PASS/exclusions + official IDs
                       ↓
Task 3: streaming preflight + coverage ledger + passage/provenance lookup
                       ↓
       document BM25 index + document dense vectors
                       ↓
Task 4: 1.200 queries → BM25 top-400 + dense top-400 → RRF top-400
                       ↓
       union parent IDs → lookup toàn bộ passages của các parent
                       ↓
       embed passages cần dùng, cache theo identity/fingerprint
                       ↓
       dense top-200 passages/query → rerank → selector đã khóa
                       ↓
       grounding + strict validation → submission.zip + SHA-256
                       ↓
       BTC score/feedback → baseline metrics + quyết định thí nghiệm
```

Chỉ chọn passage trong parent pool như Task 2. Tính embedding passage một lần cho union parent IDs giúp tránh encode mọi passage khi baseline không truy cập đến chúng. Quy tắc lấy toàn bộ passages trong mỗi parent và cách scoring phải giữ tương đương với Task 2. Đây là quyết định triển khai cache, chưa phải thay đổi chiến lược retrieval.

## 3. Task 3 — Build full-release index và lookup

### T3.1 — Khóa scope, identity và coverage

1. Khóa hashes Phase 1/2 release, input snapshot, mapping, frozen chunking config, semantic exclusions, official corpus và queries. Kiểm liên kết release trước build.
2. Đọc tất cả shards được release tham chiếu theo batch; kiểm checksum trước tiêu thụ. Không dùng glob để tự lấy thêm part.
3. Đối chiếu official IDs bằng join thực tế: duplicate IDs/URLs, unknown IDs, PASS, held, staging, docs không có passages và official URLs không có eligible content. Xuất ledger theo lý do và thống kê nguồn/ngôn ngữ.
4. Chênh lệch số lượng hiện quan sát là 147.073 giữa official rows và eligible docs; chỉ sau khi kiểm unique IDs/anti-join mới kết luận tập IDs cụ thể chưa được index. Không gọi 1.073.525 docs là toàn bộ 1.220.598 URLs.
5. Kiểm một-một `(release_sha256, doc_id, passage_index) ↔ passage_id`; kiểm text/hash/offset theo `normalized_text`. Toàn bộ eligible rows phải có lookup; không rechunk hoặc gộp official IDs theo text giống nhau.

Đầu ra: `corpus_lock.json`, `coverage_ledger.parquet`, `coverage_report.json`, `input_integrity_report.json`.

### T3.2 — Tạo passage store và document representations

- Xây lookup lưu trên đĩa từ doc ID tới passage rows, kèm title, URL, offsets, source/representation hashes và release identity. Có thể dùng SQLite cùng Parquet; index row là vị trí nội bộ, luôn ánh xạ về identity chính thức.
- Streaming các shard chỉ một lần cho mỗi stage; không quét lại cả 922 shards cho từng query.
- Giữ representations đã test: BM25 = title + tối đa 6 passages đầu; dense document = title + tối đa 2 passages đầu, max length 256. Passage order theo `passage_index`, giữ nguyên separator/text formatting của Task 2.
- Publish theo shard/partition với checksum, row count, first/last identity và trạng thái complete. Checkpoint chỉ được resume nếu input/config/model fingerprint trùng.

Đầu ra: `lookup/`, `document_representations/`, `row_mappings/`, `checkpoint_manifest.json`.

### T3.3 — Build BM25 và dense document indexes

- Tái dùng tokenizer và BM25 Lucene semantics trong Task 2; lưu vocab, parameters, row mapping và tokenizer/code fingerprint. Kiểm memory của quá trình xây sparse index; dùng phân vùng nếu implementation hiện có vượt budget.
- Nếu BM25 chia shard, document frequency/IDF và average length phải tính toàn corpus trước; query dùng cùng statistics và merge top-K toàn cục. Không merge raw scores từ các BM25 index tự tính IDF riêng rồi gọi tương đương.
- BGE-M3 pin revision `5617a9f61b028005a4858fdac845db406aefb181`. Giữ document input/max length, normalization và precision contract như Task 2; ghi dimension thực tế của encoder.
- Ghi vectors theo shard/memmap cùng explicit doc IDs và SHA. Không giữ toàn bộ texts/vectors dưới dạng Python objects; không encode lại documents mỗi lần chạy query.
- Baseline đầu dùng dense exact search theo block, merge top-K và stable tie-break theo doc ID. ANN/quantization là thí nghiệm sau nếu benchmark chỉ ra cần; phải đo sai khác candidate so với exact trước khi dùng.
- Cache index lịch sử chỉ được tái sử dụng khi toàn bộ fingerprint khớp, không suy tương thích từ tên BGE-M3.

Đầu ra: `indexes/bm25/`, `indexes/dense_document/`, `index_manifest.json`.

### T3.4 — Benchmark, resume và nghiệm thu

1. Khóa pilot 10.000 eligible docs theo ngôn ngữ/độ dài và 50 official queries trước benchmark. Các số 10.000/50 là scope kiểm thử tài nguyên được đề xuất, chưa phải run đã có.
2. Trên đúng 1.000 docs/20 queries của Task 2, so đường index mới với runner cũ: IDs/order và canonical payload phải khớp; kiểm dense/reranker score theo tolerance đã ghi. Test đảo shard order, tie-score, resume và lỗi input/hash.
3. Đo docs/s encoding, tokens/s, RAM RSS peak, GPU allocated/reserved peak, BM25 peak build RAM, disk temporary/final và query latency cold/warm. Chỉ lập ETA full sau số đo này.
4. Cho dừng sau một partition và resume: không duplicate/missing rows, không encode lại partition complete, final logical fingerprint khớp run liên tục.
5. Build toàn eligible release và nghiệm thu coverage/count/checksums trên toàn bộ rows, không lấy pilot PASS thay cho full build PASS.

Budget khởi đầu đề xuất: 1 GPU rảnh, RSS tối đa 32 GiB, task disk tối đa 128 GiB, CPU threads giới hạn 8; điều chỉnh bằng số đo pilot trước full build. Có thể cấu hình hai GPU để chia embedding khi tài nguyên thực cho phép. Khi bắt đầu phải kiểm tra lại GPU/RAM/disk; snapshot hiện tại không bảo đảm chúng còn trống lúc chạy.

Gate hoàn tất Task 3 — `FULL_RELEASE_INDEX_PASS`:

- Đủ 1.073.525 eligible doc IDs và 2.951.224 eligible passage rows nếu reconciliation khớp manifest; không unknown/held/staging IDs.
- BM25/dense indexes cùng corpus lock, doc IDs và representations; không thiếu/trùng/sai vector row mapping.
- Integrity/grounding toàn input đã tiêu thụ PASS; query parity trên locked subset PASS.
- Resume/tie-break PASS; có benchmark thực và tài nguyên trong budget.
- `index_manifest.json` complete chỉ xuất sau khi mọi gate đạt; index build FAIL/BLOCKED ghi đúng lý do và giữ checkpoint.

## 4. Task 4 — Baseline toàn release, ZIP và score thật

### T4.1 — Khóa baseline trước khi chạy

Giữ cấu hình migration đã verify để tạo mốc có thể so sánh; threshold -2 hiện chưa được hiệu chỉnh cho toàn release.

| Thành phần | Baseline khóa |
|---|---|
| Query/corpus | 1.200 official query IDs, corpus lock của Task 3 |
| Document candidates | BM25 top-400 và dense top-400; weighted RRF 1:1, k=60; fused top-400 |
| Passage candidates | Toàn bộ eligible passages thuộc parent pool; dense shortlist tối đa 200/query |
| Reranker | BGE-reranker-v2-m3 revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, max length 512, raw logits; rerank tất cả shortlist |
| Selection | Threshold -2; max_docs 200, max_chunks 60, cap 1 chunk/doc; document order và dedup như Task 2 |
| Optional stages | ICD/intent/expansion/multi-vector/Phase 3 canonicalization tắt |

Khóa `baseline_config.json`, code revision/diff snapshot, dependency versions, model revisions, input format và precision. Thay đổi bất kỳ nội dung nào tạo run/arm mới.

### T4.2 — Retrieval/reranking có cache và checkpoint

1. Chạy document retrieval trên index hoàn chỉnh; lưu riêng hai branch scores/ranks, union trước fusion và final parent pool theo từng query.
2. Khóa union parent IDs, materialize lookup và embed toàn bộ eligible passages thuộc union. Cache vectors theo passage identity + title/text/model/tokenization/normalization/precision; resume theo partition.
3. Chạy dense passage search theo parent pool. Log số passages có thể truy cập, số thực sự shortlisted/reranked và unique parents. Không ép đủ top-K bằng candidates trùng.
4. Rerank theo query batches và ghi durable score cache. Với 1.200 queries và cap 200, tối đa 240.000 query–passage pairs; đây là giới hạn workload theo config, không phải ETA. Số passages phải embed trong union được thống kê riêng.
5. Ghi token lengths/truncation, stage latency, RSS/VRAM/disk và query-level completion. Lỗi query là failed/pending checkpoint; không biến lỗi inference thành prediction rỗng hợp lệ.
6. Replay selector từ cache trên CPU; kiểm lại một mẫu query bằng inference độc lập, candidate/order và payload theo protocol repeatability. Không cần encode lại toàn corpus để kiểm replay.

Đầu ra: `document_candidates.parquet`, `parent_union_manifest.json`, `passage_embedding_manifest.json`, `rankings.jsonl.gz` hoặc cache partition tương đương, `diagnostics.jsonl`, `runtime_report.json`.

### T4.3 — Export và strict validation

- Xuất đúng 1.200 query IDs, kể cả prediction rỗng do selector; log count distribution và lý do gate/cap/dedup.
- Mỗi evidence truy được về đúng eligible Phase 2 passage/source slice và parent official ID/URL. Không thêm title vào evidence text xuất cho BTC.
- Validator kiểm full official corpus/query snapshot explicit, schema, duplicate IDs, parent consistency, output caps, ZIP CRC/entries, JSON và finite scores nội bộ.
- Replay cache phải sinh cùng canonical prediction SHA; ghi SHA-256 của chính file ZIP dùng chấm.
- Tách provenance/metadata khỏi ZIP; publish `submission.zip` và completion manifest sau validation PASS.

Gate `FULL_BASELINE_EXPORT_PASS`: 1.200/1.200 queries đã xử lý, input/index hashes khớp, grounding/strict validator/cache replay PASS, có ZIP/hash và metrics tài nguyên. Gate này chưa chứa kết luận F2.

### T4.4 — Nhận score BTC và xác lập đối chứng

1. Nối artifact baseline lịch sử từ `outputs/scored_feedback_recheck_20261009/RECHECK_SUMMARY.json`: original ZIP/SHA, phase, query coverage, scorer và per-query diagnostics. Thiếu liên kết nào thì ghi rõ.
2. Chấm đúng ZIP full baseline mới bằng đường chấm BTC đang áp dụng. Khi thực hiện phải xác minh phase/quota/luật scorer từ thông tin BTC hiện hành; không dùng quota ghi trong comment code làm căn cứ.
3. Lưu submission ID/time/phase, ZIP SHA, raw feedback và exact DOCS_F2/CHUNKS_F2/FINAL. Nếu có scorer/gold hợp lệ tại local thì tái tính; CSV feedback cũ chỉ chấm arm cũ, không đủ để tự chấm ZIP mới.
4. So với lịch sử trên cùng điều kiện chấm. Nếu query set/phase/scorer khác, báo hai score riêng và giới hạn so sánh; nếu corpus/chunk representation đã đổi, delta end-to-end không được gán riêng cho candidate hay reranker.
5. Có per-query results tương ứng thì phân tích stage loss và paired delta trên tập eligible chung. Chỉ có aggregate score thì chỉ báo aggregate delta, không tự dựng CI.

Trạng thái kết thúc phải phân biệt:

- `FULL_BASELINE_EXPORT_PASS / SCORE_PENDING`: ZIP hoàn tất, chưa có feedback mới; F2 mới `UNMEASURED`.
- `FULL_BASELINE_SCORED`: feedback đã nối đúng ZIP/hash/phase; báo điểm thực, kể cả điểm thấp hơn lịch sử.
- `FAIL/BLOCKED`: chỉ rõ gate/runtime/input/đường chấm nào chưa đạt; không công bố complete thay thế.

Đầu ra: `baseline_manifest.json`, `submission.zip`, `validation_report.json`, `scoring_receipt.json`, `raw_feedback/`, `baseline_metrics.json`, `FULL_BASELINE_REPORT.md`. Receipt/metrics chính thức chỉ ghi khi có kết quả chấm thực.

## 5. Thứ tự thực hiện, file dự kiến và ước lượng

| Thứ tự | Công việc | Kết quả cần có trước bước sau |
|---|---|---|
| 1 | T3.1–T3.2: corpus lock, streaming lookup/representations | Coverage reconciliation và identity PASS |
| 2 | T3.3–T3.4: BM25/dense build, pilot, parity/resume, full build | `FULL_RELEASE_INDEX_PASS` |
| 3 | T4.1–T4.2: baseline khóa, 1.200 queries, passage/reranker caches | Toàn bộ queries complete, score cache hợp lệ |
| 4 | T4.3: replay/grounding/export | `FULL_BASELINE_EXPORT_PASS`, ZIP SHA |
| 5 | T4.4: BTC feedback và phân tích | Score nối đúng artifact hoặc `SCORE_PENDING` |

Các file dự kiến thêm, chưa tồn tại theo plan này:

- `src/medrag/data/phase2_corpus_store.py`: streaming eligible records, disk lookup và coverage.
- `src/medrag/retrieval/phase2_full_index.py`: BM25/dense builders, row mapping, checkpoint và search.
- `src/medrag/data/phase2_full_runner.py`: retrieval/reranking partition, cache/replay và export.
- `scripts/workflows/vibiomir/build_phase2_full_index.py`, `run_phase2_full_baseline.py`.
- `configs/vibiomir/phase2_full_index_v1.json`, `phase2_full_baseline_v1.json`.
- Focused tests: subset parity, global BM25 statistics, ID/vector mapping, parent-filtered search, tie-break, resume, malformed lineage và export coverage.

Output namespace mới đề xuất: `outputs/v4/phase2_full_index/<run_id>/` và `outputs/v4/phase2_full_baseline/<run_id>/`. Giữ nguyên runner/config subset để còn regression đối chiếu.

Ước lượng công kỹ thuật: Task 3 khoảng 2–3 ngày; Task 4 khoảng 1–2 ngày. Thời gian embedding/index build/full inference và chờ BTC được dự toán riêng sau pilot, chưa có số giờ đã kiểm chứng cho workload này.

Sau Task 4 có score, nối sang kế hoạch ablation C/R/S: thử BM25-only/dense-only/hybrid cùng budgets/downstream để đo candidate contribution, rồi khóa pool để đo ranking, cuối cùng replay selection. Một full baseline có điểm chưa đủ để kết luận thành phần nào tạo gain.

## 6. Artifact làm căn cứ

- `outputs/v4/phase2_batch_subset/20261016/SUBSET_INTEGRATION_REPORT.md` và `repeatability_report.json`.
- `configs/vibiomir/phase2_batch_migration_v1.json`.
- `src/medrag/data/phase2_batch_runner.py`, `src/medrag/data/phase2_batch.py`, `r2ai/rank.py`.
- Phase 1 `release/phase1_release_manifest.json` và Phase 2 `manifests/phase2_release_manifest.json`.
- `outputs/scored_feedback_recheck_20261009/RECHECK_SUMMARY.json` và `SCORING_RECHECK_REPORT.md`.
- `docs/candidate_retrieval_evidence_ranking_f2_plan_20261016_vi.md`.

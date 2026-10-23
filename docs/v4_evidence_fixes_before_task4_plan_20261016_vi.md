# Kế hoạch sửa V4 trước Task 4 toàn corpus

Ngày: 16/10/2026. Trạng thái: **đang triển khai**. F0 và selection/export F1 đã chạy; F2–F3 có diagnostics/arm subset nhưng còn score cache, qrels và strict validator; F4 mới đạt canary lexical trên đĩa, chưa đạt backend production dense/full corpus; F5 chưa chạy. Xem `docs/v4_evidence_fixes_execution_20261016_vi.md`.

Phần thực hiện tiếp theo đã được cụ thể hóa thành [kế hoạch Task 4 và chuẩn output](v4_task4_output_acceptance_plan_20261016_vi.md), gồm contracts, artifacts, dependencies và gates PASS/FAIL.

## 1. Quyết định và phạm vi

Hoàn tất sửa correctness trên V4 subset trước full query inference. Backend toàn corpus được phát triển theo contracts này trong Task 4. Các partition passage store/embedding đã đúng fingerprint có thể tiếp tục được xây và tái sử dụng.

Task 3 đã đưa Phase 2 vào `load_prepared → build_pipeline → Pipeline`; đây là reference runtime. Không lấy hành vi hay canonical predictions của batch `r2ai` làm chuẩn tương đương cho V4. Task 3 mới có một query smoke với model thật, chưa có baseline V4 đủ 20 queries hoặc F2.

Phạm vi sửa: candidate/evidence contracts, selection/export, trạng thái scoring, diagnostics, regression tests và yêu cầu backend Task 4. Giữ nguyên Phase 1/2 release, normalized text, passage IDs, offsets và provenance. Không cần crawl/rechunk để thực hiện các sửa này. Tuning thresholds, fusion weights, context expansion và model mới đi ở các arm riêng.

## 2. Các phát hiện làm căn cứ

| Phát hiện trong code hiện tại | Đánh giá | Hướng xử lý |
|---|---|---|
| `build_lexical()` dùng title + full source text, passage BM25 dùng tất cả passages; `Dense` encode từng passage | Lỗi chỉ biểu diễn đầu bài đã được xử lý ở V4 subset | Backend full phải giữ khả năng truy hồi passage ở mọi vị trí |
| Adapter `records()` cắt `d["evidence"][:1]` | Export âm thầm quyết định số evidence, không có selection contract | Tách selector và serializer; serializer giữ đúng output đã chọn |
| V4 hiện chưa có selector explicit giữa retrieval/rerank và adapter | Chỉ xóa slice sẽ xuất cả candidates, có thể tăng noise hoặc xuất evidence chưa score | Thêm policy chọn evidence có caps và diagnostics |
| Reranker lấy tối đa 3 evidence/doc theo thứ tự doc, tổng 200 pairs | Một số docs không được score; budget là tradeoff, chưa chứng minh là nguyên nhân giảm F2 | Ghi trạng thái rõ; thử parent-first ở cùng tổng budget |
| `fuse()` dùng key `(official_id, passage_id)` | `fused_candidate_limit=400` hiện là 400 items gồm doc hits và passage hits, không phải 400 unique parents | Đặt tên/trace đúng đơn vị; giữ semantics khi thay backend |
| Fallback tìm evidence bằng `passage_bm25.search(query, len(self.passages))` | Quét/scoring toàn passage corpus cho fallback không phù hợp full runtime | Thêm lookup/search theo parent, giữ nguyên passage identity |
| Dense và reranker có max length 512, nhưng title/query cũng chiếm tokens | Index tất cả passages không đồng nghĩa model nhìn đủ mọi token trong từng input | Đo truncation bằng đúng tokenizer trước khi cân nhắc đổi text |
| `v4_core.yaml` tắt models; subset config bật models | Đây là cấu hình có chủ đích, không phải bug | Tạo production config explicit và preflight kiểm stages thực sự bật |
| Kế hoạch full-index cũ yêu cầu 2/6 passages đầu và parity với batch Task 2 | Mâu thuẫn với reference V4 Task 3 | Các quy định đó bị thay thế bởi tài liệu này |

## 3. Luồng mục tiêu và các bất biến

```text
Phase 1/2 eligible release → passage/source store + identity lookup
                                      ↓
             V4 document BM25 trên full normalized body
             + BM25 trên mọi passage + dense trên mọi passage
                                      ↓
                fusion theo policy V4 được version hóa
                                      ↓
                evidence candidates + parent lookup
                                      ↓
                phân bổ pairs → reranker → score cache
                                      ↓
                explicit V4 selector → selected evidence
                                      ↓
                CLI/API serializer và submission adapter
                                      ↓
                grounding/strict validation → ZIP/hash
```

- Passage giữa/cuối có thể đưa parent vào candidate pool mà không cần parent lọt một gate chỉ đọc đầu bài. Điều này nói về khả năng truy hồi; top-K hữu hạn vẫn có thể bỏ passage relevant.
- Evidence text luôn là exact Phase 2 text, offsets trỏ vào `normalized_text`. Title chỉ dùng trong model input. Không ghép đoạn hoặc sinh evidence mới.
- Identity là corpus/release + parent ID + passage ID; vector row chỉ là lookup. Hai official IDs cùng text vẫn là hai parents.
- `candidate`, `reranked`, `selected`, `exported` là các tập riêng. Export phải bằng selected về membership, text và thứ tự; không được có cap thứ hai trong adapter.
- Khi reranker bật, evidence chưa được rerank không tự trở thành evidence đã chấm. Mọi thiếu score, bị budget loại hoặc lỗi inference đều có trạng thái riêng.
- Cùng config, inputs và backend exact phải có thứ tự ổn định. Mọi thay đổi thuật toán hoặc tokenization dùng config/arm mới.

## 4. Các bước triển khai

### F0 — Khóa reference V4 và phân loại job/artifact đang chạy

1. Ghi command, host/container, PID nếu có, stage, output namespace và checkpoint của full job thực tế. Quyết định theo stage đã xác minh, không chỉ dựa vào tên V4.
2. Khóa snapshot Task 3: 1.000 docs, 2.743 passages, 20 queries; manifest tại `outputs/v4/phase2_subset/20261016/selection_seed20261016/manifest.json`.
3. Chụp code/config/input hashes và dependency/model revisions. Worktree có nhiều file uncommitted/untracked, nên chỉ ghi Git HEAD là không đủ: lưu inventory hashes của code thực được dùng và diff phù hợp.
4. Chạy reference V4 hiện tại trên đủ 20 queries với cùng subset model config; lưu raw candidates, evidence, scores và serialized output. Không dùng kết quả repeatability của batch `r2ai` thay kết quả này.
5. Chụp số liệu trước sửa: evidence bị adapter cắt, unique parents, passages được score/chưa score, token lengths, latency và memory. Các con số mới chỉ ghi sau khi chạy.

Đầu ra: `baseline_lock.json`, `artifact_reuse_report.json`, `reference_v4_20q/`.

### F1 — Selection/export explicit, không làm mất evidence âm thầm

1. Thêm V4 `SelectionPolicy` và selected-result contract. Selector nhận candidate scores/status, chọn docs/evidence theo cấu hình; adapter chỉ đổi sang schema ViBioMIR.
2. Khóa policy compatibility ban đầu: tối đa 1 evidence/doc để tái lập hành vi export hiện tại trên healthy fixtures. `max_docs` và `max_chunks` là config explicit; ghi đúng số hiệu dụng của từng run, không tự nhập threshold -2 hoặc cap 60 từ `r2ai`.
3. Thêm policy kiểm thử với 2 evidence/doc để chứng minh cả hai passages đã chọn sống qua API/submission. Đổi cap production là quyết định ablation ở F3.
4. Khi reranker bật, mặc định selector chỉ nhận scored evidence hợp lệ. Evidence không được score do hết budget bị loại với reason rõ; document không có evidence hợp lệ không được tự nhận evidence đầu bài.
5. Nếu reranker tắt có chủ đích, cho phép policy lexical/dense dựa trên retrieval ranking; ghi chế độ đó trong manifest. Cấu hình production yêu cầu model mà model không sẵn sàng phải fail.
6. Dùng identity để loại candidate trùng do nhiều retriever. Không dedup exact text xuyên parent IDs. Policy xử lý passages overlap trong cùng parent phải explicit; vòng đầu chỉ loại trùng identity.
7. CLI giữ raw diagnostic output để kiểm tra; API và submission cùng tiêu thụ selected-result contract. So sánh cùng query/config/result limits; API hiện limit 10 và CLI có `--limit`, nên phải khóa cùng giới hạn khi test parity.
8. Giữ query rỗng hợp lệ sau selection khác với query lỗi. Thêm strict validator V4 cho schema, official IDs, query coverage, parent relation và selected/exported equality.

Gate `V4_SELECTION_EXPORT_PASS`: selected = exported; fixture cap 2 xuất đủ hai evidence đúng parent/text; không xuất unscored khi reranker bật; compatibility healthy fixture không đổi ngoài các khác biệt đã giải thích.

### F2 — Scoring, budgets và truy vết passage cuối

1. Thêm explicit status: `reranked`, `not_reranked_budget`, `retrieval_only`, `failed`. Chỉ nhận scores hữu hạn; sai số lượng, NaN/Inf hoặc exception phải fail query và giữ checkpoint để retry.
2. Cache theo query text/ID, passage/parent identity, title/text hash, model/tokenizer revisions, max lengths, precision và formatting. Đổi selection có thể replay scores; đổi pair input phải score lại pair đó.
3. Ghi từng tầng: số doc hits, passage hits, fused items, unique parents, parents có evidence, shortlisted pairs, scored pairs và selected/exported counts. Reasons ghi theo từng quyết định, tránh gọi mọi giới hạn là top-K document.
4. Đo token lengths của title+passage và query+title+passage bằng tokenizer thật; ghi phần bị cắt theo truncation policy thực. Tách evidence ở passage cuối với evidence ở cuối một model input bị truncate.
5. Khóa stable tie-break theo parent/passage identity; bổ sung validation dimension, số vector rows và finite values tại encoder/index boundary.
6. Fixture bắt buộc có ít nhất 8 passages, title và 7 passages đầu không có keyword; passage cuối chứa evidence. Chạy lexical path và dense path độc lập, rồi fusion/rerank/selection/API/CLI/export. CPU doubles kiểm wiring; real-model smoke ghi riêng.

Gate `V4_EVIDENCE_CONTRACT_PASS`: fixture tail đi qua đủ đường; statuses/caps/truncation quan sát được; cache sai fingerprint bị từ chối; lỗi inference không biến thành prediction rỗng.

### F3 — Đo phân bổ rerank và số evidence trên subset

Giữ corpus, candidate identities và tổng budget cặp để so riêng thay đổi allocation.

| Arm | Allocation | Selection | Mục đích |
|---|---|---|---|
| A0 | Theo doc order hiện tại, cap 3/doc, tổng 200 | Explicit cap 1/doc | Reference sau sửa correctness |
| A1 | Parent-first: 1 pair cho tối đa 200 parents ưu tiên có evidence, rồi dùng slots dư cho passage thứ 2/3 | Như A0 | Đo độ đa dạng parent ở cùng compute |
| A2 | Như arm allocation đã khóa | Cap 2/doc, giữ nguyên max_docs/max_chunks | Đo multi-evidence và tác động cap tổng |

Nếu có 400 parents và budget 200 pairs, không thể bảo đảm score mọi parent. Phải ghi rõ 200 parents nào được ưu tiên và số không được score. Tăng budget lên 400 là arm riêng; không gộp gain compute vào gain allocation.

- Dùng 20 official queries để kiểm integration/replay; không dùng random subset này để kết luận chất lượng toàn corpus.
- Chạy regression trên 107 qrels cũ khi corpus chứa các source/span được gán nhãn, reconciliation được IDs/offsets. Tập này đã qua nhiều vòng sử dụng: là dev/regression, không phải holdout mới.
- Đo evidence-span coverage, unique-parent coverage, positive survival trước/sau rerank/selector, output counts và latency. Nếu chưa có official scorer/gold, F2 chính thức vẫn `UNMEASURED`; metric proxy ghi riêng.
- Chọn A0 làm baseline nếu A1/A2 chưa có bằng chứng đủ. Tăng cap không tự chứng minh tăng relevance. Parent-level RRF, max/top-2 aggregation mới và thresholds là các ablation riêng, không bắt buộc cho correctness.

Đầu ra: `allocation_ablation.json`, `stage_survival.parquet`, `selection_config.json`, báo cáo regression và quyết định giữ/chuyển arm.

### F4 — Backend toàn corpus trong Task 4 giữ contracts đã sửa

1. Đưa backend interfaces vào V4 factory/Pipeline: sparse doc full-body search, sparse passage search, dense passage search, parent lookup và evidence store. API/CLI/batch dùng chung orchestration. Không đơn thuần bỏ giới hạn 1.000 trong loader in-memory.
2. Corpus store và builders đọc release shards theo batch; verify linkage/checksums, PASS allowlist, semantic exclusions, official ID/URL, normalized-text hashes và offsets. Dùng lookup trên disk, không đưa toàn bộ text/vectors thành Python objects.
3. Reconcile mục tiêu dự kiến: 1.073.525 eligible docs, 2.951.224 eligible passages sau 6 docs/49 passages exclusions. Xuất anti-join của official inventory; không gọi eligible content là đủ 1.220.598 official URLs.
4. Doc BM25 dùng title + full normalized body; passage BM25 index mọi eligible passage. Khóa tokenizer và công thức hiện tại; V4 hiện tương ứng k1=1,2, b=0,75. Backend phân shard phải dùng global DF/IDF/average length và merge top-K đúng. Backend đổi scoring semantics được ghi thành arm khác.
5. Dense lưu vector cho mọi eligible passage, title prefix và max length theo config đã khóa. Lookup row ↔ passage_id/parent/text hash là một-một. Có thể tạo shards và query theo block/memmap trước; exact dense là reference để đo ANN nếu thêm ANN.
6. Giữ fusion hiện tại ở reference backend trước khi đổi thuật toán: mixed doc/passage keys, budgets và cách tổng hợp score phải được mô tả đúng. Parent-level fusion là thay đổi ranking cần so riêng, không coi là backend parity.
7. Thay fallback quét toàn corpus bằng parent-filtered passage search/lookup, vẫn tìm theo relevance và chỉ dùng passages thuộc parent đó. Parent không có evidence hợp lệ được trace, không được backfill tùy ý.
8. Production config explicit bật dense/reranker, revisions pin, index fingerprint và selection policy. Preflight đối chiếu stages thực tế; không sửa mặc định subset chỉ để che config thiếu.
9. Checkpoint theo partition và query; complete manifest chỉ publish sau integrity checks. Resume phải cùng fingerprints, không duplicate/missing rows và không encode lại partition complete.
10. Canary dùng cùng 1.000-doc snapshot và đủ 20 queries: backend exact mới phải tương đương **V4 reference đã sửa** trong tolerance khóa trước. Không yêu cầu bằng `r2ai`; không kỳ vọng result của full corpus bằng result subset.

Gate `V4_FULL_BACKEND_PASS`: coverage/mapping/checksums/global sparse stats, subset parity, tie-break và resume PASS; benchmark pilot có số đo RSS/VRAM/disk/throughput trước khi mở full query inference.

### F5 — Chạy đủ 1.200 queries và bàn giao

- Chỉ chạy sau gates F1/F2 và F4; F3 chốt arm baseline có thể là A0 nếu arm tối ưu chưa đủ bằng chứng.
- Ghi candidates/pair scores/selection decisions theo query và cache có thể replay. Query failed/pending phải retry hoặc báo thiếu; không đánh dấu complete khi chưa đủ 1.200 IDs.
- Grounding theo eligible Phase 2 rows và exact normalized source slices; strict validation gồm selected/exported equality, query/parent IDs, caps, schema, ZIP CRC/entries và canonical payload SHA.
- Ghi code/config/index/selection/ZIP hashes trong run manifest. Lần rerun cùng cache phải cho cùng canonical payload.
- Kết thúc kỹ thuật là `FULL_V4_EXPORT_PASS / SCORE_PENDING`. Chỉ ghi F2 và `FULL_V4_SCORED` khi có scorer/feedback BTC của đúng ZIP/hash/phase. Lượt nộp BTC là bước riêng theo yêu cầu của người dùng.

## 5. Tái sử dụng artifact và xử lý job đang chạy

| Artifact/stage đã có | Quyết định |
|---|---|
| Phase 1/2 releases, V4 Task 3 snapshot | Giữ nguyên; làm input và reference |
| Passage store có IDs/offsets/hashes đúng | Tái sử dụng sau kiểm fingerprint/coverage |
| Dense passage vectors đúng title/text/model/precision contract | Tái sử dụng sau kiểm row mapping và checksums; không hứa dùng lại chỉ dựa vào tên file |
| Sparse full-body/full-passage index | Tái sử dụng nếu tokenizer/global statistics/scoring khớp |
| Document index 2/6 lead passages | Lưu làm legacy baseline; không làm gate bắt buộc cho V4 full |
| Pair-score cache | Chỉ tái dùng cặp cùng input/model/tokenization; thiếu pairs thì score phần thiếu |
| Predictions/ZIP tạo bằng adapter cũ | Giữ baseline; replay selector/export từ score cache nếu cache đủ, không bắt buộc rerun embeddings |

Khi triển khai F0, job build store/embedding đúng contract có thể tiếp tục. Job query inference hoặc job đóng gói cuối phải dùng config/code đã khóa; thay code đang được workers sử dụng chỉ sau checkpoint và chuyển run namespace có kiểm soát. Kế hoạch này chưa yêu cầu dừng tiến trình nào.

## 6. File dự kiến và thứ tự giao việc

| Gói | File sửa/thêm dự kiến | Phụ thuộc |
|---|---|---|
| F0 | Script khóa reference + manifest trong namespace mới | Input Task 3 |
| F1 | `core/contracts.py`, `evidence/selector.py` mới, adapter, API/CLI, config/validator V4 | F0 |
| F2 | `core/pipeline.py`, model adapters, score-cache/diagnostics helpers, tests V4 | F0; nối F1 |
| F3 | Script regression/ablation và config arms | F1–F2 |
| F4 | Backend/store interfaces, disk-backed sparse/dense implementation, index manifests, streaming loader/factory | Contracts F1–F2 |
| F5 | Full-query runner, cache replay, validator/export và run report | F4 PASS; arm F3 đã khóa |

Tên file mới là đề xuất; khi triển khai ưu tiên ghép với module tương ứng đã có và giữ một V4 runtime. Không tạo runner phụ đi qua `r2ai`.

Ước lượng lập kế hoạch: F0–F2 khoảng 1–2 ngày công; F3 khoảng 0,5–1 ngày; F4 khoảng 2–4 ngày tùy backend đã có; F5 khoảng 0,5–1 ngày kỹ thuật. Runtime inference/index và chờ BTC tính riêng sau benchmark. Không cần chờ hoàn tất mọi tuning ở F3 để phát triển F4.

## 7. Bộ kiểm thử và checklist nghiệm thu

1. Keyword chỉ có ở passage thứ 8/cuối: lexical-only, dense-only, fused, reranker, selector, CLI/API và export đều giữ đúng identity/text khi budgets fixture đủ.
2. Hai passages được selector chọn trong một parent đi qua serializer đầy đủ; fixture một passage cũng giữ nguyên output.
3. Hai parents có text giống nhau vẫn tồn tại; duplicate identity do fusion chỉ xuất một lần.
4. Budget nhỏ: trace đúng parents/passages chưa rerank; không tạo score giả; NaN/Inf/wrong count/error không thành output thành công.
5. Title-only document match không có passage match được xử lý đúng policy; fallback không dùng passage từ parent khác.
6. Cache mismatch input/title/model/revision/max length/row mapping bị từ chối; đổi selector hợp lệ replay không inference lại.
7. Đảo row/shard order và tie-score không tráo identity; subset exact backend parity trong tolerance đã khóa.
8. Negative fixtures: excluded IDs, sai URL/hash/offset, missing shard, unsupported config hoặc model/index mismatch fail trước complete.
9. Truncation measurement dùng đúng tokenization/pair policy; text evidence lưu/xuất vẫn giữ nguyên.
10. API/CLI parity so ở cùng limit/config; strict validator kiểm query coverage và exported = selected. Full query errors không được ngụy trang bằng empty predictions.

Chuẩn bị môi trường test trước khi chạy: lượt kiểm tra trước thấy thiếu `bs4` ở môi trường model và thiếu `pytest` ở `.venv-phase2-batch`. Dùng một môi trường có đủ project/API/model dependencies; ghi kết quả CPU contract tests, real-model canary và full backend tests riêng. Không coi mock tests là bằng chứng inference thật.

Chỉ sau F1/F2 đạt mới gọi sửa correctness hoàn tất; chỉ sau F4 đạt mới gọi backend sẵn sàng full; chỉ sau F5 và feedback đúng artifact mới kết luận chất lượng/F2.

## 8. Căn cứ tại repo

- `docs/task3_v4_workflow_integration_result_20261016_vi.md`.
- `outputs/v4/phase2_subset/20261016/selection_seed20261016/manifest.json`.
- `src/medrag/retrieval/core.py`, `src/medrag/core/pipeline.py`, `src/medrag/core/factory.py`.
- `src/medrag/adapters/vibiomir_submission.py`, `src/medrag/api.py`, `src/medrag/cli.py`.
- `configs/v4_core.yaml`, `configs/v4_phase2_subset.yaml`.
- `docs/full_index_baseline_submission_task3_task4_plan_20261016_vi.md`: phần legacy workflow/parity đã bị tài liệu này thay thế.
- `docs/candidate_retrieval_evidence_ranking_f2_plan_20261016_vi.md`: nguồn thiết kế ablation; mô tả cũ rerank `evidence[:1]` đã lỗi thời, code hiện cap 3/doc.

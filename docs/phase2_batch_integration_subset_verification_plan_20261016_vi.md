# Plan Task 1–2: Phase 2 release → batch pipeline → subset verification

Ngày lập: 16/10/2026. Trạng thái: Task 1 đã triển khai; Task 2 đã chạy real-model subset và repeatability PASS. F2 vẫn chưa đo.

## 1. Mục tiêu và ranh giới

- Task 1: tích hợp release Phase 1/2 đã nghiệm thu vào một đường batch có thể chạy retrieval, ranking, selection và export; bảo toàn passage identity và provenance.
- Task 2: xác nhận đường chạy đó bằng kiểm thử logic và inference model thật trên subset được khóa trước khi chạy.
- Kết quả nghiệm thu là tính đúng đắn tích hợp, không phải F2 tăng hay full-corpus readiness.
- Chưa xây index toàn corpus, chưa chạy 1.200 query trên toàn corpus, chưa nộp BTC, chưa đổi API/CLI mặc định và chưa thêm V5.
- Dữ liệu Phase 0/1/2, index/cache/submission lịch sử là đầu vào chỉ đọc. Output mới dùng namespace riêng.

## 2. Căn cứ hiện tại và đầu vào phải khóa

| Thành phần | Artifact hiện có |
|---|---|
| Phase 1 release | `outputs/vibiomir_phase1_full_20261014_v1/release/phase1_release_manifest.json` |
| Phase 1 documents | Các part trong candidate directory được Phase 1 release tham chiếu; chỉ lấy IDs thuộc PASS allowlist |
| Phase 2 release | `outputs/vibiomir_phase2_full_v2/manifests/phase2_release_manifest.json` |
| Phase 2 passages | 922 part được release liệt kê; tổng 2.951.273 rows, 1.073.531 documents trước semantic staging exclusions |
| Phase 2 validation | `outputs/vibiomir_phase2_full_v2/validation/full_validation_report.json` |
| Semantic exclusions | `configs/vibiomir/phase2_semantic_staging_20261015.json`: 6 document IDs |
| Official corpus/query | `outputs/vibiomir_official_verify_20261012/links_corpus_vi_en_zh.parquet` và `query.parquet`, đối chiếu revision/hash với release/registry |
| Ranking/selection/IO tái sử dụng | `r2ai/rank.py`, `evidence_selector.py`, `evidence_io.py`, `select_evidence.py`, `validate.py` |

Hai chỗ không tương thích đã xác nhận trong code:

1. V4 `load_prepared()` tự chia lại source và chỉ hỗ trợ snapshot nhỏ; không nạp release passages Phase 2.
2. `r2ai` dùng `doc_id/chunk_index/text/title`, cache chưa giữ trực tiếp `passage_id` và đầy đủ offsets/hashes. Một số đường dẫn query/corpus/model loading còn gắn với dữ liệu lịch sử.

Điểm provenance quan trọng: Phase 2 lưu `text_representation=normalized_text`. Offsets trỏ vào `normalized_text`; `text_representation_sha256` và `source_text_sha256` có ý nghĩa khác nhau. Không thay hai hash cho nhau.

## 3. Workflow được tích hợp

Manifest preflight → PASS/staging/official identity filtering → release-backed batch dataset → BM25/dense document candidates → RRF → dense passage shortlist trong parent pool → BGE passage reranking → document ranking → selector → provenance validation → ZIP/schema validation.

Đây là luồng migration để tái lập hành vi các thành phần cũ trên dữ liệu mới, chưa phải thí nghiệm tối ưu retrieval. Passage shortlist hybrid hoặc parent-allocation mới nằm ở vòng sau.

## 4. Task 1 — Tích hợp release vào đường batch

### T1.1 — Khóa input contract và preflight

- Tạo cấu hình batch riêng, khai báo đường dẫn rõ ràng cho Phase 1/2 release, official corpus/query, staging, model revisions và output namespace.
- Kiểm status/schema/revision, liên kết SHA-256 Phase 2 → Phase 1, frozen Strategy C và tokenizer revision; kiểm manifest/validation/config hashes trước xử lý.
- Đọc shard theo manifest, không dùng glob để tự nhận part ngoài release. Hash các shard thực sự sử dụng trước khi tiêu thụ, rồi ghi phạm vi kiểm chứng; không đánh dấu đã verify toàn corpus khi mới đọc subset.
- Kiểm document PASS allowlist, excluded IDs, official `doc_id + URL`; không kiểm ID bằng khoảng 1..N vì official IDs không nhất thiết liên tục.
- Tách số liệu trước/sau exclusions. Tính số passage thực tế bị loại; không suy từ số document.
- Bất nhất input, shard thiếu/hỏng, sai identity hoặc sai checksum phải dừng và không công bố output hoàn tất.

Đầu ra: `input_manifest.json`, `preflight_report.json`, `exclusion_ledger.parquet`.

### T1.2 — Release-backed loader và compatibility adapter

- Đọc Parquet theo batch/part; join metadata Phase 1 bằng `doc_id`. Không nạp toàn bộ text corpus bằng `read_table(...).to_pylist()`; dùng shard/row-group lookup hoặc disk-backed lookup khi cần.
- Không rechunk, normalize lại, ghép passage hoặc cấp lại passage IDs.
- Ánh xạ:

| Dữ liệu batch/legacy | Nguồn Phase 2/Phase 1 |
|---|---|
| `doc_id` | Phase 2 `doc_id`, đối chiếu official mapping |
| `chunk_index` | Phase 2 `passage_index`, giữ nguyên |
| `text` | Phase 2 `passage_text`, giữ nguyên |
| `title` | Phase 1 `title`; thiếu thì giữ rỗng, không tự sinh |
| Provenance sidecar | `passage_id`, URL, offsets, hai lớp source/representation hashes, revision, strategy và release hash |

- Khóa identity trong batch theo `(release_sha256, doc_id, passage_index)` và ánh xạ một-một tới `passage_id`.
- Index/vector row positions chỉ là lookup nội bộ, không được trở thành document/passage identity.
- Giữ provenance sidecar riêng và gắn hash vào cache/run manifest. Không thay schema cache lịch sử chỉ để chứa thêm trường.
- Hai documents có text giống nhau vẫn giữ hai official IDs. Không áp canonical/dedup Phase 3 vào dataset trong migration này; selector có thể giữ hành vi dedup cũ và phải log quyết định đó.

Đầu ra: batch dataset manifest, lookup/index-row mapping và `passage_provenance.parquet`.

### T1.3 — Nối runner với retrieval/ranking có sẵn

- Thêm entrypoint batch riêng; bắt buộc khai báo scope/subset và output. Task 1–2 chỉ chạy scope `subset`; không mặc định chuyển sang full corpus.
- Tái dùng các hàm tokenization/fusion/ranking phù hợp. Tách input loading khỏi module legacy có đường dẫn hard-code; không sửa global paths bằng monkeypatch và không ghi đè file legacy để giả tương thích.
- Cả query IDs, URL/language mapping và validator đều nhận explicit official paths. Nếu giữ language-balance từ pipeline cũ, statistics và mapping phải được tính từ đúng subset mới và ghi vào manifest.
- Pin cả dense-document model loading và passage/reranker loading; không chỉ pin model trong `rank.py` rồi bỏ quên `index.py`.
- Chấm các passage thuộc shortlist bằng model thật, ghi raw logits và trạng thái `not_reranked` rõ ràng. Không dùng score giả/null để vượt qua threshold.
- Lưu đầy đủ provenance thông qua dataset sidecar và cache metadata; fingerprint bao gồm row order, text/title format, model/tokenizer revision, precision, max lengths và retrieval settings.
- Cache sai fingerprint phải bị từ chối. Có thể replay selection CPU từ cache hợp lệ.

### T1.4 — Khóa cấu hình migration

Cấu hình khởi đầu tái dùng tham số của pipeline lịch sử; đây không phải cấu hình đã được chứng minh tối ưu trên release mới:

| Thành phần | Cấu hình migration |
|---|---|
| Document BM25 representation | Title + tối đa 6 passages đầu theo index, ghi rõ representation |
| Dense document representation | Title + tối đa 2 passages đầu; max length 256 như run lịch sử |
| Dense passage encoder | BGE-M3 revision hiện pin; title prefix; normalized vectors; max length 512 |
| Document candidates | Mỗi nhánh tối đa 400; RRF weights 1:1, k=60; fused tối đa 400 |
| Passage shortlist/rerank | Tối đa 200 passage/query trong các parent candidates |
| Reranker | BGE-reranker-v2-m3 revision hiện pin; raw logits; max length 512 |
| Document ranking | Theo passage đứng cao nhất như `r2ai`; chưa thêm learned aggregation |
| Selector | Bản threshold -2 đã nối feedback lịch sử: max_docs 200, max_chunks 60, cap 1/doc, document-order; doc threshold hiệu dụng -2 |
| Optional stages | Taxonomy/ICD/intent/expansion/multi-vector tắt |

Pool thực tế có thể nhỏ hơn caps; log số thực, không bù bằng candidates trùng. Ghi precision/device thực tế cho mỗi model. Ngưỡng -2 chỉ là cấu hình đối chứng migration, chưa được hiệu chỉnh cho dữ liệu mới. Nhánh dense/fusion dùng score semantics đúng với cấu hình đã khóa; mọi khác biệt với lịch sử phải xuất hiện trong manifest.

### T1.5 — Grounding, export và diagnostic artifacts

- Trước export, mỗi evidence phải ánh xạ được về đúng Phase 2 row và thỏa `normalized_text[char_start:char_end] == passage_text`; kiểm hash của representation và source lineage riêng biệt.
- Title chỉ tham gia model input; evidence xuất ra giữ nguyên `passage_text`.
- ZIP chỉ chứa schema submission; provenance/metadata nằm ngoài ZIP ở sidecar. Không đưa trường nội bộ vào schema BTC.
- Xuất đủ query IDs được chỉ định trong subset, kể cả prediction rỗng; không che lỗi pipeline bằng tự tạo output rỗng khi input hỏng.
- Validator được truyền explicit official corpus và query subset; không dùng mặc định historical corpus hoặc `--no-strict-ids` để đạt PASS.
- Công bố output bằng atomic completion manifest sau mọi check. Chặn ghi đè run có sẵn.

### Các file dự kiến thay đổi/thêm

Các tên sau là đề xuất trước khi triển khai:

- `src/medrag/data/batch_dataset.py`: release loader, identity/provenance lookup.
- `src/medrag/batch.py`: orchestration, cấu hình và run manifest.
- `scripts/workflows/vibiomir/run_phase2_batch.py`: entrypoint batch rõ input/scope/output.
- `configs/vibiomir/phase2_batch_migration_v1.json`: hợp đồng và tham số migration.
- Các helper nhỏ trong `r2ai` chỉ sửa khi cần explicit inputs hoặc reuse hàm; bảo toàn CLI/hành vi mặc định lịch sử.
- Tests cho batch dataset, integration, cache/provenance và subset export.

### Gate hoàn tất Task 1

- Có đường chạy release → batch → selection/export, không gọi chunker để tạo lại passages.
- Identity/text/offsets/hash giữ nguyên và truy ngược được; loại đúng held/staging documents.
- Explicit official mapping/query paths và model revisions.
- Negative tests của contract chạy PASS; chưa đánh dấu real-model end-to-end PASS nếu mới dùng mock.
- Có hướng dẫn chạy Task 2; source release/index/submission cũ không bị thay đổi.

## 5. Task 2 — Kiểm chứng end-to-end trên subset

### T2.1 — Khóa subset trước inference

- Mục tiêu ban đầu: 1.000 eligible documents, 20 official queries; lấy toàn bộ Phase 2 passages của mỗi document đã chọn. Không cap cắt mất phần cuối document chỉ để giảm runtime.
- Seed cố định, lưu IDs và hashes; lấy mẫu theo ngôn ngữ nguồn VI/EN/ZH và độ dài khi eligible data có đủ. Nếu thiếu nhóm, báo thiếu, không tự sinh query/tài liệu để đủ phân tầng.
- Chỉ lấy query IDs có trong official query snapshot. Query EN/ZH tự tạo ở pilot trước không được tự nhập vào tập official này.
- Có thể bổ sung parent documents của query có qrels đã kiểm tra để kiểm đường truy vết/known-positive, kèm distractors và cờ `positive_enriched=true`. Subset kiểu này không dùng để tuyên bố chất lượng retrieval đại diện corpus.
- Bộ negative fixtures riêng kiểm held IDs, semantic exclusions và malformed provenance; không đưa dữ liệu lỗi vào healthy smoke run.

Đầu ra: `subset_manifest.json`, `query_subset.parquet`, danh sách doc IDs và selection rationale.

### T2.2 — Kiểm thử logic trước model thật

Kiểm các điều kiện có thể làm sai output:

1. Phase 1/2 release không khớp; shard thiếu/hỏng; sai hash hoặc sai corpus revision.
2. PASS/held/staging filtering; official ID/URL mismatch; ID lớn nhưng hợp lệ vẫn được giữ.
3. Source text khác normalized text: offsets phải được kiểm theo representation đúng; tráo hai hash phải bị phát hiện.
4. Duplicate content ở hai parent IDs không bị nhập làm một document ở loader; row-order thay đổi không tráo vector/passage identity.
5. Cache sai model/input/row mapping bị từ chối; selector replay giữ đúng payload.
6. Nhiều passages/doc được score và giữ trong cache; export phản ánh đúng selector, không bị cắt ngầm `evidence[:1]`.
7. Prediction rỗng hợp lệ khác lỗi pipeline; schema/parent consistency/caps và query coverage đều được kiểm.
8. Chạy lại không ghi đè output hoàn tất hoặc sửa input.

Mock encoder/reranker chỉ dùng kiểm logic, báo riêng `contract_tests`; không dùng làm bằng chứng inference model thật.

### T2.3 — Chạy model thật

- Kiểm dependency, local model snapshots/revisions, RAM/VRAM và GPU đang sử dụng trước run. Không khởi động full build hay tự chiếm các GPU khác.
- Tạo BM25/dense indexes riêng cho subset; chạy đủ document retrieval → fusion → passage shortlist → BGE reranker → selection → ZIP.
- Ghi score hữu hạn, số passage thực được rerank, truncation/query+title+passage lengths và time/peak memory theo stage.
- `max_length=512` được ghi là model truncation policy, không làm thay đổi text evidence hoặc identity lưu trong release.
- Chạy lại từ cùng subset/config; stable tie-break theo identity. So candidate membership/order và canonical prediction payload; score dùng tolerance được ghi theo backend/precision. Nếu output khác, báo nguyên nhân hoặc FAIL determinism thay vì bỏ qua.
- Không yêu cầu ZIP bytes tự trùng nếu timestamp metadata khác; dùng canonical payload SHA-256 để kiểm kết quả và vẫn lưu SHA-256 của từng ZIP thực tế.
- Replay selector từ score cache bằng CPU và yêu cầu payload khớp run inference.
- Model thật chưa chạy được thì báo `REAL_MODEL_BLOCKED/NOT_RUN` cùng nguyên nhân; không chuyển PASS từ mock sang model stage.

### T2.4 — Validation và báo cáo

Yêu cầu healthy subset run:

| Check | Điều kiện nghiệm thu |
|---|---|
| Input integrity | PASS trên toàn bộ shard/records thực sự tiêu thụ; ghi rõ subset scope |
| Eligibility | 0 held/staging/unknown-official documents lọt vào dataset |
| Identity/grounding | 100% evidence xuất ra khớp Phase 2 row, parent và normalized source slice |
| Query coverage | Đúng 20 IDs đã khóa, không thiếu/trùng/thừa |
| Real models | Dense và reranker chạy thật, đúng revision; scores hợp lệ |
| Cache/replay | Cache fingerprint đúng; CPU replay khớp canonical payload |
| Output | ZIP CRC/JSON/schema, strict official IDs, parent consistency và caps PASS |
| Repeatability | Run lặp khớp output theo protocol đã ghi; các lệch số học được giải thích |
| Isolation | Output riêng; các input release/config/index lịch sử không bị ghi đè |

F2 chính thức của subset ghi `UNMEASURED`. Nếu tính known-positive metrics thì đặt ở mục diagnostic riêng, ghi rõ phạm vi nhãn và không dùng chúng làm tiêu chí PASS/FAIL tích hợp.

Đầu ra dự kiến dưới `outputs/v4/phase2_batch_subset_<run_id>/`:

```text
input_manifest.json
preflight_report.json
subset_manifest.json
query_subset.parquet
dataset/                  # lookup, provenance, eligibility/exclusions
indexes/                  # subset-only indexes và row mappings
rankings.jsonl.gz          # score cache + provenance fingerprint
selection_config.json
diagnostics.jsonl
submission.zip            # subset, không phải ZIP đầy đủ 1.200 query
validation_report.json
run_manifest.json
SUBSET_INTEGRATION_REPORT.md
```

Đường dẫn trên là output dự kiến, không phải artifact đã tạo.

## 6. Thứ tự và khối lượng triển khai

1. T1.1–T1.2: input contract, loader và provenance adapter.
2. T1.3–T1.5: runner, cache, selection/export và explicit validator inputs.
3. T2.1–T2.2: subset đã khóa và focused contract/integration tests.
4. T2.3–T2.4: real-model run, repeat/replay, validation và báo cáo.

Ước lượng lập kế hoạch: Task 1 khoảng 2–3 ngày công, Task 2 khoảng 1–2 ngày công. Đây không phải thời gian inference được benchmark; dependency/model availability và thời gian đọc/hash shard sẽ được ghi riêng. Không lấy tốc độ subset làm cam kết throughput toàn corpus.

Sau Task 2, quyết định bàn giao là `SUBSET_INTEGRATION_PASS` hoặc `FAIL/BLOCKED` với hạng mục cụ thể. Chỉ khi PASS mới chuyển sang task xây index corpus được khóa và baseline submission đủ 1.200 query; chưa suy ra tăng F2 hoặc khả năng phục vụ toàn corpus từ kết quả subset.

## 7. Kết quả triển khai Task 1 (2026-10-16)

Đã thêm các thành phần sau, tách khỏi CLI/default workflow lịch sử:

- `src/medrag/data/phase2_batch.py`: preflight release, chọn explicit IDs, đối chiếu PASS allowlist và semantic exclusions, official ID/URL, kiểm tra hash nguồn/representation, offsets và passage ID; ghi batch tương thích `r2ai` cùng provenance, preflight và exclusion ledger.
- `src/medrag/data/phase2_batch_runner.py` và `scripts/workflows/vibiomir/run_phase2_batch.py`: orchestration BM25 Lucene + dense document retrieval → RRF → dense passage shortlist → reranker → selector → ZIP và strict validator. Runner từ chối trên 1.000 document, yêu cầu IDs explicit và không ghi đè output đã có.
- `configs/vibiomir/phase2_batch_migration_v1.json`: input hashes/paths, model revisions, migration settings và selector threshold -2 đã ghi rõ; cấu hình này là baseline migration, không phải kết quả tối ưu đã xác nhận.

Lệnh thực tế đã dùng sau khi khóa IDs cho Task 2 (trong môi trường `.venv-phase2-batch`):

```bash
CUDA_VISIBLE_DEVICES=2 HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 \
  .venv-phase2-batch/bin/python scripts/workflows/vibiomir/run_phase2_batch.py \
  --doc-ids outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/doc_ids.json \
  --query-ids outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/query_ids.json \
  --subset-manifest outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/subset_manifest.json \
  --output outputs/v4/phase2_batch_subset/20261016/run_01 \
  --device cuda:0 --batch-size 16
```

`doc_ids.json` và `query_ids.json` là JSON arrays chứa integer IDs chính thức, không trùng lặp. Runner tạo thư mục output mới; nếu lỗi giữa chừng, thư mục `.incomplete-*` được giữ lại để chẩn đoán và không được đánh dấu `complete`.

Task 1 kết thúc với implementation ready for Task 2; kết quả nghiệm thu được ghi ở mục dưới.

## 8. Kết quả Task 2 (2026-10-16)

### Subset và integrity

- Khóa trước inference 1.000 documents và 20 official queries, seed `20261016`; lấy mẫu theo ngôn ngữ và tứ phân vị độ dài nguồn. Đây là sample không làm giàu positive (`positive_enriched=false`), lấy 1.000 trong 1.073.525 eligible documents sau khi loại 6 semantic-staging IDs khỏi 1.073.531 Phase 1 PASS documents.
- Hash doc IDs: `1cd8fdff5fe4ea89c5f81ba83d649b33c0a330936920efa6e94e635138ca6637`; query IDs: `bee4a58fa193ec70bfdf400f71e335acaef6d3b58e731d886447fa3794356570`.
- Loader preflight PASS: liên kết Phase 1/2 release, PASS allowlist, semantic exclusions, official ID/URL, normalized-text SHA, passage offsets/IDs và SHA của shard thực sự được đọc. Dataset gồm 2.743 passages từ 1.000 docs. Sáu exclusion tương ứng 49 passage rows đã được đối chiếu với mapping, không đưa vào healthy subset.
- Hai lần chạy dùng cùng input manifest SHA `ecb241cee8906383bd41339cd89097280dd44911a4ee8418ea26e080b3c75a08`, cùng subset/config, GPU vật lý 2 (`CUDA_VISIBLE_DEVICES=2`, runner `cuda:0`). Model revisions thực được resolve và ghi nhận: BGE-M3 `5617a9f61b028005a4858fdac845db406aefb181`; bge-reranker-v2-m3 `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`.

### Kết quả chạy và repeatability

| Run | Load/provenance (s) | Document retrieval (s) | Passage retrieval + rerank (s) | Reranked pairs | Peak VRAM allocated | Validation/cache replay |
|---|---:|---:|---:|---:|---:|---|
| `run_01` | 349.36 | 14.77 | 34.48 | 4.000 | 1.35 GB | PASS / PASS |
| `run_02` | 190.40 | 14.79 | 35.00 | 4.000 | 1.35 GB | PASS / PASS |

- Strict output validation PASS cho cả hai run: đúng 20/20 official query IDs, schema/ZIP và grounding của evidence được runner kiểm tra. Cache replay selector PASS.
- So sánh run PASS: thứ tự document và membership/order passage giống hệt; max absolute delta dense và reranker đều `0.0` (4.000 candidates được so). Canonical prediction payload SHA-256 ở cả hai run là `e09d6ad35e55d311258b586b6a42a42d4c52ba4028017bbac4a678c64a43ef14`.
- File ZIP nguyên byte khác nhau (SHA run 1 `5f0f0d351bb3858229bffad30b1a2a20f9d9ca9d5fc7cbd69b53ad95f8e635bc`; run 2 `a1b21bb4283dc1d298a16e373695a26ec017ffaf081c146b67341a8dc1f11505`), nhưng canonical JSON payload trong ZIP trùng; vì vậy repeatability được đánh giá trên payload chuẩn hóa, không đòi ZIP container bytes giống nhau.
- Mỗi run chọn trung bình 3 docs/chunks mỗi query và có 6/20 query rỗng. Đây chỉ là mô tả output trên corpus bị giới hạn ở 1.000 docs, không phải precision/recall hay F2.

### Kết luận và ranh giới

Task 2 đạt `SUBSET_INTEGRATION_PASS` cho subset đã khóa: real dense/reranker inference chạy thành công, integrity/provenance và strict export validation PASS, replay và hai lần inference lặp lại chính xác ở ranking/prediction payload. Đây là xác nhận chức năng trên subset, không phải bằng chứng rằng retrieval quality/F2 sẽ giữ nguyên trên 1.2M URLs.

F2 vẫn `UNMEASURED`: chưa chạy BTC scorer/qrels và chưa tạo/nộp submission đầy đủ 1.200 queries. Chưa có full-corpus index hoặc full-corpus inference/throughput benchmark. Không dùng tỷ lệ query rỗng 6/20 để ngoại suy vì candidates chỉ được tìm trong sample 1.000 documents. Bước tiếp theo để biết điểm thật là chạy pipeline trên corpus đầy đủ và tạo submission ZIP cho BTC; nếu cần phép đo có nhãn trước đó, phải dùng qrels/evaluator hợp lệ, không lấy subset random này làm thay thế.

Artifacts chính: `outputs/v4/phase2_batch_subset/20261016/selection_seed20261016/`, `run_01/`, `run_02/`, `repeatability_report.json` và `SUBSET_INTEGRATION_REPORT.md`. Một attempt đầu chưa hoàn tất được giữ ở `.run_01.incomplete-*`; attempt đó dừng ở bước công bố output do model loader trả `_commit_hash=null`, sau đó đã sửa cơ chế resolve revision và cả hai run hoàn chỉnh đều xác thực đúng commit.

## 9. Kế hoạch tiếp nối Task 3–4

Chi tiết ở [Plan full index và baseline submission Task 3–4](full_index_baseline_submission_task3_task4_plan_20261016_vi.md): Task 3 khóa coverage/build index toàn eligible release; Task 4 chạy 1.200 official queries, validation/export và nối score BTC với đúng ZIP SHA. Đây là kế hoạch bước tiếp theo, chưa có full build hoặc score mới từ hai task này.

# B0 Cached-First Challenger — Candidate Delta & End-to-End Readiness

**Ngày audit:** 2026-10-22  
**Phạm vi:** sáu diagnostic queries `53, 130, 380, 754, 771, 1144`  
**Chế độ:** frozen-artifact replay, targeted read-only lookup, không retrieval/model/GPU

## Kết luận

B0 có **architectural potential ở passage choice**, nhưng chưa có bằng chứng độc lập rằng potential đó cải thiện relevance hoặc official F2. Kết quả quan trọng nhất của cached-first audit là:

- P4 count ban đầu được xác nhận: 6 queries, 1.200 B0 pairs, 349 pairs không nằm trong A0 parent-first pool.
- `349` không phải true missing-score count. Exact lookup qua các cache cùng scoring contract tìm được thêm 248 scores; B0 thực sự chỉ thiếu **101/1.200** scores.
- Candidate overlap là `851/1.200` passages. Mỗi arm có 349 passage riêng.
- A0 có `1.200` unique parents; B0 chỉ có `878`, tương đương `134–160/query`.
- Trong 349 B0-only passages, chỉ 27 passages mang 27 parents mới. **322 passages** là alternatives trong shared parents. Đây là nơi B0 có thể cải thiện passage choice, nhưng cũng là nguyên nhân làm giảm parent breadth.
- Có 57 within-parent alternatives với cached reranker score cao hơn passage A0; 31 thuộc parents đang ở A0 top-60. Tất cả vẫn là `UNKNOWN`, không phải positives.
- B0 source validation pass `1.200/1.200`; duplicate pairs `0`; incompatible score rows `0`.
- Independent relevance labels hiện có: `0`. Development/holdout split vẫn `unlabeled`.
- Final decision cho claim “outperform A0”: **`BLOCKED_BY_EVALUATION`**.

Không file A0 nào bị sửa; không attach/inspect/interrupt tmux và không chạy GPU inference.

## 1. Artifact discovery và compatibility

### Frozen identities

| Contract | Giá trị | Kết quả |
|---|---|---|
| A0 config | `configs/v4_submission_p2_a0.yaml` | SHA-256 `bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03`, khớp frozen lock |
| Corpus version | `official-71be0e1239a19dfd7a717e2bc13e3bec21bd0334` | Khớp config/index/cache |
| Full-index manifest | `outputs/v4/task4/full_index_gpu3_20261019/manifest.json` | SHA-256 `efdad2cf0e1db51730f7eaf0f931242e9bba9d5b0998910f6cb168f8dede6941`, status `complete`, scope `full_release` |
| Official queries | `outputs/vibiomir_official_verify_20261012/query.parquet` | SHA-256 `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08` |
| Reranker | `BAAI/bge-reranker-v2-m3` | revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, max length `512`, precision contract `fp16` |
| Pair format | — | `query_title_newline_passage-v1` |
| Score fingerprint | — | `42350936c327dc01630472257a4444a8df27713140b6fc7e518ba01ad4239c41` |

### Actual artifacts

- Frozen A0 candidate/query traces: `outputs/v4/parent_first_evidence_pilot_20261021_v2/queries/{qid}.json`.
- A0 pair allocation: `outputs/v4/p2_passage_candidate_replay_20261021/pair_allocation.csv`, SHA-256 `306b72586982837c2bedb63c50bf34e530adde74e0cf220c787c61da99ed0cb9`.
- B0 passage-first candidate replay: `outputs/v4/p4_b0_candidate_replay_20261021_v2/`; đây là bản chuẩn. `candidate_comparison.json` SHA-256 `94f367f97a0bd9a3e0bc5f8ac37987abe5c58d519007d05a0cf73d83521def10`.
- Parent/passage/source mapping: targeted rows trong `outputs/v4/task4/full_index_gpu3_20261019/corpus.sqlite`.
- Source validation: P4 manifest ghi 1.200 validated pairs; audit này xác minh lại exact parent, offsets và source hash cho 1.200/1.200.
- Retrieval-only probes có sẵn cho development queries `39, 74, 119`: `outputs/v4/p4_b0_full_index_retrieval_probe_20261022/results.json`. Không rerun.
- Historical anchors: `outputs/v4/p0_positive_path_replay_20261021_v3/`; chỉ dùng báo cáo diagnostic sau candidate construction.

### Score-cache inventory

Năm SQLite caches được tìm thấy đều có exact same scoring contract/fingerprint:

| Cache | Entries | Ghi chú |
|---|---:|---|
| `p2_missing_pair_scores_20261021/scores.sqlite` | 4.309 | Canonical seed cho B0 missing-score job |
| `p5_2_split_pilot_20261022/scores.sqlite` | 4.309 | Compatible |
| `p5_2_split_pilot_live_20261022/scores.sqlite` | 4.309 | Compatible |
| `parent_first_evidence_pilot_20261021/scores.sqlite` | 4.000 | Compatible |
| `parent_first_evidence_pilot_20261021_v2/scores.sqlite` | 4.303 | Compatible |

Không cache nào bị merge chỉ dựa trên `(query, doc, passage)`. Mỗi reuse yêu cầu exact `cache_key`, `query_sha256`, `official_id`, `passage_id`, `input_sha256`, contract fingerprint và finite score. Kết quả: `0` incompatible cache files, `0` incompatible pair rows, `0` conflicting exact scores.

### Independent relevance labels

- Primary 406-item template: `0` completed labels.
- Secondary 102-item template: `0` completed labels.
- Frozen 60-development/60-holdout split: `label_status=unlabeled`.
- Holdout labels không được mở hoặc dùng.

## 2. Candidate delta

### Per-query sets và parent breadth

| Query | A0-only | B0-only | Shared passages | A0 parents | B0 parents | B0 new parents | Shared parents có extra B0 passage | Median B0-minus-A0 fusion rank, shared pair |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 64 | 64 | 136 | 200 | 136 | 0 | 30 | +9 |
| 130 | 55 | 55 | 145 | 200 | 152 | 7 | 31 | -22 |
| 380 | 42 | 42 | 158 | 200 | 160 | 2 | 29 | -6.5 |
| 754 | 69 | 69 | 131 | 200 | 134 | 3 | 23 | +6 |
| 771 | 63 | 63 | 137 | 200 | 137 | 0 | 40 | +14 |
| 1144 | 56 | 56 | 144 | 200 | 159 | 15 | 30 | -26 |
| **Tổng** | **349** | **349** | **851** | **1.200** | **878** | **27** | **183 parent occurrences** | — |

Toàn bộ exact lists nằm trong `B0_A0_COMPARISON.json`. Không có shared parent nào mà A0 passage hoàn toàn vắng khỏi B0: `851/851` shared parents vẫn giữ A0 passage. Khác biệt nằm ở việc B0 cấp thêm 322 passage slots cho 183 shared parents và chỉ cấp 27 slots cho new parents.

Đây là distinction chính:

- **Parent-level:** B0 mất breadth lớn, `200 -> 134–160 parents/query`.
- **Passage-level:** B0 có depth lớn hơn trong shared parents, tạo cơ hội chọn passage tốt hơn sau rerank.

### Passage-level fusion và retrieval branch survival

B0 fuse trực tiếp `(doc_id, passage_id)` từ passage BM25 + dense. A0 fuse document/passage evidence theo parent rồi dùng fallback cho document-BM25-only parents.

Trên 349 A0-only passages:

- 94 passage-BM25-only;
- 20 dense-only;
- 235 không nằm trong top passage branches;
- 277 được đánh dấu fallback passage; fallback có thể đồng thời xuất hiện ở một top branch nên không phải partition độc lập.

Trên 349 B0-only passages:

- 61 nằm trong cả passage BM25 và dense;
- 90 passage-BM25-only;
- 198 dense-only;
- 0 fallback và 0 ngoài hai passage branches, theo đúng B0 construction.

Shared passage fusion ranks thay đổi mạnh theo query. B0 xếp shared pairs tốt hơn A0 ở 394 trường hợp, bằng ở 16 và thấp hơn ở 441. Rank movement không phải relevance judgment.

### Grounding

- B0 pairs: `1.200`.
- Parent/passage mapping mismatch: `0`.
- Source-invalid passage: `0`.
- Duplicate pair trong cùng query pool: `0`.

## 3. Offline quality analysis

Không có independent labels, vì vậy không candidate nào được gán `NEGATIVE` chỉ vì không nằm trong A0 hoặc historical positives. Default judgment là `UNKNOWN`.

### Within-parent alternatives

Có **57** shared parents mà một B0 alternative đã có cached reranker score cao hơn A0 passage; **31** thuộc A0 top-60 parents. Đây là hypotheses về passage choice, không phải evidence-quality labels.

Một số score deltas lớn nhất:

| Query/doc | A0 score | B0 alternative score | Delta | A0 top-60 parent | Judgment |
|---|---:|---:|---:|---|---|
| `1144/179256` | -1.7070 | 3.1289 | +4.8359 | no | UNKNOWN |
| `130/711026` | 0.6509 | 3.6797 | +3.0288 | no | UNKNOWN |
| `380/195043` | 0.4731 | 2.6914 | +2.2183 | yes | UNKNOWN |
| `53/672902` | 0.7280 | 2.6152 | +1.8872 | no | UNKNOWN |
| `1144/165047` | 2.2656 | 4.0430 | +1.7773 | no | UNKNOWN |
| `53/185694` | 2.9961 | 4.5664 | +1.5703 | yes | UNKNOWN |
| `53/185553` | 3.7910 | 5.2617 | +1.4707 | yes | UNKNOWN |
| `53/175788` | 4.3828 | 5.7500 | +1.3672 | yes | UNKNOWN |
| `130/673415` | 3.1836 | 4.4844 | +1.3008 | yes | UNKNOWN |
| `1144/712898` | 4.2031 | 5.5039 | +1.3008 | yes | UNKNOWN |

Danh sách đủ 57 cases, passage IDs, fused ranks và score deltas nằm trong `B0_A0_COMPARISON.json`.

### Historical-positive diagnostic only

A0 pool chứa 33/37 historical parents; B0 pool chứa 32/37. B0 không thêm historical parent nào so với A0 và mất `(771,484565)`, một parent đi vào A0 qua document BM25 fallback nhưng không có top passage-BM25/dense hit. Kết quả này chỉ minh họa breadth risk; không dùng để tune B0.

Không được suy ra relevance từ overlap, score delta hoặc historical containment. B0 có theoretical potential ở within-parent passage choice, nhưng relevance benefit vẫn chưa kiểm chứng.

## 4. Reranking readiness

### Exact score accounting

| Query | B0 pairs | Valid cached | Missing |
|---:|---:|---:|---:|
| 53 | 200 | 183 | 17 |
| 130 | 200 | 185 | 15 |
| 380 | 200 | 193 | 7 |
| 754 | 200 | 168 | 32 |
| 771 | 200 | 190 | 10 |
| 1144 | 200 | 180 | 20 |
| **Tổng** | **1.200** | **1.099** | **101** |

Reconciliation của count cũ:

- 349 B0 pairs không thuộc A0 parent-first pair pool;
- 248/349 đã có exact compatible score trong older document-order/P2 caches;
- chỉ 101/349 cần inference mới.

Manifest deterministic nằm tại `outputs/v4/b0_cached_first_readiness_20261022/missing_pairs/{qid}.json`. Mỗi row khóa query, query hash, parent, passage, exact payload, input hash, cache key và B0 fused rank.

### GPU job đã chuẩn bị nhưng chưa chạy

- Job manifest: `outputs/v4/b0_cached_first_readiness_20261022/gpu_scoring_job.json`.
- Resumable scorer: `scripts/v4/score_b0_missing_pairs.py`.
- Seed cache: `outputs/v4/p2_missing_pair_scores_20261021/scores.sqlite`.
- Output riêng dự kiến: `outputs/v4/b0_missing_scores_20261022/`.
- Model calls dự kiến tối đa: 101; script tái kiểm cache trước mỗi shard và chỉ score exact missing rows.

Không command GPU nào đã được thực thi. Job không được chạy khi A0 còn ưu tiên nếu chưa có authorization/resource window riêng.

## 5. End-to-end replay readiness

`src/medrag/core/passage_first.py` đã có các bước:

1. direct passage fusion;
2. rerank tối đa 200 pairs;
3. group passages về parent theo max reranker score;
4. rank documents;
5. independent top-60 one-chunk-per-parent selection;
6. adjacent source context;
7. source and submission-object validation.

CPU replay mới: `scripts/v4/replay_b0_cached.py`. Script:

- không gọi retrieval/model;
- yêu cầu exact complete score coverage;
- từ chối partial ranking;
- tạo reranked passage candidates, document ranks, 60 unique-parent chunks/query, grounded contexts và submission-compatible prediction objects.

Dry validation với current canonical cache đã từ chối đúng tại **101 missing exact scores** và không tạo partial output. Targeted regression `test_passage_first.py + test_score_cache.py`: **4 passed**.

| Stage | Trạng thái hiện tại |
|---|---|
| Candidate identities/fusion | PASS |
| Parent/passage/source grounding | PASS, 1.200/1.200 |
| Exact cached score coverage | BLOCKED, 101 missing |
| Complete B0 document ranking | PENDING scores; không fabricate |
| Top-60 unique-parent chunks | Structurally implemented; pending complete ranking |
| Adjacent context validation | Structurally implemented; pending complete ranking |
| Submission-compatible objects | Structurally implemented; pending complete ranking |

## 6. Observed evidence, theoretical potential, untested hypotheses

### Observed

- Artifacts/contracts are compatible.
- 1.099/1.200 B0 pair scores are reusable exactly.
- B0 trades parent breadth for within-parent passage depth.
- 57 cached alternatives score above A0 passage, but are unjudged.
- No independent labels or official F2 evaluator result are available.

### Theoretical potential

- B0 may choose a more answer-bearing span inside a parent already found by A0.
- Only 101 additional scores are needed to complete the six-query end-to-end replay.

### Untested

- Whether higher-scored B0 alternatives are independently more relevant.
- Whether within-parent gains offset loss of 322 parent slots.
- Whether B0 improves document quality, chunks precision/recall/F2 or latency end-to-end.

## 7. Decision

**`BLOCKED_BY_EVALUATION`**

Không có artifact mismatch và chi phí hoàn tất rerank pilot đã giảm xuống 101 pairs, nên B0 không phải `NO_GO_B0`. Tuy nhiên, GPU scores + replay một mình không thể trả lời mục tiêu outperform A0 khi independent labels bằng 0. Trước khi promote hoặc claim quality gain:

1. hoàn tất và khóa blinded development relevance labels;
2. giữ holdout đóng;
3. sau A0/resource authorization, score đúng 101 missing pairs trong namespace riêng;
4. chạy cached CPU replay;
5. so A0/B0 trên independent development labels;
6. chỉ mở holdout sau khi architecture/policy đã khóa.

Không claim official F2 improvement từ audit này.

## 8. Reproducibility

```bash
PYTHONPATH=. .venv-phase2-batch/bin/python \
  scripts/v4/audit_b0_cached_first.py \
  --output outputs/v4/b0_cached_first_readiness_NEW \
  --comparison outputs/v4/b0_cached_first_readiness_NEW/B0_A0_COMPARISON.json \
  --readiness outputs/v4/b0_cached_first_readiness_NEW/B0_RERANK_READINESS.json
```

Audit script từ chối overwrite. GPU scorer và cached replay là separate stages; không nằm trong lệnh audit.

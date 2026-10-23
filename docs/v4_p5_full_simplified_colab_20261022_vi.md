# P5 Full Submission — Simplified Colab T4 workflow

## Quyết định riêng theo stage

| Stage | Trạng thái hiện tại | Điều kiện/ý nghĩa |
|---|---|---|
| CPU pair preparation preflight | `GO` | CPU-only preflight đã xác nhận frozen config, full index, 1.200 query/corpus hashes và P5.2 pair contract |
| CPU pair plan artifacts | `INCOMPLETE_RESUMABLE` | Đã tạo 3/1.200 query shards để kiểm tra resume; chưa đóng gói bundle đủ 240.000 pairs |
| External T4 GPU scoring | `NO-GO: bundle pending` | Chưa có full input bundle; chưa chạy full GPU inference. Khi bundle hoàn tất, mỗi pair phải có fresh model call, không dùng score cache |
| CPU import/finalize/export | `PENDING` | Chỉ chạy sau khi import report xác nhận đủ 240.000 scores, hashes/provenance hợp lệ |
| Exact parity với P2-A0 mixed-hardware reference | `NO-GO` | Giữ report P5.3B; không chặn CPU pair preparation |
| T4 repeatability | `UNVERIFIED` | Báo riêng theo P5.3C; không đồng nhất với exact A0 parity |
| Official BTC order/metric semantics | `UNVERIFIED` | Không có evaluator chính thức trong repo; local schema và grounding vẫn validate |

P5.1 cũ ghi `NO-GO` vì trộn GPU fresh inference vào preflight. P5.3C đã được
tách lại: CPU-only preflight chạy được khi server không có GPU. `run_a0_t4_v1_full.sh`
không còn dispatch CUDA scorer trên server.

## Ba thao tác

### 1. CPU prepare và export

```bash
bash scripts/v4/prepare_a0_t4_v1_cpu.sh
```

Lệnh chạy CPU preflight, P5.2 `prepare --mode full`, kiểm tra immutable pair
shards, rồi đóng gói bundle Colab. Pair plan gồm 1.200 queries và 240.000
query-passage pairs. Lock, pair IDs, passage IDs, input SHA-256, model revision,
frozen config hash và pair manifest được kiểm tra. Pair bundle không chứa corpus,
BM25/dense indexes, score shards hoặc cached scores.

### 2. External Colab T4 scoring

Mở `notebooks/a0_t4_v1_full_colab.ipynb`, chọn Tesla T4, mount Drive, upload
`a0_t4_v1_colab_bundle.zip`, rồi Run All. Notebook gọi cùng
`p5_3a_colab_orchestrator.py`/P5.2 scorer đã chạy thành công ở canary. Checkpoint
Drive theo query chỉ được nhận khi có 200 model calls, 0 cache hits, valid pair
hash và score hash. Colab resume giữ checkpoint đúng input bundle.

### 3. CPU import, finalize, export

Đưa score ZIP về `outputs/v4/a0_t4_v1_colab_import/a0_t4_v1_scores.zip`, rồi:

```bash
bash scripts/v4/import_finalize_a0_t4_v1_cpu.sh
```

Importer xác minh coverage đủ 1.200 score shards/240.000 pairs, score finiteness,
query/passage IDs, cache keys, input hashes, pair manifest, bundle provenance,
runtime/model revision và fresh-inference proof trước khi import. Sau đó P5.2
CPU finalizer chạy selector/context expansion, source grounding validator và BTC
exporter. Không cần CUDA trên server.

## Trạng thái pair preparation

CPU preflight hiện có kết quả `GO` tại
`outputs/v4/a0_t4_v1_release_candidate_20261022/cpu_prepare_preflight_report.json`.
P5.2 CPU prepare đã được kiểm tra với các immutable shards hiện có. Một lần chạy
đã ghi thêm shard query 1–3 (mỗi shard 200 pairs), rồi dừng trước khi xử lý full
set theo phạm vi kiểm thử pilot. Không có reranker scores trong output. Trạng thái
và resume command nằm trong `cpu_pair_plan_progress_report.json`; chạy lệnh Stage 1
để tiếp tục từ các shard đã hash-validate. Pair manifest và full Colab ZIP chỉ
được tạo sau khi đủ 1.200 shards.

## Reports

- `cpu_prepare_preflight_report.json` — gate CPU-only, không yêu cầu exact A0 score parity hoặc GPU.
- `cpu_pair_plan_progress_report.json` — số shard đã tạo, hashes và resume state.
- `cpu_pair_plan_report.json` — hashes/counts của 1.200 shard sau khi chuẩn bị xong.
- `gpu_scoring_readiness_report.json` — bundle/runtime contract và trạng thái external run; hiện chưa sẵn sàng do bundle chưa đủ.
- `colab_score_import_report.json` — kết quả import/provenance sau Colab.
- `exact_reference_parity_report.json` — P5.3B parity với A0 reference.
- `t4_reproducibility_report.json` — P5.3C repeatability độc lập.
- `full_submission_readiness_report.json` — trạng thái tổng hợp 3 stage, không gộp hai gate parity.

Full CPU pair-plan preparation chưa hoàn tất; chỉ có 3 shards phục hồi được.
Full GPU scoring và finalization chưa chạy trong task này.

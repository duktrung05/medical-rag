# Bộ khung V4 đang dùng — 25/10/2026

Champion: **cap-10 evidence-window**, BTC FINAL do người dùng cung cấp **0.2324**
(DOCS_F2 0.2788, CHUNKS_F2 0.1860). S2 là baseline frozen FINAL 0.2318.
R7 targeted: [đã trace 30 queries và patch parent-floor vượt pool gate](v4_r7_targeted_execution_20261025_vi.md);
score gate đã chạy pinned CUDA FP16 canary và kết thúc NO_GO do control doc loss.
Kế hoạch evidence-window đã chạy và dừng ở development:
[2 manual gains, 0 losses; thiếu gate 3 gains](v4_evidence_window_allocation_execution_20261025_vi.md).
Kế hoạch/contract gốc:
[điểm evidence riêng cho 60 chunk slots](v4_evidence_scoring_next_plan_20261025_vi.md).

Cap-10 follow-up đã PASS validation và held-out, trạng thái SCREENED_CANDIDATE:
[execution report](v4_evidence_window_cap10_validation_execution_20261025_vi.md).
Full 1.200-query cap-10 ZIP đã export và validate, chờ BTC:
[execution report](v4_evidence_window_cap10_full_execution_20261026_vi.md).
Roadmap thay đổi kiến trúc hướng tới FINAL >0.4:
[R11 high-recall → dual document/evidence models](v4_over_040_architecture_roadmap_20261026_vi.md).
R11 canary exact-prefix PASS nhưng full cohort dừng `STOP_BUDGET` vì raw-only
đã vượt trần 4 CPU-hours: [execution report](v4_r11_high_recall_coverage_execution_20261026_vi.md).
Score-gate hiện tại: [exact-batch parity PASS, downstream canary NO_GO](v4_r7_score_gate_execution_20261025_vi.md).
Tiến độ mới: [khóa evaluation và pilot allocation](v4_evidence_evaluation_allocation_20261025_vi.md).
F01/F03/F06 đã sửa và re-audit; pilot annotation 13.546 contexts đã dừng sau
[audit signal toàn bộ S2](v4_s2_allocation_signal_decision_20261025_vi.md).
Đọc [review đã verify](v4_project_review_20261025_vi.md) trước khi tối ưu tiếp:
review mô tả lỗi lịch sử; không dùng lại R5 inherited labels hoặc R6 offsets cũ.

## Đường chính

```text
official data + Phase 1/2 eligible release
  → full_index_gpu3_20261019
  → A0 frozen pairs/scores/final
  → supplemental cached passage scores
  → a0_refined_candidates_20261024
  → refine_a0_submission: s2_docs_adaptive
  → champion ZIP + manifest + score receipt
```

| Phần | File/thư mục |
|---|---|
| Runtime V4 | `src/medrag/` |
| Frozen precursor config | `configs/v4_submission_p2_a0.yaml` |
| Demo config riêng | `configs/v4_core.yaml` |
| Full index | `outputs/v4/task4/full_index_gpu3_20261019/` |
| Sparse backing store | `outputs/v4/task4/sparse_full_20261016/` |
| Dense backing store | `outputs/v4/task4/dense_full_gpu3_20261019/` |
| A0 checkpoints | `outputs/v4/a0_fast_a6000_final_20261023/` |
| Supplemental scores | `outputs/v4/a0_remaining_passages_20261024/` |
| Refined checkpoints | `outputs/v4/a0_refined_candidates_20261024/` |
| Champion | `outputs/v4/a0_quality_submissions_20261024/s2_docs_adaptive/` |
| Reference required by exporter | `outputs/v4/a0_quality_submissions_20261024/rerank_refined/` |

Full index dùng symlinks tới sparse/dense backing stores. Giữ cả backing stores.
Các namespaces `a0_t4_*`, `a0_fast_parent_first_*`, `p5_gpu_incremental_*` còn
giữ vì chứa inherited lineage/transport provenance của champion.

## Dữ liệu giữ có chủ ý

- `data/`, `.cache/`, `.venv-*/`: raw data, weights và môi trường.
- Official verify, Phase 0, Phase 1/2 full releases, Step 1/2 và dependency packages:
  inputs/upstream data chưa xác định là disposable.
- `outputs/scored_feedback_recheck_20261009/` và
  `outputs/submissions/vibiomir_threshold_neg2_20261008/`: feedback cùng đúng
  predictions đã được chấm, cần cho identity alignment.
- `outputs/v4/r3_chunk_reranker_20261024/`: model/train membership;
  **chưa áp dụng vào champion**.
- `outputs/v4/r5_s2_context_candidates_20261024/` và
  `r5_s2_same_input_original_20261024/`: context pool/token lengths còn được R6/R8 dùng.
- R6, R8 và project review: audit/candidates cần để sửa evaluator tiếp.

Giữ artifact không có nghĩa là duyệt experiment đó để chạy production.

## Trạng thái

| Nhánh | Trạng thái |
|---|---|
| cap-10 evidence-window | CHAMPION, BTC FINAL 0.2324 |
| S2 | FROZEN BASELINE, BTC FINAL 0.2318 |
| Rerank refined | REFERENCE trực tiếp của S2 exporter |
| R3 model | EXPERIMENTAL; kiểm split trước khi dùng |
| R4 sparse/high-confidence | REJECTED, BTC FINAL 0.2316; outputs đưa vào external archive |
| R5 selection | NO_GO; giữ inputs audit thiết yếu và summary |
| R6 | DIAGNOSTIC; có lỗi được chỉ ra trong review |
| R8-A | NO_GO; giữ kết quả/candidates để sửa tiếp |

## Docs nên đọc

1. [Review hiện tại](v4_project_review_20261025_vi.md).
2. [S1/S2 selection](v4_s1_s2_document_selection_20261024_vi.md).
3. [A0 refinement](v4_a0_quality_fix_20261024_vi.md).
4. [R5 execution](v4_r5_execution_20261024_vi.md).
5. R6/R8 final reports trong namespaces tương ứng tại `outputs/v4/`.

Kế hoạch pilot/refactor cũ và reports rải ở root đã đưa vào kho ngoài checkout.
Xem [cleanup inventory](workspace_cleanup_20261025/README.md) để khôi phục.

Mỗi vòng tiếp theo cần một namespace, effective config, input hashes, split và
decision. Sửa evaluator/gates trước khi tối ưu tiếp; giữ S2 làm baseline frozen.

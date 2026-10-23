# B0 Next Experiment Decision

## Final decision: `BLOCKED_BY_EVALUATION`

B0 artifacts và scoring contract tương thích với frozen A0. Cached-first audit giảm true missing-score workload từ con số overlap-based `349` xuống **101 exact pairs**. B0 cũng có 57 within-parent alternatives với cached reranker score cao hơn A0 passage, trong đó 31 thuộc A0 top-60 parents.

Nhưng hiện có **0 independent relevance labels**. Những score deltas này là hypotheses, không phải evidence rằng passages trả lời tốt hơn. Đồng thời B0 giảm parent breadth từ 1.200 xuống 878 trên sáu queries, chỉ thêm 27 parents mới và dành 322 B0-only slots cho extra passages của shared parents. Historical diagnostic còn cho thấy B0 mất một A0 parent do không dùng document-BM25 fallback.

Vì vậy:

- Không `NO_GO_B0`: architecture có passage-choice potential và artifact path sạch.
- Chưa `GO_B0_GPU_PILOT` như một quality experiment: không có evaluator độc lập để diễn giải kết quả.
- Không có artifact mismatch.
- Trạng thái đúng là `BLOCKED_BY_EVALUATION`.

## Điều kiện gỡ blocker

1. Reviewer hoàn tất và khóa blinded development labels; holdout vẫn đóng.
2. A0 hoàn thành hoặc có resource window riêng được phê duyệt.
3. Chạy prepared missing-score job cho tối đa 101 pairs, không retrieval mới.
4. Chạy `scripts/v4/replay_b0_cached.py` từ complete cache.
5. So sánh A0/B0 trên independently judged passage relevance, document coverage và chunks metrics.
6. Chỉ khi policy đã khóa mới đánh giá một lần trên holdout.

Không promote B0 và không claim official F2 gain trước các gate này.

## Prepared, not executed

- Missing-pair shards: `outputs/v4/b0_cached_first_readiness_20261022/missing_pairs/`
- GPU job manifest: `outputs/v4/b0_cached_first_readiness_20261022/gpu_scoring_job.json`
- Resumable scorer: `scripts/v4/score_b0_missing_pairs.py`
- CPU cached replay: `scripts/v4/replay_b0_cached.py`
- Planned isolated score output: `outputs/v4/b0_missing_scores_20261022/`

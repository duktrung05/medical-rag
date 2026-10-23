# B1 Final-Output Relevance Evaluation — A0 vs B1

**Decision: `BLOCKED_BY_EVALUATION`**

This report reconciles the blinded final-output packet against the hidden A0/B1 map. The six queries are development diagnostics, not an independent holdout.

## Evaluation status

| Item | Value |
|---|---:|
| Review pairs | 23 |
| Completed valid reviews | 0 |
| Unreviewed rows | 23 |
| Invalid/incomplete rows | 0 |
| B1 better | 0 |
| A0 better | 0 |
| Equal | 0 |
| Uncertain | 0 |

Unjudged outputs remain `UNKNOWN`; they are not counted as negatives or ties.
Overall relevance comparison is unavailable until valid judgments exist.

## Final-output delta

- 16 shared top-60 parents have changed exported passage context.
- 7 A0 documents leave top-60 and 7 B1 documents enter top-60.
- 181 shared outputs move rank while retaining the same evidence content.
- The reviewer packet contains 23 pairs: 16 passage changes and 7 membership swaps.
- Every exported context passed source-span grounding checks.

## Safety regressions

Safety comparisons assessed: 0/23 pairs. Zero on an unreviewed packet does not establish absence of regressions.

- Wrong-patient regressions: 0.
- Wrong-number regressions: 0.
- Negation regressions: 0.
- Clinical-accuracy score regressions: 0.

## Query-level comparison

| Query | Judged | B1 better | A0 better | Equal | Uncertain |
|---:|---:|---:|---:|---:|---:|
| 53 | 0 | 0 | 0 | 0 | 0 |
| 130 | 0 | 0 | 0 | 0 | 0 |
| 380 | 0 | 0 | 0 | 0 | 0 |
| 754 | 0 | 0 | 0 | 0 | 0 |
| 771 | 0 | 0 | 0 | 0 | 0 |
| 1144 | 0 | 0 | 0 | 0 | 0 |

## CHUNKS F2

Status: **NOT_COMPUTED_INSUFFICIENT_COMPATIBLE_LABELS**.

The paired packet judges only changed outputs and supplies no compatible complete top-60 relevance set; absolute CHUNKS F2 would be invalid.

## Reviewer workflow

Complete `B1_FINAL_OUTPUT_REVIEW.csv` without opening the hidden map. Then run:

```bash
PYTHONPATH=. .venv-phase2-batch/bin/python scripts/v4/reconcile_b1_final_output_reviews.py \
  --output-json B1_FINAL_OUTPUT_EVALUATION.completed.json \
  --output-report B1_FINAL_OUTPUT_RELEVANCE_EVALUATION.completed.md \
  --manifest outputs/v4/b1_final_output_evaluation_20261022/evaluation_manifest.completed.json
```

## Generalization and decision rule

These findings cover only six development queries. The B1 runtime policy must remain frozen; holdout labels may evaluate it but must not be used to alter it.

The preregistered reconciliation gate is: require all 23 valid reviews; choose `KEEP_A0` for any patient/number/negation regression or an A0 pairwise majority; choose `GO_B1_HOLDOUT` only for a B1 pairwise majority with non-inferior aggregate direct-answer and clinical-accuracy scores and no listed safety regression. Otherwise remain `BLOCKED_BY_EVALUATION`.

Current decision: **`BLOCKED_BY_EVALUATION`**. The blinded packet has not been completely and validly reviewed.

No automatic promotion is authorized by this report.

# B1 Budget-Matched Design

Status: **frozen proposal for independent pilot evaluation; not promoted**.

## Design goal

B1 isolates passage-selection value while preserving A0 document breadth. It is capped at exactly 200 rerank pairs per query and never uses historical-positive IDs/text, holdout labels, or reranker scores to allocate those pairs.

## Primary B1 policy: breadth-preserving intersection

For each query:

1. Freeze A0's 200 candidate parents and their ordering contract.
2. Construct the frozen B0 passage-first top-200 candidate set using the existing passage-fusion contract.
3. For each A0 parent, compute the label-free dense representative already defined by the P2-A1 retrieval rule.
4. If that exact `(parent, dense representative passage)` occurs in B0 top-200 and differs from A0's passage, allocate that one pair.
5. Otherwise allocate the original A0 pair.
6. Rerank exactly 200 pairs, one per A0 parent. Rank parents by the existing A0 contract: reranker score descending, frozen parent retrieval score descending, official ID ascending.
7. Apply the unchanged A0 top-60 unique-parent/context/export contract.

This is not score-gated: whether an alternative's cached reranker score is higher or lower cannot affect pair allocation. The 57 diagnostic IDs are not features and are not embedded in the policy.

## Frozen-pilot replay result

| Invariant or outcome | Value |
|---|---:|
| Queries | 6 |
| Pair budget | 200/query; 1,200 total |
| Unique parents | 200/query; 1,200 total |
| Parent breadth relative to A0 | 100% |
| Passage substitutions | 37 |
| Exact compatible cached scores | 1,200 |
| Missing selected scores | 0 |
| Substitutions with higher/lower reranker score | 14 / 23 |
| Changed passages in final top-60 | 18 |
| Top-60 parent promotions/demotions | 7 / 7 |

Per-query substitutions are 11, 4, 1, 2, 14, and 5 for queries 53, 130, 380, 754, 771, and 1144 respectively. The deterministic plan is recorded in `outputs/v4/b1_feasibility_audit_20261022/b1_pair_plan.json`.

The earlier naive design—choose the highest B0-fusion passage for every retained A0 parent—was rejected because it produced 0 substitutions and exactly reproduced A0. The primary design above is the smallest non-trivial breadth-preserving challenger supported by existing traces and caches.

## Why not pure B0 or 190+10

Pure B0 uses 322 of 1,200 slots for additional passages in shared parents and retains only 878 parents. It excludes 349 A0 parents, including 21 that reach A0 top-60. Its exact replay is also blocked by 101 scores.

A 190-parent + 10-second-passage arm already exists in P2-A2, but it deliberately sacrifices 10 parents/query and dropped 9 A0 top-60 parents across the six-query replay. It is therefore not the preferred B1 architecture when the present objective is to protect breadth. It may be retained only as a secondary ablation, not as the primary challenger.

## Evaluation gate

Use [`B0_BLINDED_PAIR_REVIEW.csv`](B0_BLINDED_PAIR_REVIEW.csv) for independent review. Before unblinding:

- lock completed annotations and reviewer identity/version;
- report agreement and adjudication procedure;
- compare preference/relevance in the 57 hypothesis pairs against the 57 query-matched controls;
- analyze the 27 new-parent comparisons separately;
- keep `UNKNOWN` for unjudged passages;
- do not inspect or use holdout labels to change the rule.

Minimum promotion condition: independent evidence must show a meaningful passage-quality advantage without unacceptable contradiction/patient-match regressions, followed by a separate holdout evaluation. Reranker score deltas alone cannot satisfy this gate.

## Missing scores

The complete 101-pair manifest is `outputs/v4/b1_feasibility_audit_20261022/missing_score_manifest.json`. No GPU job was run. Primary B1 selects zero of those missing pairs, so its cached pilot can proceed without inference. Pure B0 cannot.

## Decision

Current decision: **`BLOCKED_BY_EVALUATION`**.

B1 is operationally feasible and cached-replay-ready, but it is not evidence-ready. Do not promote it before independent blinded judgments and a later untouched holdout evaluation.

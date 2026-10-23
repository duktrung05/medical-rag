# B0 Passage Gain vs Parent Breadth Loss — B1 Feasibility Audit

**Decision: `BLOCKED_BY_EVALUATION`**

Audit date: 2026-10-22. Scope: six frozen pilot queries (`53`, `130`, `380`, `754`, `771`, `1144`). This was a CPU-only replay over frozen candidates, traces, corpus rows, and exact score caches. It made no retrieval, model, inference, or GPU calls and did not write to A0.

## Executive finding

B0 contains a real passage-choice signal, but the available evidence does not establish a relevance gain:

- The 322 B0-only passages inside shared parents belong to 183 query-parent instances. Exact cached-score outcomes are 66 higher, 181 lower, 1 equal, and 74 missing relative to the A0 passage.
- The previously reported 57 cases are **57 parents whose best valid B0 alternative scores above the A0 passage**. There are 66 higher-scoring individual alternatives because some of those parents contain more than one such passage. Thirty-one of the locked 57 are A0 top-60 parents.
- All 101 missing B0 scores decompose exactly into **74 shared-parent alternatives + 27 new-parent passages**. No score was imputed.
- Pure B0 reduces parent breadth from 1,200 to 878. Its candidate pool cannot contain 349 A0 parents, including **21 parents that are in A0's final top-60**. Therefore the breadth difference reaches the final-output boundary, not only the 200-pair pool.
- A breadth-preserving B1 hybrid is feasible: keep all A0 parents, use the label-free dense representative only when that pair also survived frozen B0 top-200, otherwise keep the A0 passage. It changes 37/1,200 passages, keeps 1,200/1,200 parents, and all 1,200 selected pairs already have exact compatible scores.
- The B1 cached replay changes 18 passages inside final top-60 and causes 7 parent promotions plus 7 parent demotions. Of its 37 substitutions, 14 score higher and 23 lower than A0. Score direction is not a relevance judgment.

The architecture is therefore testable without a GPU, but promotion is blocked until the blinded packet receives independent judgments.

## Frozen inputs and safeguards

| Check | Result |
|---|---:|
| Frozen A0 config SHA-256 | `bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03` |
| Score-contract fingerprint | `42350936c327dc01630472257a4444a8df27713140b6fc7e518ba01ad4239c41` |
| A0 pairs | 1,200 |
| B0 pairs | 1,200 |
| Valid B0 cached scores | 1,099 |
| Missing B0 scores | 101 |
| Invalid/incompatible/duplicate B0 pairs | 0 / 0 / 0 |
| Retrieval/model/GPU calls | 0 / 0 / 0 |
| Full index loaded | No |

The replay revalidated the official query hash, P4 candidate hash, P2 pair-allocation hash, index manifest, corpus revision, source grounding, and exact query/parent/passage/input score keys. It read only the targeted corpus records needed for the candidate union.

Historical positives and holdout labels were not read by the B1 policy or the review sampler. At packet creation, independent labels remain unavailable.

## Analysis of all 322 alternatives

| Query | Alternatives | Higher | Lower | Equal | Missing | Locked higher parents | Locked higher in A0 top-60 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 53 | 64 | 16 | 31 | 0 | 17 | 13 | 7 |
| 130 | 48 | 14 | 26 | 0 | 8 | 12 | 9 |
| 380 | 40 | 10 | 25 | 0 | 5 | 9 | 4 |
| 754 | 66 | 8 | 28 | 1 | 29 | 7 | 7 |
| 771 | 63 | 10 | 43 | 0 | 10 | 9 | 1 |
| 1144 | 41 | 8 | 28 | 0 | 5 | 7 | 3 |
| **Total** | **322** | **66** | **181** | **1** | **74** | **57** | **31** |

Every alternative has its query, parent, A0 passage/rank/score, alternative passage/rank/score status, delta, A0 final parent rank, and top-60 flag in `outputs/v4/b1_feasibility_audit_20261022/alternative_analysis.json`. Missing values are marked `MISSING`; independent relevance is `UNKNOWN`, never `NEGATIVE`.

The score distribution does not justify selecting the 57 at runtime. Those 57 are diagnostic hypotheses locked from the prior audit, and are used only as a review stratum.

## Parent breadth and final top-60

| Query | A0-only parents | A0-only parents in A0 top-60 | New B0 parents | B1 substitutions | Changed passage in B1 top-60 | B1 parent in/out |
|---:|---:|---:|---:|---:|---:|---:|
| 53 | 64 | 5 | 0 | 11 | 8 | 1 / 1 |
| 130 | 55 | 5 | 7 | 4 | 3 | 1 / 1 |
| 380 | 42 | 4 | 2 | 1 | 1 | 0 / 0 |
| 754 | 69 | 2 | 3 | 2 | 2 | 0 / 0 |
| 771 | 63 | 3 | 0 | 14 | 1 | 3 / 3 |
| 1144 | 56 | 2 | 15 | 5 | 3 | 2 / 2 |
| **Total** | **349** | **21** | **27** | **37** | **18** | **7 / 7** |

Pure B0 top-60 remains blocked because all 27 new-parent scores and 74 alternative scores are absent. Ranking only the available 1,099 pairs would bias the result and was not performed.

A separate cached-score sensitivity analysis retained all A0 parents and applied a higher-scoring alternative where one was already known. It yielded 13 parent promotions, 13 demotions, and 44 changed top-60 passages. This is an upper-bound diagnostic using post-rerank information, **not** a legal runtime policy and not evidence of relevance improvement.

## Blinded review packet

[`B0_BLINDED_PAIR_REVIEW.csv`](B0_BLINDED_PAIR_REVIEW.csv) contains 141 randomized pairs:

| Stratum | Pairs | Construction |
|---|---:|---|
| Locked score-higher hypotheses | 57 | All previously locked cases |
| Query-matched controls | 57 | Deterministic sample from other shared-parent alternatives; 44 lower-score and 13 missing-score cases |
| New parent vs A0-only parent | 27 | Every new B0 parent, matched within query to a unique A0-only parent by nearest frozen candidate rank |

The CSV exposes only the question, left/right titles and texts, and blank judgment fields. It contains no arm, score, model output, document ID, passage ID, or rank. Left/right orientation is SHA-256 deterministic: 72 swapped and 69 not swapped. Sampling and orientation were locked before labels.

The hidden map is `outputs/v4/b1_feasibility_audit_20261022/review_item_map.json`; it must not be shown to reviewers before annotations are locked. All 349 A0-only parents are included in the structural/top-60 analysis; 27 unique A0-only parents are included as blinded counterparts to the 27 new parents.

Recommended independent review fields are passage relevance (0/1/2), patient/population match, numeric support, contradiction, and left/right/tie preference. Analyze the locked hypothesis and control strata separately, and report confidence intervals rather than treating unjudged rows as failures.

## Missing-score readiness

`outputs/v4/b1_feasibility_audit_20261022/missing_score_manifest.json` references all six immutable shards and their hashes for the 101 missing B0 scores. The existing GPU job remains `PREPARED_NOT_EXECUTED`.

The breadth-preserving B1 plan selects none of those missing pairs and is fully replayable from cache. Pure B0 still requires all 101 scores. This audit did not execute that job.

## Evidence tiers and decision

**Observed:** B0 provides 322 alternatives; 21 A0 top-60 parents are absent from its pool; the proposed B1 has 37 real substitutions, retains full breadth, has complete scores, and changes final top-60 composition.

**Theoretical potential:** within-parent replacement can improve passage grounding without deleting parents. The 57 locked hypotheses and 31 top-60 intersections identify where a gain could matter.

**Untested:** whether reviewers prefer those alternatives, whether preferences generalize beyond these six queries, and whether any gain improves official F2.

Final decision is **`BLOCKED_BY_EVALUATION`**. Run the independent blinded review first. Do not promote B0 or B1, inspect holdout labels, or claim official-F2 improvement from reranker deltas.

## Reproduction

```bash
PYTHONPATH=. .venv-phase2-batch/bin/python scripts/v4/audit_b1_feasibility.py \
  --output outputs/v4/b1_feasibility_audit_20261022
```

The script refuses to overwrite outputs and validates frozen hashes and counts before writing.

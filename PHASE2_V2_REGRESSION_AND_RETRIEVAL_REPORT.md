# Phase 2 V2 Regression and Retrieval Report

Review date: 2026-10-15

## Verdict

Tasks 1-3 are complete on a locked 1,000-document pilot. Chunker V2 fixes the
observed numeric boundaries and materially reduces small passages. The paired
150-passage semantic review is complete, and an independent BM25 retrieval
experiment was run on 10 manually evidenced official queries.

The result does **not** approve a full build. Strategy B remains the conditional
production candidate for semantic focus, while the small retrieval pilot favors
Strategy C. More manually labeled retrieval queries are required before resolving
that trade-off. No primary index and no Phase 1 artifact was changed.

## Task 1 - Boundary Refinement

Chunker version: `vibiomir-hierarchical-chunker-v2`.

Implemented changes:

- decimal and dotted-number-aware sentence boundaries;
- safe comma handling in hard splits;
- paragraph decomposition at `target_tokens` when needed for sentence-level
  boundary choice;
- labeled question-answer pairing up to `max_tokens`;
- backward and forward attachment for short orphan chunks;
- locked pilot manifest support for paired regression.

Observed blocker outcomes:

| Case | V1 | V2 |
| --- | --- | --- |
| 003/A, `5.963` | split, incomplete | intact, mostly complete |
| 013/A, `16.12/mm3` | split, mostly complete | intact, complete |
| 027/C, `3.272.256` | split, mostly complete | intact, complete |
| 029/A-B, confidence intervals | numeric cuts | complete numeric results |
| 039/B, Q&A | mostly complete | question, options, and answer complete |
| 042/A, short quote | threshold outside | risk factors and `>5 mm` condition together |

Structural validation passed for all strategies with zero source-slice,
max-token, and coverage failures.

| Metric | A V1 -> V2 | B V1 -> V2 | C V1 -> V2 |
| --- | ---: | ---: | ---: |
| Passages | 11,947 -> 11,795 | 9,296 -> 9,093 | 6,140 -> 6,075 |
| Small passages | 251 -> 64 | 385 -> 65 | 172 -> 67 |

## Task 2 - Focused Regression

The review reuses the exact 50 case/document/anchor selections from V1. Of 150
passages, 111 were byte-identical and inherit their already completed V1 manual
review; all 39 passages with changed boundaries were read again with before/after
source context and received V2-specific annotations.

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Complete | 32/50 | 33/50 | 41/50 |
| Complete or mostly complete | 41/50 | **44/50** | 43/50 |
| Incomplete | 9/50 | **6/50** | 7/50 |
| Good boundary | 29/50 | **34/50** | 26/50 |
| Poor boundary | 6/50 | 7/50 | **6/50** |
| No context loss | 32/50 | 33/50 | **41/50** |
| Excessive overlap | 0/50 | **1/50** | 16/50 |

After applying the separate staging exclusions, B has no incomplete result in the
44 eligible cases: 33 complete and 11 mostly complete. V2 also records real
regressions rather than treating all boundary changes as improvements, including
case 012/B losing the result paragraph and case 035/B moving detailed risk signs
after the boundary.

## Phase 1 Residue Staging

Six reviewed documents are marked `EXCLUDE_RETRIEVAL` only in a Phase 2 staging
ledger: unrelated teasers, truncated listings, search snippets, bibliography-only
content, a conference table of contents, and navigation questions without answers.

The staging manifest records `phase1_mutated: false`. The Phase 1 release,
allowlists, clean documents, and checksums remain unchanged.

## Task 3 - Retrieval Evaluation

Ten official queries were manually joined to exact source evidence spans. High
BM25 candidates without real semantic support were rejected. A passage is relevant
only if it fully contains the manually labeled evidence span.

Each strategy was indexed independently in memory. The primary index was not read,
rewritten, or replaced.

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Queries with a containing passage | 9/10 | 9/10 | **10/10** |
| Recall@1 | 0.40 | 0.30 | **0.40** |
| Recall@5 | **0.70** | 0.60 | **0.70** |
| Recall@10 | 0.80 | 0.70 | **0.90** |
| Recall@20 | 0.80 | 0.80 | **1.00** |
| MRR@100 | 0.510 | 0.416 | **0.538** |

This is a useful known-evidence pilot, not a production retrieval benchmark. Ten
queries are too few to establish a final winner. It nevertheless exposes a real
trade-off: C contains evidence more often and retrieved it best here, while B has
better semantic focus and far less excessive overlap.

## Recommendation

Keep **B: target 256, max 384, overlap 48** as the conditional production
candidate. Do not start the full build yet.

Before full processing:

1. Expand the manually evidenced retrieval set to at least 50 diverse queries,
   including VI/EN/ZH and dosage, laboratory, negation, treatment, and Q&A cases.
2. Re-run A/B/C BM25 and, when the local model runtime is available, dense BGE-M3
   retrieval on the same immutable qrels.
3. Test whether B plus adjacent-passage expansion recovers the two evidence-span
   misses without adopting C's overlap and topic-mixing cost.
4. Promote the six staging exclusions only through a new Phase 2 policy artifact;
   do not edit the accepted Phase 1 release in place.

## Artifacts

- V2 pilot: `outputs/vibiomir_phase2_20261015_pilot_v2_1`
- Semantic review: `reports/semantic_review_v2/phase2_semantic_review_150.parquet`
- Review packet: `reports/semantic_review_v2/review_packet.md`
- Staging ledger: `staging/semantic_eligibility/semantic_eligibility.parquet`
- Retrieval qrels: `reports/retrieval_evaluation/resolved_qrels.parquet`
- Retrieval rankings: `reports/retrieval_evaluation/rankings_top100.parquet`
- Retrieval metrics: `reports/retrieval_evaluation/retrieval_comparison.json`

No full Phase 2 build was run.

# Phase 2 Final Acceptance Report

Date: 2026-10-15

## Decision

Phase 2 is accepted for chunking and retrieval evaluation. Strategy C is frozen
for the production passage build. The 1,073,531-document full build has not been
started.

This decision applies to hierarchical chunking only. The six navigation or
bibliography documents remain in a separate semantic staging ledger and are not
written back to the accepted Phase 1 corpus.

## Reviewed retrieval benchmark

- 107 evidence-labelled queries: 87 Vietnamese, 10 English, and 10 Chinese.
- 103 relevant documents in a common 1,075-document corpus with fixed distractors.
- Relevance is matched by `doc_id` and an exact source character span, not by a
  strategy-specific passage ID.
- 147 legacy Vietnamese candidates were inspected. 77 were accepted and 70
  topic-only, answerless, mismatched, or weak candidates were rejected.
- Ten earlier manual pilot qrels were retained; twenty EN/ZH questions and exact
  evidence spans were manually authored from previously reviewed semantic cases.
- The primary index and all Phase 1 artifacts were unchanged.

The frozen qrels are in
`configs/vibiomir/phase2_retrieval_qrels_v2_20261015.json`. The candidate decision
ledger is in
`outputs/vibiomir_phase2_retrieval_benchmark_20261015/review/vi_candidate_review.parquet`.

## Retrieval results

| Strategy | Recall@10 | MRR@100 | Mean context chars at 10 | Mean latency |
| --- | ---: | ---: | ---: | ---: |
| A | 0.7664 | 0.5200 | 585 | 39.28 ms |
| B | 0.7664 | 0.5472 | 806 | 34.55 ms |
| B + adjacent | 0.8131 | 0.6163 | 1,951 | 35.08 ms |
| C | 0.8131 | 0.6140 | 1,220 | 27.44 ms |

Adjacent expansion fetched the previous and next B passage when present. It
added 1,145 characters on average at top 10, produced 152 redundant overlap
characters on average, and multiplied B context size by 2.42. Seed BM25 ranks
were preserved so the experiment isolates expansion rather than conflating it
with a learned reranker.

Paired bootstrap with 10,000 samples found that B + adjacent improved over base
B by 0.0467 Recall@10 (95% CI 0.0093 to 0.0935) and 0.0690 MRR (95% CI 0.0328
to 0.1126). B + adjacent and C were tied on Recall@10; their MRR difference was
0.0023 with a 95% CI of -0.0379 to 0.0425. There is no reliable retrieval win
for adjacent expansion over C in this benchmark.

By language, C Recall@10 was 0.7931 VI, 1.0000 EN, and 0.8000 ZH. These language
slices are diagnostic only because EN and ZH each contain ten queries.

## Final configuration

Strategy C uses a target of 384 tokens, a maximum of 512, a minimum short-chunk
threshold of 48, and 64 overlap tokens. It achieved 43/50 semantic-usable cases
in the 150-passage review, one fewer than B, while matching B + adjacent retrieval
with about 37.5% less context and lower measured latency.

The frozen machine-readable configuration is
`configs/vibiomir/phase2_chunking_production_v2.json`.

## Validation and build readiness

- Shared retrieval corpus validation: pass, zero failures across all three strategies.
- Focused automated tests after adjacent-expansion and frozen-config changes: 41 passed.
- The full builder now validates `--frozen-config` and accepts `--strategy C`,
  preventing config drift and an accidental A/B/C triple build.
- Full build remains approval-gated. The prepared command is recorded in the
  frozen configuration; no full output directory was created by this acceptance run.

Production indexing must continue to apply the six-document semantic staging
exclusion. The Phase 1 release remains immutable.

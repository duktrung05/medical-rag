# ViBioMIR Task 0 audit

- Scored baseline: threshold -2, FINAL_SCORE 0.1681.
- Current candidate: threshold -2.5; no linked leaderboard feedback was found.
- Ranking cache: 240,000 scored pairs across 1,200 queries.
- Distinct reranked documents/query: mean 103.23, median 101.
- Candidate selector output: mean 55.24 chunks/query; 876/1200 queries hit their chunk cap.
- Annotation split: 100 dev + 50 holdout, disjoint and deterministically stratified; all `relevant` values remain null pending human judgment.

The public competition page confirms document/chunk retrieval and recall-weighted Macro F2. The exact official chunk normalization/matching implementation was not publicly retrievable, so LCS/tokenization assumptions remain unverified.

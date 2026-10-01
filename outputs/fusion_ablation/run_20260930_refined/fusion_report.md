# Offline fusion ablation

BM25 and dense rankings were read from top-depth caches; no model was run.
Current RRF k=60 top100 is replay-checked against the saved hybrid ranking.
Union ordering: best source rank, then sum of source ranks, then chunk ID. It is an ordering heuristic, not an oracle.

Current RRF candidate Recall@100: **0.9249**; query hit rate 0.9249; lost positive labels 166.

| Depth | Best configuration | Candidate Recall | Query hit rate | Lost positives | Mean candidates |
|---:|---|---:|---:|---:|---:|
| 50 | rrf_k40_top50 | 0.9032 | 0.9032 | 214 | 50.0 |
| 100 | union_min_rank_top100 | 0.9253 | 0.9253 | 165 | 100.0 |
| 150 | weighted_rrf_bm0.50_de150 | 0.9421 | 0.9421 | 128 | 149.7 |
| 200 | weighted_rrf_bm0.75_de200 | 0.9452 | 0.9452 | 121 | 173.3 |

At depth 100, best candidate hit changes recover 4 queries and regress 3 versus current RRF.
Cache depths cap each source at 100, so depth above 200 was not tested. Candidate recall does not predict reranked top-k or final F2.

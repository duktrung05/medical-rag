# ViMed validation error analysis

Total validation queries: 2210.
Reranker cutoff: top 10; candidate depth comes from cached hybrid results.
Error types overlap; their percentages do not sum to 100%.

| Error type | Queries | Percent |
|---|---:|---:|
| BM25_ONLY_HIT | 51 | 2.31% |
| BOTH_HIT | 1902 | 86.06% |
| BOTH_MISS | 121 | 5.48% |
| CANDIDATE_MISS | 166 | 7.51% |
| DENSE_ONLY_HIT | 136 | 6.15% |
| DOC_AND_CHUNK_MISS | 72 | 3.26% |
| DOC_HIT_CHUNK_MISS | 179 | 8.10% |
| FUSION_DROPPED | 45 | 2.04% |
| RERANKER_MISS | 41 | 1.86% |
| SELECTION_MISS | 85 | 3.85% |

## Primary failure stage

| Stage | Queries |
|---|---:|
| hit_at_evaluated_cutoff | 1959 |
| retrieval_or_fusion | 166 |
| reranker | 41 |
| selection | 44 |

Cached rankings are capped at top 100. An absent item has unknown rank beyond that depth.
Ground truth contains one inferred source context per query and is not exhaustive.

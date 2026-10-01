# ViMed validation report: final_selection

Validation queries: 2210. Winner: **rerank**.
Rule: `internal_macro_f2 → chunk_f2 → doc_f2 → lower average chunk count`. Config files are not changed.

## Retrieval

| Stage | Recall@10 | Recall@100 | MRR@100 | Zero hit @10 | Mean latency ms |
|---|---:|---:|---:|---:|---:|
| bm25 | 0.7525 | 0.8837 | 0.5839 | 547 | 85.4 |
| dense | 0.8163 | 0.9222 | 0.6504 | 406 | 50.2 |
| hybrid | 0.8226 | 0.9249 | 0.6498 | 392 | 136.1 |
| rerank | 0.9063 | 0.9249 | 0.8091 | 207 | 918.0 |

## Final Selection

| Stage | Doc P | Doc R | Doc F2 | Chunk P | Chunk R | Chunk F2 | Internal Macro F2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.1767 | 0.8837 | 0.4909 | 0.1384 | 0.6919 | 0.3844 | 0.4376 |
| dense | 0.1865 | 0.9326 | 0.5181 | 0.1532 | 0.7661 | 0.4256 | 0.4718 |
| hybrid | 0.1874 | 0.9371 | 0.5206 | 0.1544 | 0.7719 | 0.4289 | 0.4747 |
| rerank | 0.1935 | 0.9674 | 0.5375 | 0.1773 | 0.8864 | 0.4925 | 0.5150 |

`internal_macro_f2` là điểm composite nội bộ, không phải điểm chính thức của cuộc thi.
Ground truth chưa có đủ relevance judgments; các metric phản ánh source-context retrieval.

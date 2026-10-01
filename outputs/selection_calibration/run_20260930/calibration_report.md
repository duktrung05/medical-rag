# Offline selection calibration

Validation queries: 2210. Reranker rankings were read from `outputs/vimed_validation/rerank/rankings.jsonl`; no model was run.
Objective: `internal_macro_f2`, equal mean of per-query document and chunk F2.
Thresholds come from score percentiles; deltas come from top1-to-candidate score-gap percentiles.
The coordinate search tunes chunk parameters, then document parameters, against validation; scores are selection estimates and may be optimistic.

## Fixed top K comparison

| Selector | Internal Macro F2 | Doc P/R/F2 | Chunk P/R/F2 | Avg chunks | Avg docs |
|---|---:|---:|---:|---:|---:|
| fixed_top1 | 0.8170 | 0.8846/0.8846/0.8846 | 0.7493/0.7493/0.7493 | 1.00 | 1.00 |
| fixed_top3 | 0.6469 | 0.3175/0.9525/0.6803 | 0.2863/0.8588/0.6134 | 3.00 | 3.00 |
| fixed_top5 | 0.5150 | 0.1935/0.9674/0.5375 | 0.1773/0.8864/0.4925 | 5.00 | 5.00 |
| fixed_top10 | 0.3365 | 0.0978/0.9778/0.3492 | 0.0906/0.9063/0.3237 | 10.00 | 10.00 |

## Top 10 dynamic configurations

| Rank | Internal Macro F2 | Chunk threshold/delta/min/max | Doc threshold/delta/min/max | Doc F2 | Chunk F2 | Avg chunks/docs |
|---:|---:|---|---|---:|---:|---:|
| 1 | 0.8170 | -3.42/4.837/0/1 | -3.42/4.837/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 2 | 0.8170 | -3.42/4.837/0/1 | -3.42/7.745/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 3 | 0.8170 | -3.42/4.837/0/1 | -3.42/9.521/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 4 | 0.8170 | -3.42/4.837/0/1 | -3.42/11.18/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 5 | 0.8170 | -3.42/4.837/0/1 | -3.42/13.55/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 6 | 0.8170 | -3.42/4.837/0/1 | -0.9531/4.837/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 7 | 0.8170 | -3.42/4.837/0/1 | -0.9531/7.745/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 8 | 0.8170 | -3.42/4.837/0/1 | -0.9531/9.521/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 9 | 0.8170 | -3.42/4.837/0/1 | -0.9531/11.18/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |
| 10 | 0.8170 | -3.42/4.837/0/1 | -0.9531/13.55/0/1 | 0.8846 | 0.7493 | 1.00/1.00 |

## Caveat

`internal_macro_f2` là contract nội bộ. Ground truth hiện có một source context suy ra mỗi query, chưa phải nhãn relevance đầy đủ. Chọn tham số và báo điểm trên cùng validation có thể lạc quan; giữ test split cho đánh giá cuối.

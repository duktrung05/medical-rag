# Baseline dense production benchmark — XQuAD dev

This is an engineering regression fixture, not a medical quality benchmark. The
configured corpus has 108 chunks; the split has 178 Vietnamese queries, each
with a positive in Vietnamese, English, and Chinese.

## Configuration and execution

- Config: `configs/baseline_dense.yaml`
- Index: `artifacts/dense_index` (exact inner product, normalized embeddings)
- Model and tokenizer: `BAAI/bge-m3`, commit `5617a9f61b028005a4858fdac845db406aefb181`
- Device: CPU
- Corpus hash in index manifest: `d25b0adbfc139c4fa1b578a24d78e57857d0553ae39b992adec7bd82707de054`
- Index build: `/home/trung.nguyen12/miniconda3/bin/python -m scripts.build_dense_index --config configs/baseline_dense.yaml --device cpu`
- Benchmark: `/home/trung.nguyen12/miniconda3/bin/python -m scripts.run_dense_baseline --config configs/baseline_dense.yaml --output-dir outputs/baseline_dense/dev/run1 --device cpu` (repeated with `run2`)

The production `DenseRetriever.search()` also returned a scored top 5 ranking
from the configured index for the first dev query.

## Results

Recall is the per query positive chunk recall for each language, averaged over
all 178 Vietnamese queries. `Macro F2` uses the repository evaluator on the
top K chunks and their parent documents.

| K | VI → VI recall | VI → EN recall | VI → ZH recall | Macro F2 |
|---:|---:|---:|---:|---:|
| 1 | 0.7865 | 0.0112 | 0.1573 | 0.3673 |
| 3 | 0.9719 | 0.7697 | 0.8596 | 0.8670 |
| 5 | 0.9775 | 0.8596 | 0.9494 | 0.8196 |
| 10 | 0.9944 | 0.9213 | 0.9775 | 0.6576 |

Mean first positive rank over the full ranking: VI 1.50, EN 4.71, ZH 2.94.
Detailed metrics, ranking rows, top K predictions, and input hashes are in each
run directory.

## Reproducibility

The two benchmark runs used identical config, corpus, query, ground truth,
index manifest, and embedding hashes. All 178 query rankings matched in full
`chunk_id` order and score; `metrics.json` also matched byte for byte. Both
`rankings.jsonl` files have SHA-256
`7f3d7dfd13ce20bdab1b61ee5e378c4f093d154b09b6701881e2fb8eadeb6a3a`.

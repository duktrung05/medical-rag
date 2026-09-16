# Experiment Tracking

This directory maintains systematic records of all experimental configurations and evaluation runs.

Columns tracked in `experiment_log.csv`:
- `experiment_name`: Name or ID of experiment (e.g. `EXP_001_BM25`)
- `num_queries`: Total queries evaluated
- `doc_p`: Document-level Precision
- `doc_r`: Document-level Recall
- `doc_f2`: Document-level Macro $F_2$
- `chunk_p`: Chunk-level Precision
- `chunk_r`: Chunk-level Recall
- `chunk_f2`: Chunk-level Macro $F_2$
- `macro_f2`: Mean composite macro $F_2$

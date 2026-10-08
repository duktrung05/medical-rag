# ViBioMIR crawl pipeline resume bundle

This upload contains the completed 2026-10-07 extraction and BM25 stage needed
to continue dense indexing, retrieval, fusion, and reranking on another machine.
It intentionally excludes the 67 GB raw HTML crawl and older indexes/caches.

Included artifacts:

- `data/raw/vibiomir/query.parquet`
- `data/raw/vibiomir/links_corpus.parquet`
- `data/vibiomir/chunks_crawl_20261007.parquet`
- `data/vibiomir/bm25idx_crawl_20261007/`
- `data/vibiomir/bm25idx_crawl_20261007_ids.npy`
- `data/vibiomir/bm25idx_crawl_20261007_vocab.json`
- `data/vibiomir/bm25_crawl_20261007_candidates.parquet`

Authenticate on the server with a write token:

```bash
HF_HOME=/data_hdd_16t/trungnguyen12/.cache-huggingface-medical-rag \
  /data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/hf auth login
```

The background uploader waits for that login and resumes safely after network
interruptions. Its log is `upload.log` and its PID is stored in `upload.pid`.

After the Hub snapshot reports complete, restore it on the local machine:

```bash
hf download duktrung/med-rag snapshot/hf_vibiomir_snapshot.py \
  --repo-type dataset --local-dir .
python snapshot/hf_vibiomir_snapshot.py restore \
  --repo-id duktrung/med-rag --destination /path/to/medical-rag
```

#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data_hdd_16t/trungnguyen12/medical-rag/medical-rag
PY=/data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/python
cd "$PROJECT"

exec 9>outputs/dense_fuse_crawl_20261007.lock
flock -n 9 || { echo "Another dense/fuse run holds the lock"; exit 4; }

export HF_HOME="$PROJECT/.cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DISABLE_XET=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

gpu_status=$(nvidia-smi -i 0 --query-gpu=memory.used,utilization.gpu \
  --format=csv,noheader,nounits)
IFS=, read -r gpu_memory gpu_utilization <<<"$gpu_status"
gpu_memory=${gpu_memory//[[:space:]]/}
gpu_utilization=${gpu_utilization//[[:space:]]/}
if (( gpu_memory > 2000 || gpu_utilization > 10 )); then
  echo "GPU 0 became busy: ${gpu_memory} MiB, ${gpu_utilization}%" >&2
  exit 3
fi

echo "[$(date -Is)] using only physical GPU 0; cache=$HF_HOME"
echo "[$(date -Is)] checking BGE-M3 model access"
"$PY" - <<'PY'
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-m3", device="cpu")
print("BGE-M3 available", flush=True)
del model
PY

echo "[$(date -Is)] building dense document index"
"$PY" -u r2ai/index.py build \
  --chunks-file chunks_crawl_20261007.parquet \
  --lead-chunks 2 --seq 256 --batch 128 --device cuda:0 \
  --out docidx_crawl_20261007

"$PY" - <<'PY'
import numpy as np

ids = np.load("data/vibiomir/docidx_crawl_20261007_ids.npy", mmap_mode="r")
vecs = np.load("data/vibiomir/docidx_crawl_20261007_vecs.npy", mmap_mode="r")
assert ids.shape == (3_545_938,), ids.shape
assert vecs.shape == (3_545_938, 1024), vecs.shape
assert np.isfinite(vecs[::10000]).all()
print(f"dense index: {len(ids):,} documents, vector shape={vecs.shape}", flush=True)
PY

echo "[$(date -Is)] searching dense candidates"
"$PY" -u r2ai/index.py search \
  --out docidx_crawl_20261007 --top-k 400 --block 500000 \
  --device cuda:0 --lang-balance

echo "[$(date -Is)] fusing dense and BM25 candidates"
"$PY" -u r2ai/fuse.py \
  --dense docidx_crawl_20261007_candidates \
  --lexical bm25_crawl_20261007_candidates \
  --w-dense 1 --w-lexical 1 --rrf-k 60 --top-k 400 \
  --out fused_crawl_20261007

"$PY" - <<'PY'
import pyarrow.compute as pc
import pyarrow.parquet as pq

for name in (
    "docidx_crawl_20261007_candidates",
    "fused_crawl_20261007",
):
    table = pq.read_table(f"data/vibiomir/{name}.parquet", columns=["query_id", "rank"])
    assert table.num_rows == 480_000, (name, table.num_rows)
    assert len(pc.unique(table["query_id"])) == 1_200, name
    assert pc.max(table["rank"]).as_py() == 399, name
    print(f"{name}: {table.num_rows:,} rows, 1,200 queries", flush=True)
PY
echo "[$(date -Is)] dense build, search, and fusion complete"

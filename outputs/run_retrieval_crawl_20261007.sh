#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data_hdd_16t/trungnguyen12/medical-rag/medical-rag
PY=/data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/python
CHUNKS=chunks_crawl_20261007.parquet
SELECTION_CONFIG=configs/vibiomir/evidence_selector_best_20261008.json
EXTRACT_PID_FILE="$PROJECT/outputs/retrieval_extract_20261007.pid"
LOG="$PROJECT/outputs/retrieval_pipeline_crawl_20261007.log"
GPU=${RETRIEVAL_GPU:-cuda:0}

cd "$PROJECT"
exec >>"$LOG" 2>&1
echo "[$(date -Is)] retrieval pipeline watcher started; gpu=$GPU"

extract_pid=$(cat "$EXTRACT_PID_FILE")
while kill -0 "$extract_pid" 2>/dev/null; do
  echo "[$(date -Is)] extract still running pid=$extract_pid"
  sleep 120
done
if [ ! -s "data/vibiomir/$CHUNKS" ]; then
  echo "[$(date -Is)] extract ended without a nonempty chunk file"
  exit 2
fi
echo "[$(date -Is)] extract finished; validating chunk artifact"
$PY - <<'PY'
import pyarrow.parquet as pq
p = 'data/vibiomir/chunks_crawl_20261007.parquet'
s = pq.read_schema(p)
required = {'doc_id', 'chunk_index', 'title', 'text', 'n_words'}
missing = required.difference(s.names)
if missing:
    raise SystemExit(f'missing columns: {sorted(missing)}')
print('chunk rows:', pq.ParquetFile(p).metadata.num_rows)
PY

check_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi unavailable; refusing GPU stage"
    exit 3
  fi
  gpu_index=${GPU#cuda:}
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F', *' -v want="$gpu_index" '$1 == want {print $2; found=1} END {if (!found) exit 1}')
  if [ "$used" -gt 2000 ]; then
    echo "GPU $GPU is already using ${used} MiB; refusing to compete with another job"
    exit 3
  fi
  echo "GPU $GPU available with ${used} MiB used"
}

echo "[$(date -Is)] building BM25 index"
$PY -u r2ai/bm25index.py build --chunks-file "$CHUNKS" --max-chunks 6 --out bm25idx_crawl_20261007
echo "[$(date -Is)] searching BM25 candidates"
$PY -u r2ai/bm25index.py search --out bm25idx_crawl_20261007 --top-k 400 --n-threads 16 --candidate-out bm25_crawl_20261007

echo "[$(date -Is)] building dense document index on $GPU"
check_gpu
$PY -u r2ai/index.py build --chunks-file "$CHUNKS" --lead-chunks 2 --seq 256 --batch 128 --device "$GPU" --out docidx_crawl_20261007
echo "[$(date -Is)] searching dense candidates"
$PY -u r2ai/index.py search --out docidx_crawl_20261007 --top-k 400 --block 500000 --device "$GPU" --lang-balance

echo "[$(date -Is)] fusing candidates"
$PY -u r2ai/fuse.py --dense docidx_crawl_20261007_candidates --lexical bm25_crawl_20261007_candidates --w-dense 1 --w-lexical 1 --rrf-k 60 --top-k 400 --out fused_crawl_20261007

mkdir -p outputs/evidence/retrieval_crawl_20261007
echo "[$(date -Is)] reranking and final selection"
check_gpu
$PY -u r2ai/rank.py \
  --chunks-file "$CHUNKS" \
  --candidates fused_crawl_20261007.parquet \
  --pool 200 --rerank-top 200 --device "$GPU" --batch 32 \
  --selection-config "$SELECTION_CONFIG" \
  --out vibiomir_crawl_20261007 \
  --ranking-cache outputs/evidence/retrieval_crawl_20261007/rankings.jsonl.gz

echo "[$(date -Is)] retrieval pipeline complete"

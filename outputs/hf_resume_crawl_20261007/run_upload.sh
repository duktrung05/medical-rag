#!/usr/bin/env bash
set -euo pipefail
umask 077

PROJECT=/data_hdd_16t/trungnguyen12/medical-rag/medical-rag
STATE_DIR="$PROJECT/outputs/hf_resume_crawl_20261007"
export HF_HOME=/data_hdd_16t/trungnguyen12/.cache-huggingface-medical-rag
export HF_HUB_DISABLE_PROGRESS_BARS=1

mkdir -p "$HF_HOME"
chmod 700 "$HF_HOME"
cd "$PROJECT"

exec nice -n 10 ionice -c 2 -n 7 \
  /data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/python -u \
  scripts/tooling/hf_vibiomir_snapshot.py upload \
  --repo-id duktrung/med-rag \
  --state-dir "$STATE_DIR" \
  --plan "$STATE_DIR/plan.json" \
  --wait-for-auth \
  --extract-pid-file outputs/retrieval_extract_20261007.pid \
  --extract-log-file outputs/retrieval_extract_20261007.log

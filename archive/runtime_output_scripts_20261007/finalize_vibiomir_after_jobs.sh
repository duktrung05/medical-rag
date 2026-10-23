#!/usr/bin/env bash
set -euo pipefail

PROJECT=/data_hdd_16t/trungnguyen12/medical-rag/medical-rag
LOCAL_CRAWL="$PROJECT/data/vibiomir/crawl"
LOCAL_PAGES="$LOCAL_CRAWL/pages"
PEER_CRAWL=/data_hdd_16t/quybd/medical-rag/data/vibiomir/crawl
PEER_PACK=/data_hdd_16t/quybd/medical-rag/data/vibiomir_hf/crawl/pages
LOG="$PROJECT/outputs/finalize_vibiomir_after_jobs.log"

exec >>"$LOG" 2>&1
echo "[$(date -Is)] finalize watcher started"

# Do not touch pages while the local crawler can still be writing them.
while pgrep -f '[r]2ai/crawl\.py' >/dev/null; do
  echo "[$(date -Is)] local crawler still running; waiting"
  sleep 60
done
echo "[$(date -Is)] local crawler stopped"

# The peer packer is an independent job. Wait for all shards to be atomically
# renamed from .tmp to .tar before extracting anything.
while pgrep -f '[p]ack_pages\.sh' >/dev/null || find "$PEER_PACK" -maxdepth 1 -name '*.tmp' -print -quit | grep -q .; do
  echo "[$(date -Is)] peer pack still running: $(find "$PEER_PACK" -maxdepth 1 -name '*.tar' | wc -l) tar, $(find "$PEER_PACK" -maxdepth 1 -name '*.tmp' | wc -l) tmp"
  sleep 60
done

expected=$(find /data_hdd_16t/quybd/medical-rag/data/vibiomir/crawl/pages -mindepth 1 -maxdepth 1 -type d | wc -l)
packed=$(find "$PEER_PACK" -maxdepth 1 -name '*.tar' | wc -l)
if [ "$packed" -ne "$expected" ]; then
  echo "[$(date -Is)] refusing import: packed=$packed expected=$expected"
  exit 2
fi

mkdir -p "$LOCAL_PAGES"
count=0
for archive in "$PEER_PACK"/*.tar; do
  tar --skip-old-files -xf "$archive" -C "$LOCAL_PAGES"
  count=$((count + 1))
  if (( count % 25 == 0 )); then
    echo "[$(date -Is)] extracted $count/$packed archives"
  fi
done
echo "[$(date -Is)] extracted $count/$packed archives"

# Add only manifest shards that do not exist locally. Existing manifests are
# never replaced, preserving local attempts and failure history.
cp -an "$PEER_CRAWL"/manifest_*.parquet "$LOCAL_CRAWL"/
echo "[$(date -Is)] copied missing peer manifests"
echo "[$(date -Is)] local pages: $(find "$LOCAL_PAGES" -type f | wc -l)"
/data_hdd_16t/trungnguyen12/.venv-evidence-20261006/bin/python "$PROJECT/outputs/final_crawl_audit.py" > "$PROJECT/outputs/crawl_final_audit_20261006.log"
echo "[$(date -Is)] finalize watcher complete"

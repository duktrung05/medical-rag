#!/usr/bin/env bash
set -u

SESSION=vibiomir_dense_20261007
MIN_AVAILABLE_KIB=$((80 * 1024 * 1024))
MAX_GPU_MIB=32000

while tmux has-session -t "$SESSION" 2>/dev/null; do
  available_kib=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  gpu_mib=$(nvidia-smi -i 0 --query-gpu=memory.used \
    --format=csv,noheader,nounits 2>/dev/null | tr -d '[:space:]')

  if [[ -n "$available_kib" && "$available_kib" -lt "$MIN_AVAILABLE_KIB" ]]; then
    echo "[$(date -Is)] stopping: available RAM below 80 GiB"
    tmux send-keys -t "$SESSION" C-c
    sleep 5
    tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
    exit 2
  fi
  if [[ "$gpu_mib" =~ ^[0-9]+$ ]] && (( gpu_mib > MAX_GPU_MIB )); then
    echo "[$(date -Is)] stopping: GPU 0 memory above 32,000 MiB"
    tmux send-keys -t "$SESSION" C-c
    sleep 5
    tmux has-session -t "$SESSION" 2>/dev/null && tmux kill-session -t "$SESSION"
    exit 2
  fi
  sleep 30
done

echo "[$(date -Is)] dense/fuse session ended"

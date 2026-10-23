#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
exec 9> outputs/v4/task4/locks/dense_full_gpu3_resume_wait.lock
flock -n 9 || exit 0

while true; do
  if [[ -f outputs/v4/task4/dense_full_gpu3_20261019/manifest.json ]]; then
    echo "Dense full manifest already exists; no resume needed."
    exit 0
  fi

  if PYTHONPATH=. .venv-phase2-batch/bin/python -u scripts/v4/run_single_gpu.py \
      --config configs/v4_phase2_full_a0_gpu3.yaml \
      --report outputs/v4/task4/completion_20261020/reports/dense_full_gpu3_wait_preflight.json \
      --preflight-only; then
    set +e
    PYTHONPATH=. .venv-phase2-batch/bin/python -u scripts/v4/run_single_gpu.py \
      --config configs/v4_phase2_full_a0_gpu3.yaml \
      --report outputs/v4/task4/completion_20261020/reports/dense_full_gpu3_preflight.json \
      -- scripts/v4/build_release_dense.py \
      --root . \
      --sparse-index outputs/v4/task4/sparse_full_20261016 \
      --config configs/v4_phase2_full_a0_gpu3.yaml \
      --output outputs/v4/task4/dense_full_gpu3_20261019
    result=$?
    set -e
    if [[ $result -eq 75 ]]; then
      echo "GPU 3 became busy during launch; retrying preflight in 60 seconds."
      sleep 60
      continue
    fi
    exit "$result"
  else
    result=$?
    if [[ $result -ne 75 ]]; then
      echo "Preflight failed with exit $result; stopping the waiter."
      exit "$result"
    fi
    echo "GPU 3 is busy; next read-only preflight in 60 seconds."
    sleep 60
  fi
done

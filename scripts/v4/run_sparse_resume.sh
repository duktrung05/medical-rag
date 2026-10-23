#!/usr/bin/env bash
set -euo pipefail

task4_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd -- "$task4_root"
task4_log="outputs/v4/task4/sparse_full_20261016.resume_20261019.log"
mkdir -p outputs/v4/task4
exec >> "$task4_log" 2>&1
date --iso-8601=seconds
exec flock -n outputs/v4/task4/sparse_full_20261016.run.lock \
  taskset -c 0-3 nice -n 10 ionice -c 3 \
  env PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv-phase1/bin/python -u scripts/v4/build_release_sparse.py \
  --root . --output outputs/v4/task4/sparse_full_20261016

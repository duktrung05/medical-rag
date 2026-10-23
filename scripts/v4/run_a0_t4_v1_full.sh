#!/usr/bin/env bash
# Dispatcher for the three explicit A0_T4_V1 stages; server-side scorer disabled.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
stage="${1:-}"
case "${stage}" in
  cpu-prepare)
    exec "${repo_root}/scripts/v4/prepare_a0_t4_v1_cpu.sh"
    ;;
  cpu-import-finalize)
    exec "${repo_root}/scripts/v4/import_finalize_a0_t4_v1_cpu.sh"
    ;;
  colab-t4)
    printf '%s\n' 'Run notebooks/a0_t4_v1_full_colab.ipynb on Google Colab Tesla T4.'
    printf '%s\n' 'The server-side CUDA scorer is intentionally not dispatched.'
    ;;
  *)
    printf '%s\n' "Usage: $0 {cpu-prepare|colab-t4|cpu-import-finalize}" >&2
    exit 2
    ;;
esac

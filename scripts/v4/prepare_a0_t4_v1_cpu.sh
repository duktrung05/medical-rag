#!/usr/bin/env bash
# Stage 1: frozen CPU retrieval/fusion and export immutable pair shards.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${repo_root}/.venv-phase2-batch/bin/python"
output="outputs/v4/a0_t4_v1_full_20261022"
bundle_output="outputs/v4/a0_t4_v1_colab_bundle_20261022"

HF_HOME="${repo_root}/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${python_bin}" -m scripts.v4.preflight_a0_t4_v1_cpu_prepare
"${python_bin}" -c 'import json; p=json.load(open("outputs/v4/a0_t4_v1_release_candidate_20261022/cpu_prepare_preflight_report.json")); assert p.get("status")=="GO" and p.get("query_count")==1200'

HF_HOME="${repo_root}/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  "${python_bin}" -m scripts.v4.run_p5_2_split prepare \
  --mode full --config configs/v4_submission_p2_a0.yaml --output "${output}"

"${python_bin}" -m scripts.v4.audit_a0_t4_v1_cpu_pair_progress --source "${output}"

"${python_bin}" -m scripts.v4.package_a0_t4_v1_colab \
  --source "${output}" \
  --preflight outputs/v4/a0_t4_v1_release_candidate_20261022/cpu_prepare_preflight_report.json \
  --output "${bundle_output}"
"${python_bin}" -m scripts.v4.build_a0_t4_v1_colab_notebook

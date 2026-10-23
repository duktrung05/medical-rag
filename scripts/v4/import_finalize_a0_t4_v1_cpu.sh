#!/usr/bin/env bash
# Stage 3: verify/import external scores, then finalize and export on CPU.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"
python_bin="${repo_root}/.venv-phase2-batch/bin/python"
source="outputs/v4/a0_t4_v1_full_20261022"
scores_zip="outputs/v4/a0_t4_v1_colab_import/a0_t4_v1_scores.zip"
import_report="outputs/v4/a0_t4_v1_release_candidate_20261022/colab_score_import_report.json"

"${python_bin}" -m scripts.v4.import_a0_t4_v1_colab_scores \
  --scores-zip "${scores_zip}" --source "${source}"
"${python_bin}" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r.get("status")=="GO" and r.get("score_count")==240000 and r.get("cache_hits")==0, "External score import gate is not GO"' "${import_report}"
"${python_bin}" -m scripts.v4.run_p5_2_split finalize \
  --mode full --config configs/v4_submission_p2_a0.yaml --output "${source}"

sha256sum "${source}/lock.json" "${source}/pair_manifest.json" \
  "${import_report}" "${source}/export/predictions.json" \
  "${source}/export/submission.zip" \
  > "${source}/provenance/release_sha256.txt"

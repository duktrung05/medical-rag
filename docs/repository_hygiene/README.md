# Working-tree checkpoint — 2026-10-23

This is a preservation checkpoint of the local V4 research state, including A0/B0/B1/P5 work. It does not establish release readiness or prove which code produced an older experiment.

## Git boundary

- Branch: `housekeeping/working-tree-20261023`, based on `115f78704769b4ca1389445a4ae65794a80889d7`.
- Local `master` and `origin/master` refs were left unchanged. The original `master` was three commits behind the locally cached `origin/master`; no fetch, pull, merge or push was performed. Three overlapping remote source/document additions were byte-identical to the local files.
- Existing source, configs, notebooks, tests, resources, research reports and review packets are preserved in this checkpoint. Runtime code, scoring logic, frozen configs and experiment result contents were not edited by cleanup.
- `/outputs/` and `.venv-*/` are ignored. All 243 previously tracked output paths are removed from the index; existing files remain at their original disk paths. A checkout alone therefore does not provide the runtime artifacts.
- Five historical Python/shell scripts from `outputs/` are additionally preserved in `archive/runtime_output_scripts_20261007/`, with an original-path/hash manifest. Their original runtime paths remain available locally; archived copies may contain historical path assumptions.
- Fourteen deleted historical documents are preserved in `archive/docs_pre_v4/`: thirteen are byte-identical to their former tracked versions, and one adjusts two relative links. The pre-existing deletions of `docker/constraints.txt`, `docker/requirements.txt`, `experiments/README.md`, and `experiments/experiment_log.csv` are recorded, with original content retained in parent Git history and the backup diff.
- Nine missing `outputs/baseline_bm25_reproduced/dev_repeat/` files were already deleted before cleanup. They are listed in `snapshot.json` and remain recoverable from parent Git history; cleanup did not recreate them.

## Evidence and recovery

The backup directory is `/data_hdd_16t/trungnguyen12/medical-rag/working-tree-backups/20261023_101105/`. It contains the original status, refs, staged/unstaged binary diffs, a 607-file source/changed-artifact archive, and source checksums. The full data/output trees and virtual environments are not backed up. To recover a source file, extract the archive into a separate directory and compare the file before copying it into an active checkout. Parent Git history holds tracked files already missing at the start.

`snapshot.json` records the backup location and the SHA-256 of the compressed complete output inventory. That inventory has 62,380 file entries, totaling 113,847,431,601 logical bytes. It records path, size and nanosecond mtime; it does not hash all large artifacts. `runtime_provenance.jsonl` adds SHA-256 for 338 existing tracked outputs and selected small lock/manifest/config/report files. `source_sha256.json` hashes 463 runtime/config/test/notebook/resource files as observed at cleanup. These are observations of current local files, not retrospective producer attestations.

Large artifacts stay at their original locations, including `outputs/v4/`. Existing code/artifact locks are authoritative where present. No full-index scan, retrieval benchmark, model download, GPU job or submission was started by cleanup.

## Validation

The following command completed in 19.96 seconds using the existing `.venv-phase2-batch` environment:

```bash
.venv-phase2-batch/bin/python -m pytest -q \
  tests/v4 tests/repository \
  tests/workflows/vibiomir/test_crawl_vibiomir_batch.py \
  tests/workflows/vibiomir/test_evidence_selector.py \
  --ignore=tests/v4/test_cli_api_tail.py
```

Result: **258 passed, 3 failed**. The failures are preserved in the backup's `tests-core.log`:

- `test_configs_use_their_own_runtime_or_smoke_schema`: the repository test feeds additional V4 configurations to the legacy `PipelineConfig` schema.
- `test_partial_rerank_retains_authentic_scores_and_null_tail` and `test_rank_integration_writes_replayable_cache_and_empty_submission`: `huggingface_hub.errors.LocalEntryNotFoundError`, with the requested model snapshot unavailable locally and outbound access disabled.

The first full attempt stalled after 25 successful cases at a FastAPI test. An isolated run of `tests/v4/test_cli_api_tail.py` timed out after 25 seconds and was killed after a further three seconds. Its stack is blocked in `starlette.testclient.TestClient.__enter__` / AnyIO. Neither API test completed; `tests/integration/runtime/test_api.py` was not reached. `tests-api.log` records the stack. API behavior is unverified in this environment.

`git diff --cached --check` also reports pre-existing formatting in the newly checkpointed reports, CSV data and source files, including Markdown hard breaks and CSV CRLF. Those bytes are retained for provenance; this checkpoint does not claim a whitespace-clean historical diff. No credentials matching the checked common HF/AWS/GitHub/private-key patterns were found in newly added text files.

## Subsequent experiments

Start from a clean checkout and record `git rev-parse HEAD` with each run. Use a fresh uniquely named directory under `outputs/v4/`, and retain its exact config, input/artifact fingerprints, dependency versions and existing lock metadata. Commit code/config changes separately before starting a run. The current checkpoint identifies the preserved code state; an older result must still be traced through its own lock, source hashes and logs. To share a result, explicitly curate a small report and its provenance under `docs/`; leave large runtime files in artifact storage.

`git status --porcelain=v1 --untracked-files=all` should be empty after the checkpoint. Ignored output/env files are intentionally absent from that status; the inventory and local artifacts remain necessary for reproducing experiments.

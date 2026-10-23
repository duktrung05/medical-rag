# Migration changelog

Baseline: `115f78704769b4ca1389445a4ae65794a80889d7`; work is uncommitted on local `master`. The temporary refactor branch was removed at the user's request.

- Added `src/medrag/` contracts, registry, extraction, loader, passage builder, lexical/dense retrieval, fusion, evidence verification, optional reranker, V5 interfaces, CLI and pilot. New runtime has no legacy imports.
- Added `configs/v4_core.yaml`, `configs/v5_extensions.yaml`, `configs/v4_corpus_registry.json` and `medrag`/`medrag-v4` entrypoints in `pyproject.toml`.
- Added `tests/v4/test_core.py` and `scripts/v4/pilot.py`.
- Added `docs/refactor_v4_v5/` audit, safety and status reports.
- Archived 14 tracked historical root documents to `archive/docs_pre_v4/` after the cleanup request. Updated README and the V4 pilot's source reference; no report content was discarded.
- After explicit user approval of the exact list, deleted nine byte-identical ViBioMIR ZIP copies in `CLEANUP_DUPLICATE_ZIPS.csv` and verified all nine retained copies by SHA-256. `CLEANUP_EXECUTION.json` records the result; 185,091,708 bytes were released.
- Made V4 the default local CLI, API and Docker runtime. Original entrypoint sources and deployment files are archived intact under `archive/legacy_runtime/`. Added a bounded, checksum-verified official-corpus preparation and read-only search path; old ViBioMIR data and indexes were not changed.
- Seven baseline user-modified tracked files were left untouched by this refactor. Local `master` is three commits behind `origin/master`; no pull or push was performed because remote additions overlap working-tree files.

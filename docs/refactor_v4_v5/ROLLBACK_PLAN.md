# Rollback plan

Baseline commit: `115f78704769b4ca1389445a4ae65794a80889d7`. The user directed that V4/V5 become the main architecture, so work was moved to local `master`; the temporary `refactor/v4-v5-greenfield` branch was removed because it pointed to the same commit. No reset or stash was used. Seven tracked files already had user edits at baseline; see `FILE_CLASSIFICATION.csv` rows marked `tracked_modified_before_refactor`. Preserve those edits during any later Git operation.

The default `src/cli.py`, `src/api.py` and Docker service now enter V4. Their original source, Dockerfile and Compose file are preserved in `archive/legacy_runtime/`; the baseline commit also contains them. To run V1/V2, make a separate worktree at the baseline commit and use its commands there. Do not switch this dirty working tree if Git would overwrite user edits. No recrawl is required for the already available historical corpus. The new official corpus has 240,000 URL IDs absent from old local raw data, so old artifacts cannot recreate full official coverage.

Do not treat Hugging Face snapshots as a complete backup. The local raw HTML has 3,547,708 gzip files and may be the only complete copy. `PROTECTED_DATA_MANIFEST.json` lists protected prefixes and verified hashes. Recovery of code from Git is possible; recovery of local-only data is not guaranteed without an independent backup.

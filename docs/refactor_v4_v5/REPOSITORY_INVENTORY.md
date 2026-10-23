# Repository inventory (2026-10-12)

- Checkout: `/data_hdd_16t/trungnguyen12/medical-rag/medical-rag`; baseline `115f78704769b4ca1389445a4ae65794a80889d7`; current branch `master` (local, three commits behind `origin/master`).
- Baseline: 498 tracked paths, 245 untracked paths, 3,549,057 ignored paths. Existing tracked modifications: 7. Nothing was stashed or reset. The temporary refactor branch was removed after switching the same working tree to `master`.
- `data/`: 117 GB; `outputs/`: 2.4 GB; `.git/`: 267 MB. `data/vibiomir/` has 3,548,836 files, including 3,547,708 raw HTML gzip files.
- Directories: `src/` (older generic runtime), `r2ai/` (competition workflow), `scripts/`, `configs/`, `data/`, `tests/`, `outputs/`, `experiments/`, `docs/`. `src/retrieval_v2/` is absent in this checkout.
- Git LFS tracks ViMedAQA files under `data/raw/vimedaqa/`; ViBioMIR bulk data is mostly ignored. No symlinks found beneath `data/`, `outputs/`, `artifacts/`. Sibling Hugging Face caches and `../smoke_test/vibiomir_*` submission/feedback files exist and were not modified.
- `FILE_CLASSIFICATION.csv` is the baseline audit plus cleanup decisions; it is not a live inventory of every new V4 file. Ignored bulk data is covered by protected prefixes in `PROTECTED_DATA_MANIFEST.json`. Four deleted duplicate paths were outside the repository in `../smoke_test/`; see `CLEANUP_EXECUTION.json`.
- Cleanup update: 14 old tracked reports now live in `archive/docs_pre_v4/`. Nine approved duplicate ZIP copies were deleted; retained copies were rehashed. No other ViBioMIR artifact was removed.
- Official corpus audit at `outputs/vibiomir_official_verify_20261012/verification.json`: old local 4,394,718 URL rows; official revision has 1,220,598; 240,000 official URLs absent locally. The prior IDs cannot be treated as the full official corpus.
- Mainline cutover: `src/cli.py`, `src/api.py`, `Dockerfile`, `compose.yaml`, root README and package entrypoints now select V4. The previous source is preserved in `archive/legacy_runtime/`.

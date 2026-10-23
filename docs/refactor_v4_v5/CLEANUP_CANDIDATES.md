# Approved cleanup execution — 2026-10-12

Following the request to reduce old docs and test outputs, 14 tracked historical reports were **moved, not deleted**, from `docs/` to `archive/docs_pre_v4/`. Root `docs/` now contains only newer plans and the V4 refactor reports. `ARCHIVED_DOCS.csv` records each original and destination hash; 13 files are byte-identical, while one has two relative links repaired. README was updated. Two legacy probe scripts still default to writing new review files at their old `docs/` paths when rerun; their historical reports are preserved in the archive.

## Exact ZIP deletion performed

The user explicitly approved the **9 exact paths** in `CLEANUP_DUPLICATE_ZIPS.csv`. Those ZIPs were deleted, freeing **185,091,708 bytes (176.5 MiB)**. Five were under `outputs/submissions/`; four were duplicate ZIPs in sibling `../smoke_test/`. `CLEANUP_EXECUTION.json` records each deleted path, retained path, size and SHA-256. Every retained ZIP was rehashed after deletion and matched its approved checksum.

No other ZIP was deleted. Historical scored baselines, unique ablation submissions, `outputs/vibiomir_official_verify_20261012/`, all raw/extracted data, indexes, ranking caches and scoring feedback remain untouched. The 378 UNKNOWN rows still need provenance review before any later cleanup list.

The original refactor request required approval of a **specific path list before permanent deletion**. That approval was obtained for the nine CSV rows only. Moving archived docs did not discard their content.

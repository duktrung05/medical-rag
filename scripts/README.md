# Scripts

The official project runtime is `python -m src.cli`, which forwards to `src.medrag.cli`. V4 preparation and retrieval commands are documented in the repository [README](../README.md). `scripts/v4/pilot.py` is a wrapper for the small read-only local HTML smoke.

`scripts/runtime/`, `scripts/workflows/`, `scripts/tooling/` and `scripts/experiments/` are historical workflows retained for reproducibility. They are not called by the default V4 CLI, API or Docker service. Their original usage guide is archived at [scripts_README_v1.md](../archive/legacy_runtime/scripts_README_v1.md).

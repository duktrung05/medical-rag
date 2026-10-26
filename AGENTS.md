# Repository Guidelines

## Project Structure & Module Organization

Medical-RAG targets Python 3.11+ and multilingual medical retrieval. Active V4 code lives in `src/medrag/`, organized into data, chunking, retrieval, ranking, evidence, and validation modules. `src/cli.py` delegates to the V4 CLI. Use `scripts/v4/` for competition workflows and `configs/` for YAML/JSON settings. Tests live in `tests/v4/`, alongside older unit, integration, workflow, and repository suites. Consult `docs/V4_WORKSPACE_GUIDE.md` before changing experiment workflows. Treat `archive/` and older source modules as compatibility/history; shared helpers may still be active.

## Build, Test, and Development Commands

Run commands from the repository root:

- `python3 -m pip install -e '.[dev]'`: install the package, pytest, coverage tooling, and Ruff.
- `python3 -m pytest -q tests/v4`: run V4 contracts and regressions.
- `python3 -m pytest -q tests`: run all repository suites.
- `ruff check src/medrag tests/v4`: lint the active runtime and tests.
- `python3 -m src.cli check-config`: validate the configured corpus snapshot and checksum; requires local corpus data.
- `python3 -m src.cli pilot`: run the demo pilot.
- `python3 -m scripts.v4.refine_a0_submission --help`: inspect competition export options.

Install `.[api]` for FastAPI/Uvicorn or `.[v4-models]` for model-backed retrieval. Search requires an existing prepared snapshot.

## Coding Style & Naming Conventions

Use four-space indentation, `snake_case` for modules/functions, `PascalCase` for classes, and descriptive configuration keys. Follow neighboring code for type annotations and `pathlib.Path` usage. Ruff is available through the dev extra; no repository-specific formatter configuration is defined. Keep reusable logic in `src/medrag/` and orchestration in scripts.

## Testing Guidelines

Pytest discovers `test_*.py` files, including `unittest.TestCase` suites. Add focused regression tests for changed behavior, particularly corpus identity, evidence offsets, retrieval, and submission contracts. Use `tmp_path` or temporary directories for fixtures; keep real datasets, outputs, and model caches outside `tests/`. No minimum coverage threshold is configured. Run relevant suites first, then broader tests as needed.

## Commit & Pull Request Guidelines

History mixes short messages with descriptive subjects and `chore:` prefixes; no strict convention is established. Prefer imperative, scoped subjects. PRs should explain the problem, behavior changes, validation commands/results, and related issues. For experiments, record the effective configuration, input hashes, split, and decision in `docs/`.

## Artifact & Configuration Hygiene

Keep generated runs under fresh `outputs/v4/` namespaces. Preserve the frozen S2 baseline, provenance, and index backing stores. Runtime outputs and caches are ignored; commit curated documentation instead. Keep demo configuration separate from competition policy.

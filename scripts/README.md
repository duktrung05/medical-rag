# Scripts

Use the [workspace guide](../docs/V4_WORKSPACE_GUIDE.md) before starting a run.
The CLI/API demo is separate from competition S2.

| Role | Entry points |
|---|---|
| Demo CLI | `python -m src.cli` → `src.medrag.cli` |
| Frozen competition pair/scoring workflow | `v4/run_p5_2_split.py` |
| Fast A0 preparation and completion | `v4/run_p5_fast_a0.py`, `v4/complete_p5_a0.py` |
| Supplemental cached passage scores | `v4/score_a0_remaining_passages.py` |
| Champion S2 refinement/export | `v4/refine_a0_submission.py` |
| R6 audit, pending fixes from review | `v4/audit_r6_stage_oracle.py` |
| R8 lexical experiment, NO_GO | `v4/run_r8_evidence_extraction.py` |
| Reversible cleanup | `tooling/cleanup_v4_workspace.py` |

Run modules from the repository root, for example
`python -m scripts.v4.refine_a0_submission --help`. Use fresh namespaces for new runs.

Other V4 files include shared helpers, index builders and historical experiments.
Their presence does not indicate an approved production branch. Historical scripts
may require archived outputs to be restored first.

`runtime/`, `workflows/` and `experiments/` remain for compatibility and tests;
their [older guide](../archive/legacy_runtime/scripts_README_v1.md) is preserved.

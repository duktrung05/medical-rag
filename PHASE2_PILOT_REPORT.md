# Phase 2 Hierarchical Chunking Pilot Report

Report date: 2026-10-15

## Status

Phase 2 implementation, tests, dry-run and the 1,000-document A/B/C pilot are
complete. Automated integrity validation passed. The run status is intentionally
`pilot_complete_unreviewed`: manual passage review has not been confirmed, and no
full corpus processing was run.

Pilot output: `outputs/vibiomir_phase2_20261015_pilot_v1`

## Files created or changed

- `src/medrag/data/phase2/`: schema, structure recognition, segmentation,
  BGE-M3 tokenizer, passage construction, provenance, validation and evaluation.
- `scripts/workflows/vibiomir/build_phase2_passages.py`: release gate, dry-run,
  stratified pilot, A/B/C execution, atomic writes, checksummed resume and prepared
  streaming full runner.
- `tests/workflows/vibiomir/test_phase2_pipeline.py`: 24 Phase 2 tests.
- `PHASE2_DATA_CONTRACT_AUDIT.md`: current accepted input contract.
- `PHASE2_PILOT_REPORT.md`: this report.

Phase 0 and Phase 1 files were not modified.

## Input gate and dry-run

- Verified `922/922` clean document shards.
- Verified `922/922` PASS allowlist shards.
- Verified `922/922` held-document shards.
- PASS: `1,073,531` unique IDs.
- Held: `2,068`; PASS/held intersection: `0`.
- Official PASS ID-URL mapping errors: `0`.
- Tokenizer: `BAAI/bge-m3` revision
  `5617a9f61b028005a4858fdac845db406aefb181` from local cache.

## Pilot composition

The same deterministic pilot IDs were used for all strategies. Selection was not
the first 1,000 IDs; it was stratified over language, domain, source, length,
heading presence, paragraph profile, mixed-language and measurement signals.

| Stratum | Documents |
| --- | ---: |
| Vietnamese | 492 |
| Chinese | 469 |
| English | 30 |
| Other | 9 |
| Mixed-language signal | 230 |
| Medical measurement signal | 442 |
| HTML | 969 |
| PubMed | 31 |
| Reliable heading detected | 98 |
| Short / medium / long / very long | 219 / 514 / 211 / 56 |

The pilot covers 88 URL domains. A conservative traditional-Chinese character
probe found one document; this is a coverage signal, not a definitive script
classification.

## A/B/C result

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Target / max tokens | 192 / 256 | 256 / 384 | 384 / 512 |
| Overlap tokens | 32 | 48 | 64 |
| Passages | 11,947 | 9,296 | 6,140 |
| Mean tokens | 165.1 | 219.2 | 329.3 |
| Median tokens | 174 | 236 | 362 |
| P95 tokens | 223 | 330 | 408 |
| P99 tokens | 256 | 384 | 512 |
| Passages/document mean | 11.947 | 9.296 | 6.140 |
| Passages/document P95 | 39 | 30 | 20 |
| Small passages | 251 | 385 | 172 |
| Character overlap ratio | 5.47% | 8.93% | 8.73% |
| Chunking time | 19.66 s | 21.93 s | 24.19 s |

Successful resume took about 27 seconds including release checksum verification
and reloaded all completed checkpoints without duplication. The successful pilot
invocation recorded `90.86 s` processing time and `1,150.38 MB` peak RSS. Peak RSS
includes release ID sets and pilot reload; the full runner holds one shard at a time.

## Validation result

- Total passages across strategies: `27,383`.
- Unique passage IDs: `27,383`; collisions: `0`.
- Source mapping errors: `0`.
- Token count errors: `0`.
- Passages above max token budget: `0`.
- Structural coverage failures: `0`.
- Validation failure rows: `0`.
- Remaining `.partial` files: `0`.
- Output artifact/checkpoint checksum mismatches: `0`.
- Tests: `43 passed` across the Phase 1 and Phase 2 workflow suites.

The Phase 2 suite covers deterministic IDs, exact source slices, token budgets,
coverage, overlap, headings, long sentences, VI/ZH/mixed Unicode, negation,
dosages, measurements, no cross-document merge, resume and write-once output.

## Passage review sample

`reports/manual_review_sample.parquet` contains 150 passages prioritized for
negation, constraints, dosage/measurement and Chinese content.

Observed good cases include intact `HbA1c`, `eGFR`, dosage expressions, Vietnamese
negation and Chinese medical sentences with exact source offsets. An actual BPE
edge case in Chinese produced 257 tokens during the first attempt; the run stopped,
the hard-split post-condition was fixed, a regression test was added, and the same
pilot then completed with zero oversized passages.

Observed issues requiring human judgment: some source structures yield tiny
standalone passages such as `1.`, `2.` or `Nam`. These are structurally valid and
preserve source coverage, but may be weak evidence. Automated structural coverage
does not prove semantic completeness.

## Recommendation

Use **Strategy B as the production candidate**, pending manual review. It reduces
passage volume by about 22% versus A while keeping a median of 236 tokens and a
384-token hard ceiling. Strategy C is cheaper in passage count but may dilute
fine-grained retrieval; Strategy A is more granular but creates more passages.

This recommendation is based on structure, size and integrity metrics only. No
valid relevance benchmark was run, so there is not enough evidence to claim that B
has better Recall@K, DOCS_F2, CHUNKS_F2 or reranking quality.

## Commands run

```bash
.venv-phase1/bin/python -m pytest -q \
  tests/workflows/vibiomir/test_phase2_pipeline.py \
  tests/workflows/vibiomir/test_phase1_pipeline.py

.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase2_passages.py \
  --candidate outputs/vibiomir_phase1_full_20261014_v1 \
  --dry-run

.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase2_passages.py \
  --candidate outputs/vibiomir_phase1_full_20261014_v1 \
  --output outputs/vibiomir_phase2_20261015_pilot_v1 \
  --pilot-documents 1000 \
  --run-id phase2_20261015_pilot_v1

.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase2_passages.py \
  --candidate outputs/vibiomir_phase1_full_20261014_v1 \
  --output outputs/vibiomir_phase2_20261015_pilot_v1 \
  --pilot-documents 1000 \
  --run-id phase2_20261015_pilot_v1 \
  --resume
```

## Full corpus command (not run)

After strategy approval, run a new write-once namespace. The runner writes one
output/checkpoint per Phase 1 shard and stops on validation failure.

```bash
.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase2_passages.py \
  --candidate outputs/vibiomir_phase1_full_20261014_v1 \
  --output outputs/vibiomir_phase2_full_<run_id> \
  --run-id <run_id> \
  --full
```

Resume only the identical input/strategy/tokenizer run:

```bash
.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase2_passages.py \
  --candidate outputs/vibiomir_phase1_full_20261014_v1 \
  --output outputs/vibiomir_phase2_full_<run_id> \
  --run-id <run_id> \
  --full \
  --resume
```

Do not run full processing until the 150-passage sample is reviewed and the
strategy is explicitly approved.

## Acceptance review update

The 150-passage paired semantic review is now complete. Its verdict is **not yet
accepted for full build** because it found numeric boundary splitting, Q&A context
loss and source-quality defects in several Phase 1 PASS documents. See
`PHASE2_ACCEPTANCE_REVIEW.md`. Strategy B remains the conditional candidate.

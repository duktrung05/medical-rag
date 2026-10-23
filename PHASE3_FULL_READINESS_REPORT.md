# Medical-RAG V4/V5 — Phase 3 Full Readiness

## Status

**READY FOR CONTROLLED FULL EXECUTION. Full Phase 3 was not started by this remediation.**

Phase 0, Phase 1, Phase 2, and the primary index were not modified. Existing Task
2–6 pilot outputs were not rerun.

## Remediated modules

- Task 3 now has a `--full-disk-backed` path that streams PASS documents shard by
  shard, stores record text/shingles/signatures and LSH bands in SQLite, performs
  global cross-shard bucket grouping, writes deterministic candidate partitions,
  verifies pairs partition by partition, and emits a single compatible Parquet
  relationship artifact.
- Task 5 retains its pilot path and adds a separate `--full-disk-backed` path over
  all 2,951,273 passages. Exact Task 4 identities are joined on disk. LSH,
  adjacent-chunk, and Task 3 document-pair candidate sources are deduplicated on
  disk. Intentional overlap, cross-document overlap, partial reuse, boilerplate,
  and medical guard review remain distinguishable.
- Task 6 adds `--mode full`. Duplicate members, parent quality metadata, and
  passage text are joined through SQLite. Only verified normalized-exact Task 2/4
  groups are eligible. Task 3/5 relationships are scanned as negative gates and
  are never used for transitive grouping.
- Task 4 full mode now validates full Task 2/3 outputs instead of requiring the
  old pilot Task 3 fingerprint.
- A finalizer performs the one final validation requested by Task 7 and publishes
  a reference-based Phase 3 release manifest and summary report.

## Disk-backed and resume behavior

Durable checkpoints cover input shards, LSH band construction, ordered bucket
partitions, candidate generation, pair-verification partitions, and output
completion. Every checkpoint records the input fingerprint, configuration hash,
and shingling version. Completed shards and pair partitions are skipped on
resume. Primary keys make reprocessing an interrupted unit idempotent.

Records, band rows, candidate pairs, and hot-bucket accounting live on disk.
Candidate partitions are derived from stable entity keys, not ingestion order.
Parquet partition files and final outputs use temporary files plus atomic rename.

Hot buckets are not silently truncated. Buckets larger than the configured bound
are classified as `high_frequency_template_filtered`; bucket size, theoretical
pair count, examined count, and policy are persisted. Their unexamined pairs are
reported as candidate-recall exclusions.

## Resource controls

Default controlled limits are recorded in
`configs/vibiomir/phase3_full_scalability_v1.json`:

- RSS: 16 GiB.
- Task 3 temporary disk budget: 256 GiB.
- Task 5 temporary disk budget: 512 GiB.
- Task 6 temporary disk budget: 128 GiB.
- Batch size: 512 records.
- No automatic multiprocessing or CPU saturation.

Disk preflight runs before ingestion. Current free workspace disk observed during
remediation was approximately 3.0 TiB. A budget violation raises a resumable,
fail-closed error rather than publishing a complete result.

## Focused regression results

- 52 tests passed; 0 failed.
- Cross-shard candidate discovery: PASS.
- Disk-backed grouping and cross-band deduplication: PASS.
- Hot-bucket accounting: PASS.
- Completed-shard and candidate-stage resume/idempotency: PASS.
- Determinism across shard ingestion order: PASS.
- Pilot in-memory versus full disk-backed candidate consistency: PASS.
- Known near-duplicate preservation: PASS.
- Medical dosage mismatch safety: PASS.
- Passage medical guard retained as review-only: PASS.
- Full canonical disk-backed ranking/membership: PASS.
- RSS budget enforcement: PASS.

## Bounded synthetic stress result

The stress run used 20,000 records across 40 shards:

- Status: PASS.
- Known cross-shard pairs recovered: 20/20.
- Elapsed time: 27.534 seconds.
- Peak RSS growth: 20 MiB; final process peak RSS: 39.25 MiB.
- Disk used: 13,548,521 bytes (12.92 MiB).
- Band rows: 80,000.
- Deduplicated candidates: 20.
- Hot buckets: 1.
- Hot-bucket potential pairs: 1,953; examined: 0; all explicitly reported.

This synthetic result validates bounded state and control flow. It is not a
throughput forecast for real medical text, whose shingle volume and candidate
density are higher.

## Acceptance gate

`configs/vibiomir/phase3_pilot_acceptance_20261015.json` is `accepted`. The gate
was derived from existing PASS validations for Tasks 2–6, zero labeled Task 5
false positives, immutable Phase 1/2 and Task 1 fingerprints, the focused test
result, and the bounded stress result. It explicitly does not assert exhaustive
candidate recall.

## Candidate recall limitations

MinHash/LSH candidate generation is probabilistic. No exhaustive labeled pair
set or brute-force corpus comparison exists, so neither Task 3 nor Task 5 may
claim complete recall. In addition, high-frequency buckets are deliberately
filtered to protect resources; their theoretical and examined pair counts are
reported. The final release manifest therefore sets
`candidate_recall_complete=false`.

## Planned full commands

Run sequentially; use a new namespace if a completed output already exists.

```bash
.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase3_exact_document_groups.py \
  --full \
  --pilot-acceptance configs/vibiomir/phase3_pilot_acceptance_20261015.json \
  --output outputs/vibiomir_phase3_20261015_task2_full_v1

.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase3_near_document_pairs.py \
  --full-disk-backed \
  --task2 outputs/vibiomir_phase3_20261015_task2_full_v1 \
  --candidate-partitions 256 --maximum-bucket-size 200 --batch-rows 512 \
  --rss-limit-gib 16 --temporary-disk-budget-gib 256 \
  --output outputs/vibiomir_phase3_20261015_task3_full_v1

.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase3_passage_content_groups.py \
  --full --resume \
  --pilot-acceptance configs/vibiomir/phase3_pilot_acceptance_20261015.json \
  --task2-groups outputs/vibiomir_phase3_20261015_task2_full_v1/document_dedup/exact_document_groups.parquet \
  --task3-pairs outputs/vibiomir_phase3_20261015_task3_full_v1/document_dedup/near_duplicate_pairs.parquet \
  --output outputs/vibiomir_phase3_20261015_task4_full_v1

.venv-phase1/bin/python scripts/workflows/vibiomir/build_phase3_passage_overlap_pairs.py \
  --full-disk-backed --resume \
  --task4 outputs/vibiomir_phase3_20261015_task4_full_v1 \
  --task3-pairs outputs/vibiomir_phase3_20261015_task3_full_v1/document_dedup/near_duplicate_pairs.parquet \
  --candidate-partitions 512 --maximum-bucket-size 200 --batch-rows 512 \
  --rss-limit-gib 16 --temporary-disk-budget-gib 512 \
  --output outputs/vibiomir_phase3_20261015_task5_full_v1

.venv-phase1/bin/python scripts/workflows/vibiomir/select_phase3_canonical_content.py \
  --mode full --resume \
  --task2 outputs/vibiomir_phase3_20261015_task2_full_v1 \
  --task3 outputs/vibiomir_phase3_20261015_task3_full_v1 \
  --task4 outputs/vibiomir_phase3_20261015_task4_full_v1 \
  --task5 outputs/vibiomir_phase3_20261015_task5_full_v1 \
  --rss-limit-gib 16 --temporary-disk-budget-gib 128 \
  --output outputs/vibiomir_phase3_20261015_task6_full_v1

.venv-phase1/bin/python scripts/workflows/vibiomir/finalize_phase3_release.py \
  --task2 outputs/vibiomir_phase3_20261015_task2_full_v1 \
  --task3 outputs/vibiomir_phase3_20261015_task3_full_v1 \
  --task4 outputs/vibiomir_phase3_20261015_task4_full_v1 \
  --task5 outputs/vibiomir_phase3_20261015_task5_full_v1 \
  --task6 outputs/vibiomir_phase3_20261015_task6_full_v1 \
  --output outputs/vibiomir_phase3_20261015_final_release_v1
```

For Task 3, Task 5, or Task 6, add `--resume` after an interrupted run. Task 2
must use a fresh namespace if interrupted before completion.

## Remaining limitations and operational risks

- Real-corpus candidate density and runtime are unknown until the controlled full
  run. The implementation is bounded and resumable, but no wall-clock completion
  estimate is asserted from the synthetic test.
- High-frequency bucket filtering intentionally trades candidate recall for safe
  execution. Coverage loss is measurable but not recoverable without a separate
  reviewed template-cluster policy.
- SQLite database and WAL size depend on real shingle counts. The declared disk
  budgets are conservative operational caps, not predicted usage.
- Task 2 exact-document full execution is atomic but does not resume mid-build;
  this is unchanged because the remediation scope required checkpoint changes for
  Task 3 and Task 5.

No remaining implementation blocker was identified for beginning the controlled
full sequence. Final release must not be marked complete if any task stops on a
resource budget, reports incomplete coverage state, or fails validation.

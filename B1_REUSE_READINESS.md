# B1 incremental submission readiness

**B1_REUSE_READINESS: PARTIAL**  
Full 1,200-query submission readiness: **NO-GO — RETRIEVAL_REQUIRED for queries without archived branch hits.**

This audit used code, small metadata and three completed full A0 shards (1, 4, 53), plus one archived pilot trace. The tmux session `p5_cpu` exists. A directory-only snapshot found 110 completed A0 JSON shards; this is a changing preparation count, not a full-run readiness claim. No A0 file/config/checkpoint was edited, no process was controlled, no SQLite was opened, and no retrieval, model loading or inference was executed.

## Algorithm and reuse boundary

The primary B1 implementation is the frozen breadth-preserving intersection described in `B1_BUDGET_MATCHED_DESIGN.md` and implemented by `audit_b1_feasibility.py`, not pure passage-first B0. Its standalone policy is recorded in `configs/v4_b1_reuse_policy.json`.

| Stage | A0 | B1 |
|---|---|---|
| Raw retrieval | Document BM25 400, passage BM25 200, dense 200 | Reuse exact same raw branch hits |
| Fusion | RRF k=60, fused limit 400, parent aggregation | Preserve A0 parent pool; additionally fuse passage BM25+dense for B0 top-200 intersection |
| Pair allocation | First fused evidence, 200 parents, one pair each | P2-A1 dense representative when exact pair is in B0 top-200; otherwise original A0 pair |
| Rerank order | Score descending, original parent retrieval score descending, parent ID ascending | Same |
| Chunk selection | Independent 200-doc / 60-chunk cap, one chunk per parent, adjacent source context | Same |
| Validation/export | Source validation, canonical BTC payload and ZIP | Reuse same functions, B1 namespace |

Raw branch ranks and dense per-passage similarity cannot be recovered from the current full shards' aggregated parent ranks. `run_p5_2_split.prepare` writes candidates, diagnostics and 200 pairs but discards branch hits. Candidates after fusion alone cannot prove the frozen B0 top-200 membership or dense representative. Query 53 can reuse an archived trace because its query hash and all 200 original A0 pair identities/input hashes match. Queries 1 and 4 have no such trace and are marked `RETRIEVAL_REQUIRED`. The adapter does not replace B1 with A0 when these inputs are missing.

## Measured overlap and score accounting

| Scope | Reconstructable | Parent overlap | Exact pair overlap | T4 scores reused now | Missing T4 scores now |
|---|---:|---:|---:|---:|---:|
| Full shards sampled: 1, 4, 53 | 1/3 | 200/200 for query 53 | 189/200 for query 53 | 0 | 200 for query 53 |
| Full 1,200 queries | At least 1 verified; remainder not scanned | Unmeasured | Unmeasured | Unmeasured | Unknown |

Query 53 substitutes 11 passages. After the matching A0 T4 scores are sealed, its 189 identical pairs are eligible for score reuse and only 11 substitution pairs require fresh inference. At present all 200 scores are pending under the T4-only rule. The historical cached pilot scores are deliberately ineligible for this full T4 preparation. Do not extrapolate 11 missing pairs/query across 1,200 queries.

Archived six-query diagnostics reported 1,163/1,200 pair overlap and 37 substitutions, with 100% parent breadth. Those are development diagnostics, not full-run estimates. Neither historical relevance labels nor review packets are inputs to the adapter.

## Adapter and bounded verification

`scripts/v4/prepare_b1_reuse.py` is a standard-library-only planner. It requires explicit query IDs, reads JSON files, validates query/lock/input/cache-key hashes, deduplicates canonical identities, and writes only B1 staging artifacts. Each query checkpoint binds the A0 shard and branch trace SHA-256. Resume accepts identical existing checkpoints and refuses changed data. It opens no shared score cache.

Measured command:

```bash
python3 scripts/v4/prepare_b1_reuse.py \
  --source outputs/v4/a0_t4_v1_full_20261022 \
  --output outputs/v4/b1_reuse_readiness_20261022/full_with_existing_traces \
  --query-ids 1 4 53 \
  --traces outputs/v4/parent_first_evidence_pilot_20261021_v2/queries
```

Machine report: `outputs/v4/b1_reuse_readiness_20261022/full_with_existing_traces/readiness.json`. B1 plan and missing-only T4 request list: `plans/53.json`, fields `shard`, `reused_scores`, `missing_t4_pairs`. This file is a staging envelope, not an executable inference job. All 200 generated B1 identities matched the frozen query-53 plan exactly, and the A0 shard SHA was verified unchanged after the test.

Score reuse requires sealed A0 score checkpoints, matching pair/input/contract hashes, pinned T4 runtime versions, model revision, tokenizer package version and model snapshot hashes. The current lock fingerprint is `42350936c327dc01630472257a4444a8df27713140b6fc7e518ba01ad4239c41`. The adapter rejects incompatible sealed provenance and never imports scores directly from SQLite. Missing T4 requests contain no inferred/cached CPU scores.

## Remaining path to a separate B1 ZIP

1. Obtain exact raw branch artifacts for remaining queries. The current A0 run does not persist them. Wait for A0 priority work to finish before any separately authorized retrieval; do not change or restart the running A0 job to add logging.
2. Reconstruct and validate all 1,200 B1 pair plans. Materialize plans in a separate B1 split-run namespace with its own lock, pair manifest, SHA-256 ledger and cache. The current adapter produces staging envelopes; a full handoff/materialization gate remains required.
3. After A0 T4 import/seal, reuse exact compatible scores and form missing-only T4 jobs. Verify identical tokenizer/model snapshot hashes and scoring runtime for fresh scores. Full missing-pair count is unknown until step 2.
4. Assemble complete B1 score shards bound to B1 pair shard hashes. Use B1-only checkpoint/resume and cache paths. The A0-specific importer/sealer enforce full A0 coverage and cannot accept a missing-only B1 job unchanged; a B1 handoff/import wrapper is still required.
5. Call the existing `finish_pipeline` / `Pipeline.finish_rerank`, `validate_selected`, `canonical_payload`, `package_zip` and `validate_zip` through the B1 wrapper. These functions already implement CPU finalization, adjacent source context, grounding and BTC export. `run_p5_2_split.finalize` is A0-lock/pilot-parity-specific and must not be pointed at the A0 namespace for B1 staging.
6. Record independent relevance decision and submission authorization separately. Preparation does not promote B1; the current quality decision remains `BLOCKED_BY_EVALUATION`.

No full B1 retrieval, GPU job, model inference, finalization or submission export was run in this task.

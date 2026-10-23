# Phase 2 Semantic Acceptance Review

> Superseded for Chunker V2 regression and retrieval results by
> `PHASE2_V2_REGRESSION_AND_RETRIEVAL_REPORT.md`.

Review date: 2026-10-15

## Verdict

**Review complete; not accepted for full build yet.** All 150 selected passages
were inspected with their exact source text before and after the chunk boundary.
The paired review supports Strategy B as the production candidate, but it also
found boundary and source-quality defects that require a new pilot before full
processing.

No full Phase 2 build was run.

## Review set

The sample contains 50 shared document/situation cases. Each case contributes the
corresponding A, B and C passage, for 150 reviewed passages.

| Dimension | Cases |
| --- | ---: |
| VI / EN / ZH | 18 / 16 / 16 |
| HTML / PubMed | 34 / 16 |
| Long / short-medium document | 25 / 25 |
| Negation | 8 |
| Dosage | 8 |
| Lab result | 8 |
| Treatment condition | 8 |
| Question-answer | 8 |
| Boundary stress | 10 |

Every row includes full `passage_text`, `context_before`, `context_after`, exact
offsets, overlap sizes, source metadata, ratings and a passage-specific note.
Source-slice errors in the review set: `0`.

## Manual results

| Metric | A | B | C |
| --- | ---: | ---: | ---: |
| Complete | 29/50 | 35/50 | 40/50 |
| Complete or mostly complete | 39/50 | **44/50** | 43/50 |
| Incomplete | 11/50 | **6/50** | 7/50 |
| Good boundary | 29/50 | **35/50** | 25/50 |
| Poor boundary | 10/50 | 9/50 | **7/50** |
| No context loss | 29/50 | 35/50 | **40/50** |
| Major context loss | 9/50 | **6/50** | 7/50 |
| Excessive overlap | 0/50 | **1/50** | 16/50 |

On the same 50 cases, B was semantically better than A in 13, equal in 35 and
worse in 2. C was better than B in 6, equal in 42 and worse in 2. C gains more
fully self-contained passages, but excessive overlap and broader topic mixing are
material disadvantages.

## Findings

### Chunk-boundary defects

- Decimal/measurement splitting occurred in four cases. Examples include `5.963`
  becoming a passage starting at `963`, `16.12/mm3` becoming `12/mm3`, and a
  confidence interval starting at `9 to -24.7` without its leading value.
- A short tail can remain isolated instead of being attached forward. In
  `case-042`, Strategy A produced a 22-token quote while the actionable `5 mm`
  intervention threshold began in the following passage.
- Q&A pairing is not guaranteed. In `case-039`, A and C contain the diagnostic
  question/options while the confirmed answer begins outside the passage. B keeps
  that specific question and answer together.
- Some boundaries begin with anaphora or a partial list, such as "This pain..." or
  a later numbered item without its heading.

### Source-quality defects

Six of 50 cases expose Phase 1 PASS documents that are structurally valid but poor
semantic evidence:

- unrelated news teasers plus encoded noise (`case-011`);
- dated/truncated news listings (`case-018`);
- concatenated "view full text" snippets (`case-023`);
- bibliography-only content (`case-031`);
- conference table-of-contents titles fused with author names (`case-036`);
- recommendation/navigation questions without answers (`case-040`).

Larger chunks do not repair these documents. They usually aggregate more unrelated
content, which explains several C overlap/focus failures.

### Special-content integrity

Negations, dosage strings, inequalities and laboratory values remained byte-exact.
The main failure mode was missing neighboring context, not character mutation.
PubMed abstracts generally had stronger section boundaries than HTML. Across 48
PubMed passages, 38 had good boundaries; across 102 HTML passages, only 51 did.

The automatic signal used to stratify the sample also produced semantic-category
false positives: a staff role containing "xet nghiem", a prolactin result selected
as dosage, rhetorical punctuation selected as Q&A, and navigation questions.
These were retained and reported rather than silently replaced.

## Retrieval experiment

No retrieval experiment was run because no suitable labels join to this pilot:

- Official ViBioMIR `query.parquet` contains 1,200 queries but no relevance labels.
- Legacy scored feedback is tied to legacy chunks, has unverified server
  provenance, and contains **zero positive documents overlapping the 1,000 pilot
  IDs**.
- MedQuAD qrels refer to a different corpus.

Using any of these as Phase 2 qrels would not measure chunk-size impact and could
produce a false A/B/C conclusion. The primary index was not changed.

## Production recommendation

Retain **Strategy B: target 256, max 384, overlap 48** as the candidate. It has the
best combined usable-completeness rate, best good-boundary count, one excessive
overlap case, and 22% fewer passages than A.

Before another pilot:

1. Make sentence detection decimal- and measurement-aware.
2. Add forward attachment for a short orphan tail when backward merge exceeds the
   token budget.
3. Preserve Q&A pairs and heading/list ownership when they fit under `max_tokens`.
4. Add a Phase 2 semantic eligibility ledger, or issue a new Phase 1 release, for
   teaser/navigation/bibliography/table-of-contents documents.
5. Replace keyword-only special-case sampling with validated pattern classes.
6. Re-run the same 50 paired cases plus a fresh holdout before approving full.

## Artifacts

- Review table: `reports/acceptance_review/phase2_semantic_review_150.parquet`
- CSV export: `reports/acceptance_review/phase2_semantic_review_150.csv`
- Human-readable packet: `reports/acceptance_review/review_packet.md`
- Review manifest: `reports/acceptance_review/review_manifest.json`
- Machine summary: `reports/acceptance_review/review_summary.json`
- Manual annotations: `configs/vibiomir/phase2_review_annotations_20261015.json`

The review manifest records `reviewed_passages: 150` and
`manual_review_complete: true`. This means the selected sample was actually read
and rated; it does not mean the entire pilot or full corpus has been semantically
reviewed.

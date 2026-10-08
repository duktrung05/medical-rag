# Annotation guide

For every candidate, replace `relevant: null` with `true` or `false`. Judge whether
the passage contains evidence that directly addresses the Vietnamese query; being
about the same disease or body part is not sufficient. Keep source text and IDs
unchanged. The dev and holdout query IDs must remain disjoint.

After judging a query, optionally add one `error_category` field to its top-level
object using one of these values:

- `no_relevant_document_in_candidate_pool`
- `relevant_document_but_no_relevant_passage`
- `relevant_passage_below_reranker_or_selector_gate`
- `extraction_or_boilerplate_failure`
- `passage_boundary_or_missing_context`
- `matching_rule_uncertain`
- `no_pipeline_error_observed`

These labels describe the cached candidate pool. They do not establish recall over
the full 4.39M-document corpus.

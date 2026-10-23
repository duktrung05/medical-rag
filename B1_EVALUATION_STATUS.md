# B1 independent relevance evaluation status

**Decision: `BLOCKED_BY_EVALUATION`**  
**Deadline: 2026-10-30**

The six development queries have a frozen, blinded packet of 23 pairs. No human relevance labels have been supplied: 0 reviewed, 23 pending. A0 remains frozen and B1 has not been promoted.

## Preparation and validation

- Source packet: `B1_FINAL_OUTPUT_REVIEW.csv`; private reconciliation map: `outputs/v4/b1_final_output_evaluation_20261022/review_item_map.json`. Keep the map and `B1_FINAL_OUTPUT_DELTA.json` away from reviewers until labels are locked.
- The packet SHA-256 matches the frozen `review_packet_manifest.json`. All 23 IDs are unique and match the private map and delta. The six query texts are nonempty and consistent within each query.
- Left and Right titles and complete exported contexts match the private map exactly. Every mapped context has its recorded source-span grounding checks true, and its character length matches the recorded source span. These checks use the frozen replay's source-grounding provenance; this preparation did not recrawl source pages or rerun retrieval.
- 16 pairs compare changed passages for a shared parent; 7 compare top-60 membership swaps. Left/Right orientation is swapped in 11 of 23 pairs. The reviewer packet and labels contain no arm assignment, rank, reranker score, or document ID. Opaque review IDs are only join keys.

## Reviewer handoff

Give reviewers only `B1_REVIEW_PACKET.md` and `B1_REVIEW_LABELS_TEMPLATE.csv`. The Markdown includes each full question and both complete exported passage contexts. The CSV has separate 0–3 relevance grades, pairwise preference, wrong-patient, wrong-number, negation, missing-context and uncertainty fields, plus reviewer identity and UTC timestamp. Reviewers should fill all required fields without changing row order or IDs; `na` applies only to number and negation fields.

After genuine labels are recorded and locked, run:

```bash
python3 B1_RELEVANCE_EVAL.py --labels B1_REVIEW_LABELS_TEMPLATE.csv --output B1_RELEVANCE_RESULTS.json
```

The script validates the frozen packet and labels, privately maps Left/Right to A0/B1, and writes pairwise wins, losses, ties, uncertain IDs, relevance-grade sums, safety regressions, and outcomes per query. Blank or invalid rows are reported separately. The result retains `BLOCKED_BY_EVALUATION`; no automatic promotion is performed. The script does not infer blank labels or calculate CHUNKS F2.

## Ranking-only changes and scoring limit

The frozen delta records 181 shared outputs whose evidence content is identical but rank changed. They were excluded from the 23 content-review pairs because their passage relevance is unchanged. The local validator accepts reordered arrays and the internal F2 calculation is membership based. The official BTC evaluator code, cutoff and rank weighting are not available in this repository, so official rank sensitivity is **UNVERIFIED**. If the official contract uses rank, evaluate these 181 movements separately with the official scorer or complete rank-aware judgments; do not treat this pairwise packet as a score-equivalence proof.

The 23 paired judgments cover only changed outputs on six development queries, not all top-60 results or an independent holdout. CHUNKS F2 remains uncomputed, and no comparative relevance or safety conclusion is available until humans provide valid labels. No retrieval, GPU, or model inference was used for this preparation.

"""Check whether frozen S2 parent logits alone can change any chunk slot."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path

from scripts.v4.run_r8_evidence_extraction import REFINED, ROOT, S2, load_s2
from src.medrag.data.pair_lineage import write_once
from src.medrag.data.phase2_batch import sha256_file

OUTPUT = ROOT / "outputs/v4/s2_allocation_signal_audit_20261025"


def run(output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    manifest_path = REFINED / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "PASS" or manifest["query_count"] != 1200:
        raise ValueError("Refined champion precursor is incomplete")
    frozen = load_s2()
    if set(frozen) != set(range(1, 1201)):
        raise ValueError("S2 query coverage differs")
    frozen_sha = sha256_file(S2)
    rows, counts = [], Counter()
    for qid in range(1, 1201):
        path = REFINED / "final" / f"{qid}.json"
        if sha256_file(path) != manifest["final_sha256"][str(qid)]:
            raise ValueError(f"Frozen refined query hash changed: {qid}")
        final = json.loads(path.read_text())
        ranked = final["selected"]
        rank_by_doc = {int(row["official_id"]): rank for rank, row in enumerate(ranked)}
        score_by_doc = {int(row["official_id"]): float(row["rerank_score"]) for row in ranked}
        docs = [int(doc) for doc in frozen[qid]["relevant_docs"]]
        chunks = [int(row["doc_id"]) for row in frozen[qid]["relevant_chunks"]]
        if (len(rank_by_doc) != len(ranked) or len(ranked) > 200 or len(ranked) < 60
                or len(chunks) != 60 or len(set(chunks)) != 60 or len(docs) != len(set(docs))
                or not set(chunks) <= set(docs) <= set(rank_by_doc)
                or any(not math.isfinite(score) for score in score_by_doc.values())):
            raise ValueError(f"Frozen parent identity/cardinality differs: {qid}")
        outsiders = [doc for doc in docs if doc not in chunks]
        candidate = max(outsiders, key=lambda doc: (score_by_doc[doc], -rank_by_doc[doc])) if outsiders else None
        victim = min(chunks, key=lambda doc: (score_by_doc[doc], -rank_by_doc[doc]))
        gap = score_by_doc[candidate] - score_by_doc[victim] if candidate is not None else None
        counts["queries"] += 1
        counts["s2_chunk_parents_first_60_ranked"] += set(chunks) == set(list(rank_by_doc)[:60])
        counts["no_eligible_outsider"] += candidate is None
        counts["strict_positive_logit_gap"] += gap is not None and gap > 0
        counts["zero_logit_gap"] += gap is not None and gap == 0
        counts["negative_logit_gap"] += gap is not None and gap < 0
        rows.append({"query_id": qid, "ranked_parents": len(ranked), "s2_docs": len(docs),
                     "outside_chunk_parents_in_s2_docs": len(outsiders),
                     "baseline_victim_doc": victim, "baseline_victim_rank": rank_by_doc[victim] + 1,
                     "outsider_doc": candidate, "outsider_rank": rank_by_doc[candidate] + 1 if candidate else None,
                     "outsider_minus_victim_logit": gap})
    if sha256_file(S2) != frozen_sha:
        raise ValueError("S2 ZIP changed during audit")
    output.mkdir(parents=True)
    with (output / "per_query.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "format": "s2-allocation-logit-signal-audit-v1", "status": "PASS",
        "scope": "Frozen S2 docs and already cached parent reranker logits; score-only one-slot swap diagnostic",
        "counts": dict(counts),
        "s2_zip_sha256": frozen_sha,
        "refined_manifest_sha256": sha256_file(manifest_path),
        "per_query_sha256": sha256_file(output / "per_query.csv"),
        "interpretation": ("A strict positive gap is required to replace a baseline chunk parent "
                           "using the same parent logit. Zero gaps are ties, not evidence gain. "
                           "This audit does not test a learned evidence score or official F2."),
    }
    write_once(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    run(parser.parse_args().output)

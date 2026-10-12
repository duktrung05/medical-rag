"""Additional bounded extracted-text validation; does not modify source Parquet."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]


def audit(path: Path) -> dict:
    titles = defaultdict(set)
    counts = Counter()
    scripts = Counter()
    lengths = []
    duplicate_text = Counter()
    records = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512):
        for row in batch.to_pylist():
            records += 1
            doc_id = row["doc_id"]
            title = row.get("title") or ""
            body = row.get("text") or ""
            titles[doc_id].add(title)
            counts[doc_id] += 1
            lengths.append(len(body.split()))
            duplicate_text[(doc_id, body)] += 1
            han = sum("\u3400" <= c <= "\u9fff" for c in body)
            latin = sum("a" <= c.casefold() <= "z" for c in body)
            scripts["han_dominant" if han > latin else "latin_dominant" if latin else "other"] += 1
    ordered = sorted(lengths)
    return {"rows": records, "documents": len(counts),
            "title_inconsistent_documents": sum(len(v) > 1 for v in titles.values()),
            "duplicate_text_within_document": sum(n - 1 for n in duplicate_text.values()),
            "documents_with_one_chunk": sum(n == 1 for n in counts.values()),
            "documents_with_gt20_chunks": sum(n > 20 for n in counts.values()),
            "max_chunks_per_document": max(counts.values()),
            "word_length_p10_p50_p90": [ordered[int((len(ordered) - 1) * q)] for q in (.1, .5, .9)],
            "short_lt10_words": sum(n < 10 for n in lengths),
            "script_proxy_counts": dict(scripts),
            "script_note": "Unicode character-count proxy, not language labels",
            "scope": "pilot sample only"}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=Path, default=ROOT / "outputs/retrieval_v2/pilot")
    args = ap.parse_args()
    report = audit(args.pilot / "pilot_source.parquet")
    (args.pilot / "V2_DATA_AUDIT.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

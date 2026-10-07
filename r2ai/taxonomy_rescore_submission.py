"""Create a taxonomy-rescored ZIP from an existing ViBioMIR submission.

Only candidate document order and the matching chunk order change. Chunk text
is preserved byte-for-byte after JSON decoding from the baseline submission.
"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq

from r2ai.taxonomy import DEFAULT_CONFIG, Taxonomy, rrf_rerank


def read_submission(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        names = [name for name in archive.namelist() if not name.endswith("/")]
        if len(names) != 1:
            raise ValueError(f"Expected one JSON file in ZIP, got {names}")
        rows = json.loads(archive.read(names[0]))
    if not isinstance(rows, list):
        raise ValueError("Submission JSON must be a list")
    return rows


def annotate_evidence(rows: list[dict], taxonomy: Taxonomy) -> dict[int, dict]:
    """Aggregate taxonomy from exact evidence chunks already in the baseline ZIP."""
    metadata: dict[int, dict] = {}
    for record in rows:
        for chunk in record.get("relevant_chunks", []):
            doc_id = int(chunk["doc_id"])
            item = metadata.setdefault(doc_id, {"concepts": set(), "intents": set(), "specialties": set()})
            text = str(chunk.get("chunk_text") or "")
            mentions = taxonomy.concepts_in(text)
            for mention in mentions:
                if mention["assertion"] != "negated":
                    item["concepts"].add(mention["concept_id"])
            item["intents"].update(taxonomy.intents_in(text, mentions))
    by_id = {str(row["id"]): row for row in taxonomy.concepts}
    for item in metadata.values():
        item["specialties"] = {
            specialty for concept_id in item["concepts"]
            for specialty in by_id[concept_id].get("specialties", [])
        }
    return metadata


def rescore(rows: list[dict], metadata: dict[int, dict], taxonomy: Taxonomy,
            weight: float) -> tuple[list[dict], dict[str, int]]:
    output = []
    stats = {"queries": len(rows), "metadata_docs": len(metadata), "query_doc_matches": 0}
    for row in rows:
        qid = row.get("id")
        docs = row.get("relevant_docs")
        chunks = row.get("relevant_chunks")
        if not isinstance(docs, list) or not isinstance(chunks, list):
            raise ValueError(f"Query {qid}: missing docs/chunks arrays")
        doc_ids = [int(doc) for doc in docs]
        profile = taxonomy.profile(str(row.get("query", ""))) if "query" in row else None
        # Submissions do not carry query text. Attach query texts before calling
        # this function so the same profile can be used for all of its candidates.
        if profile is None:
            raise ValueError("Submission rows need query text for taxonomy rescoring")
        stats["query_doc_matches"] += sum(
            1 for doc in doc_ids
            if set(profile["concepts"]) & set(metadata.get(doc, {}).get("concepts", []))
        )
        reranked_docs = rrf_rerank(doc_ids, profile, metadata, weight=weight)
        positions = {doc: rank for rank, doc in enumerate(reranked_docs)}
        indexed_chunks = list(enumerate(chunks))
        # Keep original order for chunks from the same document; align document
        # groups to the newly reranked document order.
        indexed_chunks.sort(key=lambda pair: (positions.get(int(pair[1]["doc_id"]), len(positions)), pair[0]))
        output.append({
            "id": int(qid),
            "relevant_docs": reranked_docs,
            "relevant_chunks": [chunk for _, chunk in indexed_chunks],
        })
    return output, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--query-parquet", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weight", type=float, default=0.25)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = read_submission(args.baseline)
    query_rows = pq.read_table(args.query_parquet, columns=["id", "query"]).to_pylist()
    query_by_id = {int(row["id"]): str(row["query"]) for row in query_rows}
    for record in records:
        qid = int(record["id"])
        if qid not in query_by_id:
            raise ValueError(f"Unknown query ID {qid}")
        record["query"] = query_by_id[qid]
    taxonomy = Taxonomy.load(args.config)
    metadata = annotate_evidence(records, taxonomy)
    print(f"queries={len(records):,}, evidence documents with metadata={len(metadata):,}", flush=True)
    rescored, stats = rescore(records, metadata, taxonomy, args.weight)
    for record in rescored:
        record.pop("query", None)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(rescored, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with zipfile.ZipFile(args.out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        archive.writestr("predictions.json", payload)
    stats.update({"weight": args.weight, "zip_bytes": args.out.stat().st_size,
                  "taxonomy_version": taxonomy.version})
    stats_path = args.out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

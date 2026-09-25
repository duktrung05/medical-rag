"""Submission validation engine strictly enforcing all competition constraints."""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Set, Union

from src.data.loader import DataLoader, DocumentChunkMap
from src.data.validator import validate_queries_integrity

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass


@dataclass
class ValidationReport:
    is_valid: bool
    total_queries: int
    expected_queries: int
    duplicate_queries: List[str] = field(default_factory=list)
    missing_queries: List[str] = field(default_factory=list)
    extra_queries: List[str] = field(default_factory=list)
    unknown_docs: List[str] = field(default_factory=list)
    unknown_chunks: List[str] = field(default_factory=list)
    internal_duplicates: List[str] = field(default_factory=list)
    parent_violations: List[str] = field(default_factory=list)

    def print_summary(self) -> None:
        print("=" * 60)
        print("SUBMISSION VALIDATION REPORT")
        print("=" * 60)
        print(f"Queries: {self.total_queries}/{self.expected_queries}")
        print(f"Unknown docs: {len(self.unknown_docs)}")
        print(f"Unknown chunks: {len(self.unknown_chunks)}")
        print(f"Duplicate queries: {len(self.duplicate_queries)}")
        print(f"Parent violations: {len(self.parent_violations)}")
        if self.missing_queries:
            print(f"Missing queries: {len(self.missing_queries)}")
        if self.extra_queries:
            print(f"Extra queries: {len(self.extra_queries)}")
        if self.internal_duplicates:
            print(f"Internal list duplicates: {len(self.internal_duplicates)}")
        print("-" * 60)
        if self.is_valid:
            print("VALID SUBMISSION [OK]")
        else:
            print("INVALID SUBMISSION [FAILED]")
            if self.duplicate_queries:
                print(f"  * Duplicate query IDs (first 5): {self.duplicate_queries[:5]}")
            if self.missing_queries:
                print(f"  * Missing query IDs (first 5): {self.missing_queries[:5]}")
            if self.extra_queries:
                print(f"  * Extra query IDs (first 5): {self.extra_queries[:5]}")
            if self.unknown_docs:
                print(f"  * Unknown document IDs (first 5): {self.unknown_docs[:5]}")
            if self.unknown_chunks:
                print(f"  * Unknown chunk IDs (first 5): {self.unknown_chunks[:5]}")
            if self.internal_duplicates:
                print(f"  * Internal duplicates (first 5): {self.internal_duplicates[:5]}")
            if self.parent_violations:
                print(f"  * Parent consistency violations (first 5): {self.parent_violations[:5]}")
        print("=" * 60)


def validate_submission_records(
    submission_records: List[dict],
    doc_map: DocumentChunkMap,
    expected_query_ids: Optional[Set[str]] = None,
) -> ValidationReport:
    """Validate raw submission IDs, query coverage, and corpus/parent references.

    The evaluator separately enforces query-set integrity before computing metrics.
    """
    seen_query_ids: Set[str] = set()
    duplicate_queries: List[str] = []
    missing_queries: List[str] = []
    extra_queries: List[str] = []
    unknown_docs: Set[str] = set()
    unknown_chunks: Set[str] = set()
    internal_duplicates: List[str] = []
    parent_violations: List[str] = []

    all_known_docs = doc_map.all_doc_ids | set(doc_map.chunk_to_doc.values())
    all_known_chunks = doc_map.all_chunk_ids

    for idx, rec in enumerate(submission_records, start=1):
        qid = rec.get("id")
        if not qid:
            duplicate_queries.append(f"Line_{idx}_MISSING_ID")
            continue

        if qid in seen_query_ids:
            duplicate_queries.append(qid)
        seen_query_ids.add(qid)

        rel_docs = rec.get("relevant_docs", [])
        rel_chunks = rec.get("relevant_chunks", [])

        # Check internal duplicates in lists
        if len(rel_docs) != len(set(rel_docs)):
            internal_duplicates.append(f"Query {qid} has duplicate doc IDs: {rel_docs}")
        if len(rel_chunks) != len(set(rel_chunks)):
            internal_duplicates.append(f"Query {qid} has duplicate chunk IDs: {rel_chunks}")

        rel_docs_set = set(rel_docs)

        # Check unknown docs
        for d in rel_docs:
            if all_known_docs and d not in all_known_docs:
                unknown_docs.add(d)

        # Check unknown chunks and parent consistency
        for c in rel_chunks:
            if all_known_chunks and c not in all_known_chunks:
                unknown_chunks.add(c)
                continue

            parent_doc = doc_map.get_parent_doc(c)
            if parent_doc and parent_doc not in rel_docs_set:
                parent_violations.append(
                    f"Query {qid}: chunk '{c}' belongs to parent doc '{parent_doc}' which is missing in relevant_docs"
                )

    if expected_query_ids is not None:
        missing_queries = sorted(expected_query_ids - seen_query_ids)
        extra_queries = sorted(seen_query_ids - expected_query_ids)

    expected_count = len(expected_query_ids) if expected_query_ids is not None else len(seen_query_ids)
    total_count = len(seen_query_ids)

    is_valid = (
        len(duplicate_queries) == 0
        and len(missing_queries) == 0
        and len(extra_queries) == 0
        and len(unknown_docs) == 0
        and len(unknown_chunks) == 0
        and len(internal_duplicates) == 0
        and len(parent_violations) == 0
        and total_count == expected_count
        and total_count > 0
    )

    return ValidationReport(
        is_valid=is_valid,
        total_queries=total_count,
        expected_queries=expected_count,
        duplicate_queries=duplicate_queries,
        missing_queries=missing_queries,
        extra_queries=extra_queries,
        unknown_docs=sorted(unknown_docs),
        unknown_chunks=sorted(unknown_chunks),
        internal_duplicates=internal_duplicates,
        parent_violations=parent_violations,
    )


def validate_submission_file(
    submission_path: Union[str, Path],
    doc_map: DocumentChunkMap,
    expected_queries_path: Optional[Union[str, Path]] = None,
) -> ValidationReport:
    """Loads a JSONL submission file and executes full validation checks."""
    sub_path = Path(submission_path)
    if not sub_path.is_file():
        raise FileNotFoundError(f"Submission file does not exist: {sub_path}")

    records = []
    with sub_path.open("r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            if line_str:
                records.append(json.loads(line_str))

    expected_query_ids: Optional[Set[str]] = None
    if expected_queries_path is not None:
        q_path = Path(expected_queries_path)
        if not q_path.is_file():
            raise FileNotFoundError(f"Expected queries file does not exist: {q_path}")
        queries = DataLoader.load_queries(q_path)
        errors = validate_queries_integrity(queries)
        if errors:
            raise ValueError("Invalid expected queries: " + "; ".join(errors))
        expected_query_ids = {q.id for q in queries}

    return validate_submission_records(records, doc_map, expected_query_ids)


def main():
    parser = argparse.ArgumentParser(description="Validate competition submission JSONL file.")
    parser.add_argument("submission", type=str, help="Path to submission.jsonl")
    parser.add_argument("--corpus-chunks", type=str, required=False, help="Path to corpus chunks.jsonl or doc_map.parquet")
    parser.add_argument("--test-queries", type=str, required=False, help="Path to evaluation queries.jsonl")

    args = parser.parse_args()

    doc_map = DocumentChunkMap()
    if args.corpus_chunks:
        chunks_p = Path(args.corpus_chunks)
        if chunks_p.suffix == ".parquet":
            doc_map = DocumentChunkMap.load_parquet(chunks_p)
        else:
            chunks = DataLoader.load_chunks(chunks_p)
            doc_map = DocumentChunkMap.from_chunks(chunks)
        if not doc_map.all_chunk_ids:
            raise ValueError(f"Corpus map contains no chunks: {chunks_p}")

    report = validate_submission_file(
        submission_path=args.submission,
        doc_map=doc_map,
        expected_queries_path=args.test_queries,
    )
    report.print_summary()

    if not report.is_valid:
        sys.exit(1)


if __name__ == "__main__":
    main()

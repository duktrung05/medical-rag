"""Convert the public MedQuAD XML collection into retrieval JSONL splits.

The conversion is document-grouped: every question/answer pair from one source
XML file stays in the same split.  Questions become retrieval queries, answers
become chunks, and structural query-to-answer links become binary qrels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


MEDQUAD_LICENSE = "CC BY 4.0"
MEDQUAD_URL = "https://github.com/abachaa/MedQuAD"
EXCLUDED_DIRECTORIES = {
    ".git",
    "10_MPlus_ADAM_QA",
    "11_MPlusDrugs_QA",
    "12_MPlusHerbsSupplements_QA",
}


def _clean_text(value: str | None) -> str:
    return " ".join((value or "").split())


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return normalized or "unknown"


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _find_xml_files(input_dir: Path) -> list[Path]:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"MedQuAD directory does not exist: {input_dir}")
    files = [
        path
        for path in input_dir.rglob("*.xml")
        if not EXCLUDED_DIRECTORIES.intersection(path.relative_to(input_dir).parts)
    ]
    if not files:
        raise FileNotFoundError(f"No eligible MedQuAD XML files found below: {input_dir}")
    return sorted(files)


def _document_id(input_dir: Path, xml_path: Path, root: ET.Element) -> str:
    relative = xml_path.relative_to(input_dir).as_posix()
    source_id = root.attrib.get("id", xml_path.stem)
    collection = xml_path.relative_to(input_dir).parts[0]
    digest = hashlib.sha1(relative.encode("utf-8")).hexdigest()[:8]
    return f"MQ_{_safe_id(collection)}_{_safe_id(source_id)}_{digest}"


def load_medquad_documents(input_dir: Path) -> list[dict[str, Any]]:
    """Load XML documents that contain at least one non-empty QA pair."""
    documents: list[dict[str, Any]] = []
    seen_query_ids: set[str] = set()

    for xml_path in _find_xml_files(input_dir):
        try:
            root = ET.parse(xml_path).getroot()
        except ET.ParseError as exc:
            raise ValueError(f"Invalid MedQuAD XML file {xml_path}: {exc}") from exc

        doc_id = _document_id(input_dir, xml_path, root)
        pairs: list[dict[str, str]] = []
        for index, pair in enumerate(root.findall("./QAPairs/QAPair")):
            question_node = pair.find("Question")
            answer_node = pair.find("Answer")
            question = _clean_text(question_node.text if question_node is not None else None)
            answer = _clean_text("".join(answer_node.itertext()) if answer_node is not None else None)
            if not question or not answer:
                continue

            external_qid = (
                question_node.attrib.get("qid") if question_node is not None else None
            ) or pair.attrib.get("pid") or str(index)
            query_id = f"{doc_id}_Q{_safe_id(external_qid)}"
            if query_id in seen_query_ids:
                query_id = f"{query_id}_{index:03d}"
            seen_query_ids.add(query_id)
            pairs.append(
                {
                    "query_id": query_id,
                    "question": question,
                    "answer": answer,
                    "question_type": (
                        question_node.attrib.get("qtype", "unknown")
                        if question_node is not None
                        else "unknown"
                    ),
                }
            )

        if not pairs:
            continue

        category_node = root.find("./FocusAnnotations/Category")
        documents.append(
            {
                "doc_id": doc_id,
                "source_document_id": root.attrib.get("id", xml_path.stem),
                "source": root.attrib.get("source", "unknown"),
                "url": root.attrib.get("url"),
                "focus": _clean_text(root.findtext("Focus")),
                "category": _clean_text(category_node.text if category_node is not None else None),
                "relative_path": xml_path.relative_to(input_dir).as_posix(),
                "pairs": pairs,
            }
        )
    return documents


def _split_documents(
    documents: list[dict[str, Any]], train_ratio: float, dev_ratio: float, seed: int
) -> dict[str, list[dict[str, Any]]]:
    if train_ratio <= 0 or dev_ratio <= 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, dev > 0, train + dev < 1")
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    train_end = round(len(shuffled) * train_ratio)
    dev_end = train_end + round(len(shuffled) * dev_ratio)
    return {
        "train": shuffled[:train_end],
        "dev": shuffled[train_end:dev_end],
        "test": shuffled[dev_end:],
    }


def convert_medquad(
    input_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    documents = load_medquad_documents(input_dir)
    if len(documents) < 3:
        raise ValueError("MedQuAD conversion requires at least three non-empty documents")
    splits = _split_documents(documents, train_ratio, dev_ratio, seed)
    summary: dict[str, dict[str, int]] = {}

    for split_name, split_documents in splits.items():
        chunks: list[dict[str, Any]] = []
        queries: list[dict[str, str]] = []
        ground_truth: list[dict[str, Any]] = []
        document_metadata: list[dict[str, Any]] = []

        for document in sorted(split_documents, key=lambda item: item["doc_id"]):
            chunk_ids: list[str] = []
            for chunk_index, pair in enumerate(document["pairs"]):
                chunk_id = f"{document['doc_id']}_C{chunk_index:03d}"
                chunk_ids.append(chunk_id)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": document["doc_id"],
                        "chunk_index": chunk_index,
                        "language": "en",
                        "text": pair["answer"],
                        "title": document["focus"] or None,
                        "context": pair["question_type"],
                    }
                )
                queries.append({"id": pair["query_id"], "query": pair["question"]})
                ground_truth.append(
                    {
                        "id": pair["query_id"],
                        "relevant_docs": [document["doc_id"]],
                        "relevant_chunks": [chunk_id],
                    }
                )

            document_metadata.append(
                {
                    key: document[key]
                    for key in (
                        "doc_id",
                        "source_document_id",
                        "source",
                        "url",
                        "focus",
                        "category",
                        "relative_path",
                    )
                }
                | {"chunk_ids": chunk_ids}
            )

        split_dir = output_dir / split_name
        summary[split_name] = {
            "documents": _write_jsonl(split_dir / "documents.jsonl", document_metadata),
            "chunks": _write_jsonl(split_dir / "chunks.jsonl", chunks),
            "queries": _write_jsonl(split_dir / "queries.jsonl", queries),
            "ground_truth": _write_jsonl(split_dir / "ground_truth.jsonl", ground_truth),
        }

    source_counts = Counter(document["source"] for document in documents)
    manifest = {
        "dataset": "MedQuAD",
        "dataset_role": "medical_retrieval_benchmark",
        "source_url": MEDQUAD_URL,
        "license": MEDQUAD_LICENSE,
        "query_language": "en",
        "corpus_languages": ["en"],
        "label_origin": "structural_question_answer_pair",
        "medical_validation": (
            "Source-derived medical QA; not independently clinically validated by this project"
        ),
        "excluded_collections": sorted(EXCLUDED_DIRECTORIES - {".git"}),
        "split_policy": "source XML documents are disjoint across train/dev/test",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "dev": dev_ratio,
            "test": 1.0 - train_ratio - dev_ratio,
        },
        "source_document_counts": dict(sorted(source_counts.items())),
        "counts": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MedQuAD XML to retrieval JSONL")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("data/medquad"))
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    summary = convert_medquad(
        args.input_dir, args.output_dir, args.train_ratio, args.dev_ratio, args.seed
    )
    print(f"Converted MedQuAD into: {args.output_dir.resolve()}")
    for split_name, counts in summary.items():
        print(f"  {split_name}: {counts}")


if __name__ == "__main__":
    main()

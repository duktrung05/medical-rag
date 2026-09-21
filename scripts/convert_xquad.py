"""Convert XQuAD Vietnamese/English/Chinese Parquet files to retrieval data.

The converter uses the shared XQuAD question IDs to align translated contexts.
It creates disjoint train/dev/test splits by paragraph group, so questions that
share a context cannot leak across splits.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


REQUIRED_COLUMNS = {"id", "context", "question", "answers"}
LANGUAGES = ("vi", "en", "zh")


def _resolve_parquet_files(path: Path) -> list[Path]:
    """Return the Parquet file or all Parquet shards below a directory."""
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError(f"Expected a .parquet file, got: {path}")
        return [path]

    if path.is_dir():
        files = sorted(path.rglob("*.parquet"))
        if files:
            return files
        raise FileNotFoundError(f"No .parquet files found below: {path}")

    raise FileNotFoundError(f"Input path does not exist: {path}")


def load_xquad(path: Path, language: str) -> pd.DataFrame:
    """Load one XQuAD language and validate the fields used by the converter."""
    files = _resolve_parquet_files(path)
    frames = [pd.read_parquet(file) for file in files]
    frame = pd.concat(frames, ignore_index=True)

    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(
            f"XQuAD {language} is missing columns {sorted(missing)}; "
            f"found {sorted(frame.columns)}"
        )

    frame = frame.loc[:, ["id", "context", "question", "answers"]].copy()
    for column in ("id", "context", "question"):
        if frame[column].isna().any():
            raise ValueError(f"XQuAD {language} contains null values in '{column}'")
        frame[column] = frame[column].astype(str).str.strip()
        if frame[column].eq("").any():
            raise ValueError(f"XQuAD {language} contains empty values in '{column}'")

    duplicates = frame[frame["id"].duplicated()]["id"].tolist()
    if duplicates:
        raise ValueError(
            f"XQuAD {language} contains duplicate question IDs, for example: "
            f"{duplicates[:3]}"
        )

    return frame


def _json_safe(value: Any) -> Any:
    """Convert pandas/numpy containers in the answers column to JSON values."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if pd.isna(value):
        return None
    return value


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def _validate_language_alignment(frames: dict[str, pd.DataFrame]) -> None:
    id_sets = {language: set(frame["id"]) for language, frame in frames.items()}
    reference = id_sets["vi"]
    for language in ("en", "zh"):
        missing = sorted(reference.difference(id_sets[language]))
        extra = sorted(id_sets[language].difference(reference))
        if missing or extra:
            raise ValueError(
                f"Question IDs do not align between vi and {language}. "
                f"Missing examples: {missing[:3]}; extra examples: {extra[:3]}"
            )


def _build_groups(
    frames: dict[str, pd.DataFrame],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Build aligned paragraph groups and map each question ID to its group."""
    lookup = {
        language: frame.set_index("id", drop=False)
        for language, frame in frames.items()
    }

    vi_context_to_ids: dict[str, list[str]] = defaultdict(list)
    context_order: list[str] = []
    for row in frames["vi"].itertuples(index=False):
        if row.context not in vi_context_to_ids:
            context_order.append(row.context)
        vi_context_to_ids[row.context].append(row.id)

    groups: list[dict[str, Any]] = []
    question_to_group: dict[str, str] = {}
    for number, vi_context in enumerate(context_order, start=1):
        question_ids = vi_context_to_ids[vi_context]
        contexts: dict[str, str] = {"vi": vi_context}

        for language in ("en", "zh"):
            translated_contexts = {
                str(lookup[language].loc[question_id, "context"])
                for question_id in question_ids
            }
            if len(translated_contexts) != 1:
                raise ValueError(
                    f"Vietnamese paragraph {number} maps to "
                    f"{len(translated_contexts)} {language} contexts. "
                    "The language files may not be parallel XQuAD versions."
                )
            contexts[language] = translated_contexts.pop()

        group_id = f"XQ_P{number:04d}"
        for question_id in question_ids:
            question_to_group[question_id] = group_id
        groups.append(
            {
                "group_id": group_id,
                "contexts": contexts,
                "question_ids": question_ids,
            }
        )

    return groups, question_to_group


def _split_group_ids(
    groups: list[dict[str, Any]],
    train_ratio: float,
    dev_ratio: float,
    seed: int,
) -> dict[str, set[str]]:
    if train_ratio <= 0 or dev_ratio <= 0 or train_ratio + dev_ratio >= 1:
        raise ValueError("Ratios must satisfy train > 0, dev > 0, train + dev < 1")

    group_ids = [group["group_id"] for group in groups]
    random.Random(seed).shuffle(group_ids)
    total = len(group_ids)
    train_end = round(total * train_ratio)
    dev_end = train_end + round(total * dev_ratio)

    return {
        "train": set(group_ids[:train_end]),
        "dev": set(group_ids[train_end:dev_end]),
        "test": set(group_ids[dev_end:]),
    }


def convert_xquad(
    vi_path: Path,
    en_path: Path,
    zh_path: Path,
    output_dir: Path,
    train_ratio: float = 0.70,
    dev_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, dict[str, int]]:
    """Convert three parallel XQuAD files into retrieval JSONL splits."""
    frames = {
        "vi": load_xquad(vi_path, "vi"),
        "en": load_xquad(en_path, "en"),
        "zh": load_xquad(zh_path, "zh"),
    }
    _validate_language_alignment(frames)
    groups, question_to_group = _build_groups(frames)
    split_ids = _split_group_ids(groups, train_ratio, dev_ratio, seed)
    group_by_id = {group["group_id"]: group for group in groups}
    vi_rows = frames["vi"].set_index("id", drop=False)

    summary: dict[str, dict[str, int]] = {}
    for split_name, selected_ids in split_ids.items():
        selected_groups = [
            group_by_id[group_id]
            for group_id in sorted(selected_ids)
        ]

        chunks: list[dict[str, Any]] = []
        group_metadata: list[dict[str, Any]] = []
        for group in selected_groups:
            group_id = group["group_id"]
            document_ids: list[str] = []
            for language in LANGUAGES:
                doc_id = f"{group_id}_{language}"
                chunk_id = f"{doc_id}_000"
                document_ids.append(doc_id)
                chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "doc_id": doc_id,
                        "chunk_index": 0,
                        "language": language,
                        "text": group["contexts"][language],
                    }
                )
            group_metadata.append(
                {
                    "group_id": group_id,
                    "document_ids": document_ids,
                    "question_ids": group["question_ids"],
                }
            )

        selected_question_ids = [
            question_id
            for question_id in frames["vi"]["id"]
            if question_to_group[question_id] in selected_ids
        ]
        queries: list[dict[str, Any]] = []
        ground_truth: list[dict[str, Any]] = []
        answers: list[dict[str, Any]] = []
        for question_id in selected_question_ids:
            group_id = question_to_group[question_id]
            query_id = f"XQ_{question_id}"
            doc_ids = [f"{group_id}_{language}" for language in LANGUAGES]
            chunk_ids = [f"{doc_id}_000" for doc_id in doc_ids]
            row = vi_rows.loc[question_id]
            queries.append({"id": query_id, "query": row["question"]})
            ground_truth.append(
                {
                    "id": query_id,
                    "relevant_docs": doc_ids,
                    "relevant_chunks": chunk_ids,
                }
            )
            answers.append(
                {
                    "id": query_id,
                    "source_question_id": question_id,
                    "answers": _json_safe(row["answers"]),
                }
            )

        split_dir = output_dir / split_name
        counts = {
            "groups": _write_jsonl(split_dir / "groups.jsonl", group_metadata),
            "chunks": _write_jsonl(split_dir / "chunks.jsonl", chunks),
            "queries": _write_jsonl(split_dir / "queries.jsonl", queries),
            "ground_truth": _write_jsonl(
                split_dir / "ground_truth.jsonl", ground_truth
            ),
            "answers": _write_jsonl(split_dir / "answers.jsonl", answers),
        }
        summary[split_name] = counts

    manifest = {
        "source": "google/xquad",
        "languages": list(LANGUAGES),
        "query_language": "vi",
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "dev": dev_ratio,
            "test": 1.0 - train_ratio - dev_ratio,
        },
        "counts": summary,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert parallel XQuAD vi/en/zh Parquet files to retrieval JSONL. "
            "Each input can be a Parquet file or a directory containing shards."
        )
    )
    parser.add_argument("--vi", type=Path, required=True, help="Vietnamese file/directory")
    parser.add_argument("--en", type=Path, required=True, help="English file/directory")
    parser.add_argument("--zh", type=Path, required=True, help="Chinese file/directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/xquad"),
        help="Output directory (default: data/xquad)",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--dev-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    summary = convert_xquad(
        vi_path=args.vi,
        en_path=args.en,
        zh_path=args.zh,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
    )

    print(f"Converted XQuAD data into: {args.output_dir.resolve()}")
    for split_name in ("train", "dev", "test"):
        counts = summary[split_name]
        print(
            f"  {split_name}: {counts['groups']} groups, "
            f"{counts['chunks']} chunks, {counts['queries']} queries"
        )


if __name__ == "__main__":
    main()

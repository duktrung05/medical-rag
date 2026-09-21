import json
from pathlib import Path

import pandas as pd

from scripts.convert_xquad import convert_xquad


def _make_language_file(tmp_path: Path, language: str) -> Path:
    rows = []
    for group_number in range(1, 7):
        for question_number in range(1, 3):
            question_id = f"g{group_number}q{question_number}"
            rows.append(
                {
                    "id": question_id,
                    "context": f"context-{language}-{group_number}",
                    "question": f"question-{language}-{question_id}",
                    "answers": {
                        "text": [f"answer-{language}"],
                        "answer_start": [0],
                    },
                }
            )
    path = tmp_path / f"{language}.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_convert_xquad_creates_aligned_leak_free_splits(tmp_path: Path) -> None:
    inputs = {
        language: _make_language_file(tmp_path, language)
        for language in ("vi", "en", "zh")
    }
    output_dir = tmp_path / "output"

    summary = convert_xquad(
        vi_path=inputs["vi"],
        en_path=inputs["en"],
        zh_path=inputs["zh"],
        output_dir=output_dir,
        train_ratio=0.5,
        dev_ratio=0.25,
        seed=7,
    )

    assert sum(item["groups"] for item in summary.values()) == 6
    split_group_ids = []
    for split_name in ("train", "dev", "test"):
        chunks = _read_jsonl(output_dir / split_name / "chunks.jsonl")
        queries = _read_jsonl(output_dir / split_name / "queries.jsonl")
        truths = _read_jsonl(output_dir / split_name / "ground_truth.jsonl")
        groups = _read_jsonl(output_dir / split_name / "groups.jsonl")

        assert len(chunks) == len(groups) * 3
        assert len(queries) == len(groups) * 2
        assert len(truths) == len(queries)
        assert {chunk["language"] for chunk in chunks} == {"vi", "en", "zh"}
        assert all(len(item["relevant_docs"]) == 3 for item in truths)
        split_group_ids.append({group["group_id"] for group in groups})

    assert split_group_ids[0].isdisjoint(split_group_ids[1])
    assert split_group_ids[0].isdisjoint(split_group_ids[2])
    assert split_group_ids[1].isdisjoint(split_group_ids[2])

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["query_language"] == "vi"
    assert manifest["languages"] == ["vi", "en", "zh"]

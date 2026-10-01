import json
from pathlib import Path

from scripts.convert_medquad import convert_medquad


def _write_document(root: Path, number: int, *, with_answer: bool = True) -> None:
    collection = root / "1_CancerGov_QA"
    collection.mkdir(parents=True, exist_ok=True)
    answer = f"Answer for disease {number}" if with_answer else ""
    (collection / f"doc-{number}.xml").write_text(
        f'''<?xml version="1.0" encoding="UTF-8"?>
<Document id="doc-{number}" source="CancerGov" url="https://example.test/{number}">
  <Focus>Disease {number}</Focus>
  <FocusAnnotations><Category>Disease</Category></FocusAnnotations>
  <QAPairs>
    <QAPair pid="1">
      <Question qid="q-{number}" qtype="treatment">How is disease {number} treated?</Question>
      <Answer>{answer}</Answer>
    </QAPair>
  </QAPairs>
</Document>''',
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_convert_medquad_creates_document_disjoint_splits(tmp_path: Path) -> None:
    source = tmp_path / "source"
    for number in range(8):
        _write_document(source, number, with_answer=number != 7)

    output = tmp_path / "output"
    summary = convert_medquad(source, output, train_ratio=0.5, dev_ratio=0.25, seed=7)

    assert sum(counts["documents"] for counts in summary.values()) == 7
    split_doc_ids = []
    for split in ("train", "dev", "test"):
        documents = _read_jsonl(output / split / "documents.jsonl")
        chunks = _read_jsonl(output / split / "chunks.jsonl")
        queries = _read_jsonl(output / split / "queries.jsonl")
        truths = _read_jsonl(output / split / "ground_truth.jsonl")
        assert len(documents) == len(chunks) == len(queries) == len(truths)
        assert all(chunk["language"] == "en" for chunk in chunks)
        assert all(len(item["relevant_chunks"]) == 1 for item in truths)
        split_doc_ids.append({item["doc_id"] for item in documents})

    assert split_doc_ids[0].isdisjoint(split_doc_ids[1])
    assert split_doc_ids[0].isdisjoint(split_doc_ids[2])
    assert split_doc_ids[1].isdisjoint(split_doc_ids[2])

    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset"] == "MedQuAD"
    assert manifest["license"] == "CC BY 4.0"
    assert "10_MPlus_ADAM_QA" in manifest["excluded_collections"]

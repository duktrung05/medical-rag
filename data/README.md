# Data Directory

- `medquad/`: primary public medical retrieval benchmark.
- `processed/`: optimized local representations and document/chunk maps.
- `dev/`: small project-owned fixtures used by unit tests.
- `raw/`: reserved for future competition data; do not mix it with benchmark labels.

## MedQuAD benchmark

The source collection is MedQuAD, released under CC BY 4.0. Download it and
convert the XML collection with:

```powershell
git clone --depth 1 https://github.com/abachaa/MedQuAD.git downloads/medquad-source
python -m scripts.convert_medquad `
  --input-dir downloads/medquad-source `
  --output-dir data/medquad
```

The converter excludes the three collections whose answer text was removed by
MedQuAD for MedlinePlus copyright compliance. It splits by source XML document,
so questions about the same source document cannot leak across train, dev and
test.

Each split contains:

- `documents.jsonl`: source URL, publisher, focus and original document ID.
- `chunks.jsonl`: non-empty medical answers used as retrieval passages.
- `queries.jsonl`: medical questions.
- `ground_truth.jsonl`: structural question-to-answer document/chunk relevance.

`manifest.json` records license, exclusions, split policy, seed and counts. These
labels come from the source QA structure; they have not been independently
clinically validated by this project.

# Data Directory

- `raw/`: Unmodified input files provided by BTC (`queries.jsonl`, `chunks.jsonl`).
- `processed/`: Optimized binary representations (`queries.parquet`, `chunks.parquet`, `doc_map.parquet`).
- `dev/`: Development splits and synthetic fixtures for threshold calibration and offline benchmarking.

## XQuAD smoke-test conversion

Download `validation-00000-of-00001.parquet` from the `xquad.vi`,
`xquad.en`, and `xquad.zh` folders into separate local directories. Convert them
with:

```powershell
python scripts/convert_xquad.py `
  --vi data/raw/xquad/xquad.vi `
  --en data/raw/xquad/xquad.en `
  --zh data/raw/xquad/xquad.zh `
  --output-dir data/xquad
```

The converter aligns translations using the shared question IDs, deduplicates
contexts into one chunk per language, and splits by aligned paragraph group.
Each of `data/xquad/train`, `data/xquad/dev`, and `data/xquad/test` contains:

- `chunks.jsonl`: retrieval corpus in Vietnamese, English, and Chinese.
- `queries.jsonl`: Vietnamese queries.
- `ground_truth.jsonl`: the three aligned relevant documents/chunks per query.
- `answers.jsonl`: QA answers kept separate from retrieval input.
- `groups.jsonl`: paragraph alignment metadata used to audit split leakage.

`manifest.json` records the seed, split ratios, and output counts.

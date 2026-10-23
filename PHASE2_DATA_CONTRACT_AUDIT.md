# Phase 2 Data Contract Audit

Audit date: 2026-10-15 (Asia/Ho_Chi_Minh)

## Gate result

**PASS for the 1,000-document Phase 2 pilot.** The earlier blocker was removed by
the completed Phase 1 release and the project-local tokenizer dependency. This
audit does not authorize a full Phase 2 build.

## Accepted input

- Candidate: `outputs/vibiomir_phase1_full_20261014_v1`
- Candidate status/scope: `complete` / `full`
- Candidate documents: `1,075,599`, all unique
- Release: `release/phase1_release_manifest.json`
- PASS allowlist: `1,073,531` unique `doc_id`
- Held: `887 QUARANTINED` and `1,181 REVIEW_REQUIRED`
- PASS/held intersection: `0`
- Clean document format: `vibiomir-phase1-clean-document-v3`
- Corpus revision: `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`
- Normalization: `unicode-nfc-whitespace-v1`
- Language detection: `script-stopwords-metadata-v2`

The dry-run recomputed and matched checksums for all `922` clean shards, `922`
PASS shards and `922` held shards. It also verified the release/candidate/source
manifest chain. Phase 2 reads only `normalized_text` from clean shards for IDs in
the exact PASS allowlist. No raw HTML, crawler output or legacy chunk is an input.

During stratified selection, all `1,073,531` PASS rows were checked against the
checksum-locked Phase 0 `doc_id` to `official_url` mapping. Mapping errors: `0`.

## Source schema used by Phase 2

The adapter reads these immutable Phase 1 fields:

```text
doc_id: int64
official_url: string
source_kind: string
title: string
normalized_text: string
source_text_sha256: string
normalized_text_sha256: string
detected_language: string
```

Passage offsets always refer to `normalized_text`. The invariant is:

```python
normalized_text[char_start:char_end] == passage_text
```

## BGE-M3 tokenizer

- Name: `BAAI/bge-m3`
- Revision: `5617a9f61b028005a4858fdac845db406aefb181`
- Local `tokenizer.json` SHA-256:
  `21106b6d7dab2952c1d496fb21d5dc9db75c28ed361a05f5020bbba27810dd08`
- Runtime: project `.venv-phase1`, `tokenizers==0.23.3`
- Loading mode: direct local `Tokenizer.from_file`; no model download

Token counts include the tokenizer's configured special tokens and are recomputed
by the validator for every pilot passage.

## Legacy component decision

`src/chunking/paragraph_chunker.py` supplied useful prior offset concepts but is
not the Phase 2 engine. It lacks release gating, hierarchy, versioned passage IDs,
multi-strategy output and sharded resume.

`scripts/workflows/vibiomir/chunk_vibiomir_probe.py` remains a legacy probe and is
not called. It targets a probe parquet and cannot enforce the accepted release.

`r2ai/extract.py` is prohibited because it reads crawler/raw content and performs
legacy extraction/chunking.

The accepted implementation is isolated under `src/medrag/data/phase2/` and
`scripts/workflows/vibiomir/build_phase2_passages.py`.

## Immutability

Phase 0 and Phase 1 were opened read-only by this workflow. All new artifacts are
under `outputs/vibiomir_phase2_20261015_pilot_v1`; no crawl, extraction,
normalization, deduplication, NER, ICD-10, KG or index build was invoked.

## Remaining approval gate

The structural pilot passes automated validation, but manual review is explicitly
unconfirmed and no retrieval relevance benchmark was run. Full processing must
wait for review of the exported passage sample and user approval of a strategy.

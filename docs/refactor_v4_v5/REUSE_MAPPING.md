# Reuse mapping

| Existing source | Decision | V4 outcome |
|---|---|---|
| `src/extraction/html_extractor.py` | REUSE concept | Charset declaration handling, structure-aware extraction reviewed; independent `src/medrag/data/extraction.py` written and tested. The old implementation is not imported. |
| `src/chunking/paragraph_chunker.py` | REUSE concept | Exact offset and token limit invariant carried into `src/medrag/chunking/passage_builder.py`; independent implementation. |
| `src/retrieval/fusion.py` | REUSE algorithm | Reciprocal rank fusion and rank provenance reimplemented in `src/medrag/retrieval/core.py`. |
| `r2ai/bm25index.py` | REUSE analyzer idea | Chinese bigrams and Unicode word tokens reimplemented; the old `max_chunks` and taxonomy coupling were removed. |
| `r2ai/validate.py` | REUSE schema knowledge | `src/medrag/adapters/vibiomir_submission.py` emits its document/evidence schema; strict ZIP validation still pending. |
| `r2ai/extract.py` | REUSE data layout | Read-only gzip shard path adapted in `src/medrag/data/loader.py`; extraction/chunking logic not copied. |

These are logic and format reuse, not active runtime imports. Existing `src/evaluation/` and `src/submission/` remain historical because their generic JSONL schema differs from the official ViBioMIR ZIP schema.

# V4 smoke test report — 2026-10-12

Commands run:

```bash
python3 -m unittest discover -s tests/v4 -v
python3 -m src.cli check-config
python3 -m src.cli pilot
```

Ten V4 tests pass. The config check resolves the pinned official corpus revision and matches its local Parquet SHA-256. The pilot reads three preserved gzip HTML documents (IDs 583, 565962, 836542), builds 28 passages, and ranks document 583 first for a Vietnamese HIV/AIDS query and document 836542 first for a Chinese menstruation query. Evidence matches exact extracted-text offsets. The pilot uses the historical local corpus version; it does not establish official-corpus coverage.

Tests cover multilingual extraction, tail evidence, offsets, duplicate text in different documents, ID/URL and corpus-version isolation, lexical ranking, negation/unit preservation, Chinese `gb18030` HTML, fake dense/reranker adapters, config registry, submission record shape, prepared-snapshot tampering, and synthetic official/historical URL disagreement. Fake adapters establish interfaces, not model quality.

The active `python3` is 3.10 and lacks `pyarrow`, FastAPI, pydantic and other declared packages; `/usr/bin/python3.11` parsed the TOML but also lacks the project dependencies. Real official Parquet streaming, HTTP runtime, BGE models, complete ZIP validation, the full pytest suite and corpus-scale indexing were not exercised. Existing large chunk/index/cache hashes were not recomputed.

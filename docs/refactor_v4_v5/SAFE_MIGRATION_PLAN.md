# Safe migration plan

1. Keep all existing ViBioMIR data and old indexes at their current paths. Use read-only inputs for V4. Bounded prepared snapshots belong under `outputs/v4/`; future persistent indexes belong in a separate V4 namespace with a manifest carrying revision and corpus hash.
2. Before reusing crawled HTML, compare `(official_id, normalized_url)` in both the pinned official and historical URL corpora. A matching numeric ID alone is insufficient. `prepare_from_official` and `load_prepared` enforce this for V4 snapshots.
3. Stream official Parquet in bounded batches when `pyarrow` is available. The old local corpus has 4,394,718 rows; the pinned official revision has 1,220,598 rows. Existing audit reports 980,598 matching ID/URL pairs and 240,000 official URLs missing locally.
4. Preserve raw text and raw SHA-256; derived metadata must carry method, confidence and provenance before it can influence scoring. Do not infer offsets for historical chunks that lack them.
5. Build a corpus-scale V4 sparse backend and dense index in a new namespace, then run contract tests and a representative multilingual pilot. Existing indexes are read only.
6. V4 is now the default CLI/API/Docker architecture on local `master`. The service fails closed when no verified official snapshot is configured. Full competition-scale operation still requires a scalable backend, scorer-format validation and official coverage. Historical commands can be replayed in a separate baseline worktree; permanent deletion still requires an approved path list.

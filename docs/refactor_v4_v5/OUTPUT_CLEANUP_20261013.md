# Output cleanup — 2026-10-13

The user requested a smaller `outputs/` while rebuilding V4/V5. This pass removed only files whose contents or disposable nature were verified:

| Deleted path | Bytes | Verification / retained source |
| --- | ---: | --- |
| `outputs/submissions/vibiomir_maxdocs80_20261008/diagnostics.jsonl` | 23,688,846 | SHA-256 `3d3f5d5c0ebf366b8d53b4e0de706a54304787b4ca3d7d33f28afb591d319c15` matched `outputs/submissions/vibiomir_crawl_rerank80_20261008/diagnostics.jsonl` before deletion and the retained file was rehashed afterward. No direct path reference outside historical inventory was found. |
| `outputs/__pycache__/*.pyc` (3 files) | 30,925 | Generated Python bytecode; the directory was removed. |

Total removed: **23,719,771 bytes**. The two submission ZIPs were retained: their archives have different hashes despite containing a `predictions.json` payload with the same hash. Other duplicate historical rankings, reports and predictions are tied to distinct replay or audit runs, so they remain at their original paths. PID and lock files were left alone because host process state could not be established from this session. The pinned official corpus, raw data, indexes, scoring feedback and unique submission artifacts were not changed.

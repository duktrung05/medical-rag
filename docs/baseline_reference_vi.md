# Baseline tham chiếu ViMed — baseline_20260930

Commit nguồn: `8a20eef4b8be44e20f644075f1c39494ea87d9f5`. Split: validation, 2210 queries, 17955 chunks.
Corpus SHA256: `b219a1f7f55df30a759753db9f01c29397e68bd8f62125a1b5347da0781ba04a`.

Baseline retrieval được đóng băng từ rankings lịch sử của bốn stage. Metrics được replay từ rankings bằng CPU và khớp artifact đã lưu.

| Stage | Recall@1 | @3 | @5 | @10 | @20 | @50 | @100 | MRR@100 | p95 ms | VRAM peak allocated | Zero hit @10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bm25 | 0.4905 | 0.6452 | 0.6919 | 0.7525 | 0.8014 | 0.8498 | 0.8837 | 0.5839 | 137.7 | 0 | 547 |
| dense | 0.5566 | 0.7167 | 0.7661 | 0.8163 | 0.8534 | 0.8891 | 0.9222 | 0.6504 | 71.9 | 1146062848 | 406 |
| hybrid | 0.5480 | 0.7276 | 0.7719 | 0.8226 | 0.8643 | 0.9023 | 0.9249 | 0.6498 | 192.3 | 0 | 392 |
| rerank | 0.7493 | 0.8588 | 0.8864 | 0.9063 | 0.9167 | 0.9217 | 0.9249 | 0.8091 | 1086.5 | 1238667776 | 207 |

Document Recall@K và số zero-hit ở mọi cutoff nằm trong `outputs/baseline_reference/baseline_20260930/baseline_metrics.json`.

Replay metrics và kiểm tra checksum snapshot bằng CPU:

```bash
python -m scripts.replay_baseline --baseline baseline_20260930
```

Năm cấu hình gốc được sao lưu nguyên văn trong `configs/archive/baseline_20260930/`. Rankings, metrics, manifest và run identity được sao lưu riêng theo stage.

Giới hạn provenance: tokenizer revision chưa được ghi; hash `run_identity` của hybrid/rerank khác hash config hiện tại và hash trong manifest. Các giá trị lịch sử được lưu nguyên trạng. Hybrid/rerank latency là tổng stage đã cache, không phải latency API trực tiếp.

Replay CPU kiểm chứng metrics ranking. Tái chạy GPU chưa được thực hiện; test baseline hiện bị chặn khi import vì môi trường Python 3.14 thiếu `torch`.

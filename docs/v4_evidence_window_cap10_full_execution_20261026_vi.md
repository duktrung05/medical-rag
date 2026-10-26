# Full cap-10 candidate — execution result

Ngày hoàn tất: 2026-10-26. Trạng thái: **BTC_PROMOTED_CHAMPION**.

BTC score do người dùng cung cấp: FINAL `0,2324`, DOCS_F2 `0,2788`,
CHUNKS_F2 `0,1860`. So với S2, FINAL tăng `0,0006`, DOCS_F2 không đổi và
CHUNKS_F2 tăng `0,0012`. Receipt được lưu cạnh artifact với provenance rõ ràng.

## Output

- Queries: 1.200/1.200.
- Documents: 140.095, exact list/order parity với S2 từng query.
- Chunks: 72.000.
- Changed queries: 1.181.
- Chunk-parent swaps: 11.445, trung bình 9,54/query.
- ZIP size: 87.350.407 bytes, dưới gate 100 MB.
- ZIP SHA256: `e422b9076bf59f9a93d4819776026e789e7cf3f407b3befee8738fc5abb270f8`.
- Payload SHA256: `887d62ded7f133757a08bfbb7a4836a9f88e1133e85052b9250410137a54c666`.

ZIP đã PASS single-entry, CRC, 1.200-query coverage, document/chunk caps,
parent membership, duplicate checks và payload-hash equality.

## Inference

Full package có 418.751 pairs. 60 pilot queries/22.110 pairs tái sử dụng exact
locked scores; 1.140 queries/396.641 pairs được chia deterministic cho hai A6000:

| Worker | GPU | Queries | Pairs | Model seconds | Throughput |
|---:|---|---:|---:|---:|---:|
| 0 | physical GPU 2 | 570 | 200.173 | 462,04 | 433,23 pairs/s |
| 1 | physical GPU 3 | 570 | 196.468 | 457,74 | 429,21 pairs/s |

Cả hai worker đạt exact-batch parity 128/128 với max absolute difference 0.0.
Score coverage sau merge là 1.200 queries/418.751 pairs, không overlap/missing.

## Offline signals

- Manual evidence: 3 gains, 0 losses. Ba gains đều thuộc development đã dùng
  chọn cap (`5:313055`, `398:519606`, `747:51610`), nên không phải independent
  confirmation.
- Historical anchor-preservation proxy: 110 gains, 42 losses, net +68. Đây
  không phải official evidence recall hoặc relevance judgment.
- Validation/held-out screens trước full run đã PASS, nhưng không có manual gain
  mới trên hai stage đó.

Treatment thay gần 16% của 72.000 chunk slots. Net historical proxy dương nhưng
42 losses cho thấy candidate có trade-off thật. BTC xác nhận hướng thay allocation
có lợi nhưng mức tăng nhỏ: toàn bộ FINAL gain đến từ CHUNKS_F2, đúng với việc
documents giữ nguyên.

## Artifacts

- Candidate ZIP: `outputs/v4/evidence_window_cap10_full_20261025/export/submission.zip`
- Predictions: `outputs/v4/evidence_window_cap10_full_20261025/export/predictions.json`
- Run manifest: `outputs/v4/evidence_window_cap10_full_20261025/run_manifest.json`
- User-reported score receipt: `scoring_receipt_user_20261026.json`
- Score manifest: `outputs/v4/evidence_window_cap10_full_20261025/score_manifest.json`
- Pair lock/manifest: `lock.json`, `pair_manifest.json`
- Implementation: `scripts/v4/run_evidence_window_cap10_full.py`

Artifact này là champion mới theo score người dùng cung cấp; chưa ghi đè hoặc
xóa S2 artifact.

# R7 parent-floor score gate — kết quả thực thi 25/10/2026

## Quyết định hiện tại

Trạng thái: **NO_GO AT DOWNSTREAM CANARY**. Exact-batch runtime parity PASS và
385 canary pairs đã được score trên GPU 2. Hai manual parents được thu hồi vào
final documents nhưng không vào 60 chunk slots; control 53 mất một historical
positive document. Theo gate đã khóa, phần còn lại của 30-query cohort không
được score và không có submission mới. S2 vẫn là champion; ZIP SHA256 vẫn là
`081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.

## Package đã khóa

Namespace quyết định: `outputs/v4/r7_parent_floor_score_20261025_v2/`.
Namespace không hậu tố được giữ làm audit của parity gate v1 bị sai batch.

- 30 development queries, dùng lại raw traces đã khóa; không chạy lại retrieval.
- 3.844 added parents và đúng 3.844 scorable pairs; không có parent unscorable.
- 1.782 seeds lấy trực tiếp từ supplemented passage item.
- 2.062 document-only parents dùng passage-BM25 parent top-1 fallback.
- Canary `610/1159/53`: đúng 385 pairs.
- 128 old-score runtime anchors trong bốn historical batches nguyên vẹn, mỗi
  batch 32 pairs; batch được chọn gần -4, -2, chunk cut 60 và S2 document cut.
- Seed policy không đọc labels. Hai target passages chỉ được dùng để đánh giá
  sau replay, không được dùng để chọn input.

Hai target manual đã hiện diện đúng trong package:

| Query/parent | Passage | Added order |
|---|---|---:|
| 610/305379 | `p2_19752c375988910b35275999d9d3b241ae17fbf72fa6b2b2ba9ddfed841629eb` | 6 |
| 1159/514656 | `p2_51dca637b10ad5218dc2a5ab6051f86fd62cbfff33a55897c5f1ba0fe8585bde` | 22 |

Package size khoảng 11 MiB và chỉ chứa exact query/payload inputs, hashes,
provenance và runtime contract; GPU scorer không cần full corpus hoặc dense
index. `pair_manifest.json` SHA256 là
`79b8477ac3013b378db4cbb3b1e504fd032502d5c72571664838b808b681fb7f`.

## Runtime gate

Preflight và scoring chạy bằng `.venv-p5-gpu-a6000` trên physical GPU 2,
`GPU-3eb95da2-57af-94b3-2ae9-8e5d46d9c079`, chính UUID đã tạo S2:

- Torch `2.6.0+cu124`, CUDA `12.4`, model revision và pinned package versions
  khớp S2.
- Parity v1 chấm anchors rời khỏi batch gốc: FAIL, max difference `0.015625`.
  Exact historical batch diagnostic sau đó đạt 32/32 scores giống tuyệt đối,
  chứng minh sai khác do batch padding chứ không phải GPU/runtime.
- Package v2 giữ bốn batch gốc: 128/128 anchors exact, max difference `0.0`,
  mọi threshold decision giữ nguyên.
- 385 canary pairs được score trong `1.4872` giây, khoảng 258,9 pairs/giây.

Kết quả downstream:

| Query/parent | S2 | Challenger | Rerank score | Kết luận |
|---|---|---|---:|---|
| 610/305379 | Không có doc/chunk | Doc rank 104, không chunk | -0.4838867 | Docs-only gain |
| 1159/514656 | Không có doc/chunk | Doc rank 68, không chunk | 0.3015137 | Docs-only gain |
| 53/756661 | Doc rank 117 | Candidate rank 145, ngoài cap 140 | 1.8134766 | Historical doc loss |

Tổng known-positive changes trong ba queries: 5 registry-row document gains,
1 document loss, 0 evidence gains và 0 evidence losses. Các duplicate registry
rows của cùng parent được báo riêng trong con số 5; ở parent/query level có hai
manual target gains cùng một additional historical parent gain tại query 1159.
Structural validators PASS, nhưng control loss làm canary thành
`STOP_CANARY_NO_DOWNSTREAM_GAIN`. Không score 3.459 pairs còn lại.

## Kiểm tra

- Package hash/count/input validation: PASS.
- Batched passage-BM25 fallback parity với parent-local API: PASS.
- Exact-batch runtime parity: PASS, maximum absolute difference `0.0`.
- Downstream structural validators: PASS; quality/loss gate: FAIL.
- Targeted tests: `45 passed` gồm score-gate helpers, parent-floor fusion,
  submission refinement và evidence evaluation contracts.
- Python compilation và `git diff --check`: PASS.
- Official F2: **UNMEASURED**.

Worker: `scripts/v4/run_r7_parent_floor_score_gate.py`. Full cohort vẫn bị
fail-closed vì `canary_report.status != PASS_OPEN_COHORT`.

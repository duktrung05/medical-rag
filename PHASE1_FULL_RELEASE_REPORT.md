# Phase 1 Full Release Report

Ngay bao cao: 2026-10-15

## Ket luan

Phase 1 da hoan tat full build cho dung `1,075,599` document trong Phase 0
allowlist. Release chi cho phep Phase 2 doc tap `PASS`:

| Trang thai | So document |
| --- | ---: |
| PASS | 1,073,531 |
| QUARANTINED | 887 |
| REVIEW_REQUIRED | 1,181 |
| Tong | 1,075,599 |

`PASS` va tap bi giu khong giao nhau; hop cua hai tap khop dung toan bo candidate.

## Quality gate tren pilot v4

- Pilot: `outputs/vibiomir_phase1_20261014_pilot_v4`
- Quy mo pilot: `10,000` document, `922` part
- Quality rules: `phase1-quality-rules-v3`
- Mau evaluation: `500` document, gom `400` pilot va `100` quarantine control
- Ket qua mau: `399 PASS`, `1 REVIEW`, `100 LABEL_CONFIRMED`
- Normalization invariant failure: `0`
- Encoding issue trong mau: `0`
- Quarantine control duoc xac nhan: `100/100`
- Tat ca `31/31` document con bi flag sau khi sua rule da co trong flagged review.

Artifact evaluation:
`outputs/vibiomir_phase1_quality_20261014_v4/quality_summary.json`.

## Ket qua review 32 document ban dau

Ca `32` document cua pilot v3 da duoc xem xet:

| Nhom | So document | Quyet dinh |
| --- | ---: | --- |
| HTML contamination | 11 | QUARANTINED |
| Navigation/advertising-heavy | 3 | QUARANTINED |
| Missing title, noi dung tot | 15 | PASS override |
| Missing title, ngan va dominated boi navigation | 1 | QUARANTINED override |
| Mojibake chi nam trong ten tac gia citation | 1 | PASS override |
| False positive do chu viet tat tieng Viet `H.A` co dau | 1 | Sua detector, PASS |

Sau khi sua false positive, pilot v4 con `31` document bi flag. Chinh sach tren
31 flag nay cho `16 PASS` va `15 QUARANTINED`; document false positive thu 32
khong con bi flag va tu dong PASS.

## Chinh sach da chot

Policy: `configs/vibiomir/phase1_quality_policy_v1.json`

- Document khong co quality flag: `PASS`.
- Hard-quality flag nhu HTML, navigation, challenge, boilerplate, replacement
  character hoac very-short text: `QUARANTINED`.
- Flag khong thuoc hard quarantine, vi du `missing_title`, `mojibake_suspected`
  hoac `source_not_nfc`: `REVIEW_REQUIRED` theo mac dinh.
- Manual override duoc ap dung truoc rule mac dinh. Policy hien co `17` override
  tu dot review pilot.
- Policy SHA-256:
  `12cbb80193215e48040d7a5c17049e3bbb85cbcd5506cd92180e57dd19cee4d2`.

## Full candidate

- Thu muc: `outputs/vibiomir_phase1_full_20261014_v1`
- Status: `complete`
- Document: `1,075,599`
- Unique document ID: `1,075,599`
- Part: `922`
- Document co it nhat mot quality signal: `2,084`
- Candidate manifest SHA-256:
  `ca98fa74f6c0a32a339dbdd4bed9ee79cb7cf0ccc7f5d3cb59555510cf3d0d87`.
- Source snapshot SHA-256:
  `c83623f09670d7b8d7ad8542651f1c15c7be6175102e51b593b474a661a4e9d1`.

Quality signal tren full candidate:

| Flag | So document |
| --- | ---: |
| missing_title | 1,188 |
| html_contamination | 756 |
| navigation_heavy | 95 |
| unicode_replacement_character | 25 |
| boilerplate_signals | 9 |
| mojibake_suspected | 9 |
| challenge_text | 1 |
| source_not_nfc | 1 |
| very_short_text | 1 |

Tong theo flag co the lon hon so document vi mot document co the mang nhieu flag.

## Release artifacts

Release manifest:
`outputs/vibiomir_phase1_full_20261014_v1/release/phase1_release_manifest.json`

- `release/pass_document_ids/`: `922` Parquet shard, chua chinh xac
  `1,073,531` `doc_id` ma Phase 2 duoc phep su dung.
- `release/held_documents/`: `922` Parquet shard, chua `2,068` document bi giu
  cung status, reason va quality flags.
- `release/decision_ledger/`: `922` Parquet shard, chua quyet dinh cho du
  `1,075,599` document.

## Verification

- Da doc lai allowlist va held set: unique trong tung tap, intersection `0`,
  union `1,075,599`.
- Da tinh lai SHA-256 cua toan bo `2,766/2,766` release shard; tat ca khop
  release manifest.
- Khong co file `.partial` con lai trong release.
- Test: `18 passed` voi
  `.venv-phase1/bin/python -m pytest -q tests/workflows/vibiomir/test_phase1_pipeline.py`.

## Contract cho Phase 2

Phase 2 phai join/filter bang `doc_id` tu
`release/pass_document_ids/*.parquet`. Khong duoc dung truc tiep toan bo Phase 1
candidate, vi candidate van chua `887` document quarantined va `1,181` document
cho review.

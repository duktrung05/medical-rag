# Medical-RAG V4/V5 — Phase 1 Task 2 Implementation Report

Ngày hoàn tất: **14/10/2026** (Asia/Ho_Chi_Minh)

## Phạm vi đã thực hiện

- Đọc đầy đủ `PHASE1_EXISTING_PIPELINE_AUDIT.md` trước khi sửa code.
- Tạo environment riêng `.venv-phase1` bằng Python 3.11.15.
- Cài tối thiểu `pyarrow 21.0.0` và `pytest 9.1.1`; không dùng `sudo`, không sửa Python hệ thống.
- Triển khai selection, normalization, quality validation, streaming build, atomic output và checkpoint/resume.
- Chạy unit tests, integration test trên artifact production và full dry-run.
- Không chạy pilot/full build; không tạo clean corpus trong Task 2.

## File được tạo hoặc cập nhật

### Module mới

- `src/medrag/data/phase1/__init__.py`
- `src/medrag/data/phase1/schema.py`
- `src/medrag/data/phase1/selection.py`
- `src/medrag/data/phase1/normalization.py`
- `src/medrag/data/phase1/validation.py`
- `scripts/workflows/vibiomir/build_phase1_corpus.py`

### Test mới

- `tests/workflows/vibiomir/test_phase1_pipeline.py`
- `tests/integration/workflows/test_phase1_production_inputs.py`

### Khác

- `.gitignore`: thêm `.venv-phase1/`.
- `PHASE1_TASK2_IMPLEMENTATION_REPORT.md`: báo cáo này.

Không sửa bất kỳ raw data, extracted text, manifest/checkpoint crawler, source lock, audit manifest hoặc backup Phase 0 nào.

## Thiết kế và guard chính

- `document_state.parquet` là outer allowlist duy nhất; chỉ `phase0_status == "USABLE"`.
- Kiểm SHA-256 source lock, official corpus, query, Phase 0 state và release status.
- Kiểm đủ 1.220.598 ID–URL chính thức và đúng 1.075.599 ID `USABLE` duy nhất.
- Kiểm 9.814 `REVIEW_REQUIRED` vẫn ngoài allowlist và giao với 10 staging recovery ID bằng 0.
- Resolve canonical row theo `content_group` qua đúng 922 file được liệt kê trong release manifest.
- Kiểm exact ID–URL, source-file SHA-256, row count và SHA-256 của từng source text.
- Preflight hoàn tất toàn bộ trước khi build được phép tạo output directory.
- Lưu riêng `source_text` và `normalized_text`, cùng hai checksum và normalization version.
- Normalization chỉ áp dụng Unicode NFC, newline và horizontal whitespace có kiểm soát; không lowercase, không sửa dấu, số, thuốc, đơn vị hoặc phủ định.
- Quality validation chỉ gắn flags/reasons và `quality_review_required`; không tự động loại document.
- Output part và checkpoint được ghi atomically; resume chỉ chấp nhận fingerprint/checksum khớp.
- Output bị từ chối nếu chồng lấn `data/`, Phase 0, Step 2 hoặc official verification roots.

## Kết quả test

Lệnh:

```bash
.venv-phase1/bin/python -m pytest -q \
  tests/workflows/vibiomir/test_phase1_pipeline.py \
  tests/integration/workflows/test_phase1_production_inputs.py
```

Kết quả: **10 passed in 11.98s**.

Các test bao phủ:

- checksum/identity official corpus;
- allowlist production đúng 1.075.599 ID;
- loại 9.814 review ID và 10 staging ID;
- canonical `content_group`;
- source text checksum;
- bảo toàn medical text;
- fail closed trước output;
- không ghi đè Phase 0 fixtures/protected roots;
- checkpoint/resume idempotency;
- bounded pilot vẫn bắt buộc full preflight.

## Kết quả dry-run production

Lệnh:

```bash
.venv-phase1/bin/python -u scripts/workflows/vibiomir/build_phase1_corpus.py \
  --dry-run --batch-size 5000
```

Kết quả: **dry_run_complete**.

| Chỉ số | Kết quả |
|---|---:|
| Official/Phase 0 rows | 1.220.598 |
| Phase 0 `USABLE` duy nhất | 1.075.599 |
| `REVIEW_REQUIRED` bị loại | 9.814 |
| Staging ID trong allowlist | 0 |
| Content files verified | 922/922 |
| Physical content rows scanned | 1.086.883 |
| Canonical rows verified | 1.075.599 |
| Repeated-template hashes còn lại | 0 |
| Documents có quality-review flags | 1.225 |

Quality flags:

| Flag | Documents |
|---|---:|
| `missing_title` | 1.188 |
| `unicode_replacement_character` | 25 |
| `boilerplate_signals` | 9 |
| `challenge_text` | 1 |
| `source_not_nfc` | 1 |
| `very_short_text` | 1 |

Không document nào bị xóa bởi các flags trên. Dry-run không tạo `outputs/vibiomir_phase1_20261014_v1/`.

## Fingerprint dry-run

- Official corpus: `87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7`
- Query: `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08`
- Phase 0 state: `a8df05bda42862d643cc428c79e6237f43b5bab91c72f608b7b21aab1579462b`
- Release status: `98914ab9015399e7bc5c7336da564639e7fbcfe80a755d1d28667de6f14cb2f9`
- Release manifest: `aa3793dc7e822ee965707e9b970e4050cc486a485692f8aec68be8099330ea6e`

## Vấn đề còn tồn tại

- 1.225 document sẽ được giữ trong clean corpus nhưng cần review theo flags; đặc biệt 1 challenge-text record và 25 record có Unicode replacement character.
- Pilot mode hiện lấy các canonical row đầu tiên theo thứ tự release manifest; phù hợp kiểm tra vận hành/checkpoint nhưng không phải mẫu stratified theo source/domain/language.
- Full build sẽ đọc source lần thứ hai sau full preflight để bảo đảm không ghi clean data trước khi mọi source đã được xác minh. Đổi lại là tăng I/O có chủ đích.
- Chưa đo dung lượng output full; Task 3 nên kiểm tra disk headroom trước pilot/full build.

## Lệnh pilot đề xuất cho Task 3

Pilot vận hành 10.000 document, vẫn chạy full preflight trước khi ghi:

```bash
.venv-phase1/bin/python -u scripts/workflows/vibiomir/build_phase1_corpus.py \
  --output outputs/vibiomir_phase1_20261014_pilot_v1 \
  --pilot-documents 10000 \
  --batch-size 5000
```

Resume cùng pilot:

```bash
.venv-phase1/bin/python -u scripts/workflows/vibiomir/build_phase1_corpus.py \
  --output outputs/vibiomir_phase1_20261014_pilot_v1 \
  --pilot-documents 10000 \
  --batch-size 5000 \
  --resume
```

Không chạy hai lệnh trên trong Task 2.

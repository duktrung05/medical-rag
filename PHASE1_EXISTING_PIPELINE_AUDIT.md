# Phase 1 — Existing Pipeline Audit

Ngày audit: **14/10/2026** (Asia/Ho_Chi_Minh)  
Repository thực tế: `/data_hdd_16t/trungnguyen12/medical-rag/medical-rag`  
Phạm vi tài liệu này: **Task 1 — audit chỉ đọc trước khi xây Phase 1**. Không crawl, không re-extract, không chunk, không deduplicate, không index và không thay đổi artifact Phase 0.

## 1. Kết luận điều hành

Có thể xác định chính xác tập đầu vào Phase 1: lấy các dòng có `phase0_status == "USABLE"` từ:

`outputs/vibiomir_phase0_20261014/audit/document_state.parquet`

Artifact này có 1.220.598 dòng, SHA-256 hiện tại:

`a8df05bda42862d643cc428c79e6237f43b5bab91c72f608b7b21aab1579462b`

`audit_manifest.json` được đọc từ working tree và checksum của hai Parquet Phase 0 được tính lại. Số lượng khớp báo cáo nghiệm thu:

| `phase0_status` | Số dòng |
|---|---:|
| `USABLE` | 1.075.599 |
| `REVIEW_REQUIRED` | 9.814 |
| `DEFERRED` | 134.241 |
| `EXTRACTION_FAILED` | 266 |
| `CRAWL_FAILED` | 552 |
| `CRAWLED_SUCCESS` (metadata-only) | 126 |
| **Tổng** | **1.220.598** |

Danh sách này đủ rõ để làm allowlist Phase 1. Tuy nhiên chưa chạy bất kỳ bước Phase 1 nào trong Task 1. Trước lần chạy đầu tiên, implementation phải tự đọc toàn bộ ledger, kiểm lại đúng 1.075.599 ID duy nhất, đối chiếu URL và fail closed nếu sai dù chỉ một dòng.

Không được dùng `outputs/vibiomir_step2_20261013/release/document_status.parquet` với điều kiện `usable == true` làm allowlist. Release Step 2 còn 1.085.413 dòng usable; Phase 0 đã cách ly 9.814 dòng trong số đó, và `1.085.413 - 9.814 = 1.075.599`.

10 ID phục hồi usable mới chỉ nằm trong staging và không thuộc allowlist Phase 1:

`82086, 106695, 106840, 107200, 108063, 108227, 108264, 783862, 869573, 876706`

## 2. Nguồn chính thức và đối chiếu khóa nguồn

Nguồn được khóa trong `outputs/vibiomir_phase0_20261014/source_lock.json`:

| Thành phần | Đường dẫn đúng | Dòng | SHA-256 |
|---|---|---:|---|
| Corpus chính thức | `outputs/vibiomir_official_verify_20261012/links_corpus_vi_en_zh.parquet` | 1.220.598 | `87e44a792d7a99fb0fde5dff7f09f1a38a0a1fdf02ddd1ac93a005736e00b0a7` |
| Query chính thức | `outputs/vibiomir_official_verify_20261012/query.parquet` | 1.200 | `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08` |
| Dataset revision | `AIGuruTinix/ViBioMIR@71be0e1239a19dfd7a717e2bc13e3bec21bd0334` | — | — |

Các checksum trên đã được tính lại trong audit này và khớp `source_lock.json`/Step 1 manifest.

### Cảnh báo về corpus legacy

`data/raw/vibiomir/links_corpus.parquet` **không phải** corpus chính thức cho Phase 1:

- 4.394.718 dòng;
- SHA-256 `dee1153f5b9d9010dea18e1b7724b4915ff847b711d036bec8b933486802a971`;
- chỉ có 980.598 cặp ID–URL giống corpus chính thức;
- thiếu 240.000 URL PubMed chính thức và chứa 3.414.120 ID ngoài corpus chính thức.

`data/raw/vibiomir/query.parquet` lại byte-identical với query chính thức, nhưng Phase 1 không cần query. Để tránh nhầm nguồn, mọi guard vẫn phải trỏ tới file chính thức trong `outputs/vibiomir_official_verify_20261012/`.

## 3. Cấu trúc repository liên quan

| Khu vực | Vai trò thực tế | Quyết định Phase 1 |
|---|---|---|
| `outputs/vibiomir_phase0_20261014/` | Source lock, ledger trạng thái, recovery candidates, backup đã nghiệm thu | Chỉ đọc; đây là cổng vào bắt buộc |
| `outputs/vibiomir_step2_20261013/release/` | Ledger release và manifest chọn canonical content | Chỉ đọc; dùng để định vị content shard |
| `outputs/vibiomir_step2_20261013/content/` | Extracted document và passage đã materialize | Chỉ đọc các file `documents*`; bỏ qua toàn bộ `passages*` |
| `data/vibiomir/crawl/pages/` | Raw HTML gzip theo shard | Chỉ đọc khi cần provenance/validation; không glob toàn cây |
| `data/vibiomir/pubmed/<revision>/run_20261013_2gpu/raw/` | Raw PubMed XML gzip theo batch | Chỉ đọc khi cần provenance/validation |
| `data/vibiomir/pubmed/<revision>/run_20261013_2gpu/documents/` | Export PubMed trước materialization | Nguồn phụ để audit; clean corpus nên đọc canonical materialized documents |
| `src/extraction/` | HTML decoding/extraction cũ | Chỉ tái sử dụng helper an toàn; không re-extract trong Phase 1 |
| `src/medrag/data/` | Registry, contracts, loader/extraction V4 pilot | Tái sử dụng contract/hash/identity; không dùng pilot loader cho full corpus |
| `src/data/` | Generic schema, JSONL loader và normalizer | Không phù hợp trực tiếp với clean corpus quy mô đầy đủ |
| `r2ai/` | Pipeline thi đấu legacy: crawl/extract/chunk/index/rank | Không gọi trong Phase 1 |
| `scripts/workflows/vibiomir/` | Workflow kiểm định, materialization, checkpoint, backup | Tái sử dụng pattern manifest/checkpoint; không chạy lại workflow Phase 0 |
| `data/vibiomir/chunks*.parquet`, `bm25idx*`, vector/index artifacts | Chunk và index legacy | Ngoài phạm vi; không đọc để tạo clean document corpus |

Working tree hiện đã dirty với nhiều file modified/deleted/untracked có trước audit. Không được reset, cleanup hoặc refactor rộng. File báo cáo này là thay đổi duy nhất của Task 1.

## 4. Vị trí raw và extracted text

### 4.1 Raw HTML

Root vật lý:

`data/vibiomir/crawl/pages/<doc_id % 1000:03d>/<doc_id>.gz`

Hiện có 3.547.708 file `.gz`, là một superset legacy rất lớn. Không được suy luận phạm vi Phase 1 bằng cách quét root này. Đường dẫn raw canonical phải lấy từ `document_state.raw_path` và/hoặc document shard đã được chọn.

Crawl manifests nằm tại:

`data/vibiomir/crawl/manifest_*.parquet`

Có 1.102 manifest. Chúng là lịch sử crawler, không phải source of truth về `USABLE` của Phase 1 và không được sửa.

### 4.2 Raw PubMed

Root vật lý:

`data/vibiomir/pubmed/71be0e1239a19dfd7a717e2bc13e3bec21bd0334/run_20261013_2gpu/raw/*.xml.gz`

Có 1.320 raw batch. Provenance của từng document nằm trong `raw_file`/`raw_sha256` của document shard PubMed; nhiều document có thể cùng tham chiếu một raw batch.

### 4.3 Extracted text canonical hiện tại

Text đầy đủ nằm trong các file `documents*.parquet` được liệt kê trong:

`outputs/vibiomir_step2_20261013/release/release_manifest.json`

Manifest đang khóa 922 file content, tổng 1.086.883 document rows. Phân bố vật lý:

| `content_group` | Loại | File | Document rows |
|---|---|---:|---:|
| `legacy_html` | HTML | 834 | 833.892 |
| `pubmed` | PubMed | 48 | 239.770 |
| `saved_raw` | HTML | 1 | 222 |
| `baotayninh_css_fallback` | HTML | 1 | 1.198 |
| Các group recovery đã merge vào Step 2 | HTML | 38 | 11.801 |
| **Tổng** |  | **922** | **1.086.883** |

1.086.883 là số row vật lý đã scan, không phải số document canonical duy nhất. Release ghi nhận 1.204 row content trùng và 1.198 row bị source ưu tiên cao hơn override. Phase 1 phải chọn theo `document_status.content_group`, sau đó lọc tiếp bằng allowlist Phase 0.

Các file `passages*.parquet` đã tồn tại nhưng tuyệt đối không dùng làm input Phase 1 vì đây là artifact chunking từ giai đoạn trước.

## 5. Schema Parquet thực tế

Schema dưới đây được đối chiếu giữa Arrow footer trong file hiện hữu, schema producer trong working tree và manifest Step 1/Phase 0.

### 5.1 Official corpus và query

```text
links_corpus_vi_en_zh.parquet
  id: int64
  url: string

query.parquet
  id: int64 not null
  query: string not null
```

### 5.2 Phase 0 allowlist ledger

```text
document_state.parquet
  doc_id: int64
  official_url: string
  final_status: string
  content_group: string
  crawled_success: bool
  usable: bool
  extraction_failed: bool
  crawl_failed: bool
  deferred: bool
  review_required: bool
  phase0_status: string
  reason: string
  raw_path: string
  raw_sha256: string
  text_sha256: string
  raw_hash_verified: bool
  source: string
```

`recovery_candidates.parquet` dùng cùng schema và có 144.873 dòng; file này chỉ để triage, không phải đầu vào Phase 1.

### 5.3 Step 2 canonical status ledger

```text
document_status.parquet
  id: int64
  official_url: string
  hostname: string
  source_kind: string
  ledger_state: string
  final_status: string
  usable: bool
  content_group: string
  reason: string
```

SHA-256 hiện tại: `98914ab9015399e7bc5c7336da564639e7fbcfe80a755d1d28667de6f14cb2f9`.

### 5.4 Canonical HTML document shards

```text
id: int64
official_url: string
hostname: string
status: string
title: string
text: string
encoding: string
encoding_source: string
extraction_method: string
raw_path: string
gzip_sha256: string
raw_sha256: string
raw_bytes: int64
body_bytes: int64
text_sha256: string
error: string
```

### 5.5 Canonical PubMed document shards

```text
id: int64
official_url: string
source_kind: string
title: string
text: string
language_metadata: string
pmid: string
raw_file: string
raw_sha256: string
text_sha256: string
extractor_version: string
```

### 5.6 Legacy crawl manifest

```text
doc_id: int64
url: string
domain: string
status: string
http_status: int32
content_type: string
bytes: int64
final_url: string
error: string
```

Lưu ý: hai schema document HTML và PubMed không đồng nhất (`raw_path` so với `raw_file`, metadata encoding so với PMID/language). Phase 1 cần adapter riêng rồi mới ghi schema clean thống nhất; không nên ép đọc bằng một schema nullable chung ngay tại source.

## 6. Quan hệ `doc_id` → URL → raw → extracted text

Luồng liên kết canonical phải là:

```text
official corpus (id, url)
        ↓ exact id + exact URL
Phase 0 document_state (doc_id, official_url, phase0_status)
        ↓ chỉ phase0_status == USABLE
Step 2 document_status (id, official_url, content_group)
        ↓ content_group chọn đúng source, không chọn theo glob/row mới nhất
release_manifest.content_files[group]
        ↓ exact id + exact URL; verify file SHA-256
HTML/PubMed document row (title, text, text_sha256, raw provenance)
```

Các invariant bắt buộc:

1. `doc_id` Phase 0 phải tồn tại đúng một lần trong official corpus.
2. URL phải so sánh nguyên chuỗi với `official_url`; URL normalization chỉ được dùng để báo cáo, không được dùng để thay thế quan hệ chính thức.
3. Chỉ chấp nhận document row thuộc đúng `content_group` canonical.
4. `sha256(text.encode("utf-8"))` phải bằng cả document row `text_sha256` và Phase 0 `text_sha256`.
5. Không có ID ngoài allowlist; không thiếu ID; output cuối đúng 1.075.599 ID duy nhất.
6. Không đọc staging recovery để bù thiếu tự động.

Tên `raw_sha256` trong Phase 0 ledger cần xử lý cẩn thận: producer ưu tiên `gzip_sha256` cho HTML (`gzip_sha256 or raw_sha256`). Vì vậy trường này thường là checksum blob gzip đối với HTML, trong khi document shard vẫn giữ cả `gzip_sha256` và checksum body decompressed `raw_sha256`. Phase 1 phải bảo toàn provenance này, không đổi nghĩa hoặc ghi đè một giá trị bằng giá trị kia.

## 7. Thành phần có thể tái sử dụng

### Tái sử dụng trực tiếp

- `scripts/workflows/vibiomir/audit_phase0.py`: định nghĩa schema và chính sách allowlist đã nghiệm thu; chỉ đọc để giữ contract, không chạy lại vào output cũ.
- `outputs/vibiomir_step2_20261013/release/release_manifest.json`: danh sách canonical content files, loại source, row count và SHA-256.
- `src/medrag/core/contracts.py`: contract `Source`, cách tính `text_sha256`, giữ `official_id`, URL và provenance.
- `src/medrag/data/corpus_registry.py`: verify checksum file và identity guard. Với Phase 1 phải bổ sung exact URL equality trước; `normalized_url()` không thay thế official URL.
- Pattern `*.partial` + `os.replace`, write-once và checksum sidecar trong `materialize_html_content.py`, `materialize_pubmed_content.py`, `recover_failed_urls.py`.
- Cách stream Parquet bằng `ParquetFile.iter_batches()` trong các workflow hiện có; phù hợp quy mô 1,075 triệu document.

### Tái sử dụng có giới hạn

- `src/extraction/html_extractor.py`: `decode_html()` có xử lý HTTP/meta charset và `charset-normalizer`; chỉ hữu ích nếu một task tương lai được phê duyệt re-extract. Phase 1 hiện tại phải dùng extracted text đã khóa.
- `src/medrag/data/extraction.py`: có structure-aware HTML extraction nhưng khác extractor đã tạo Phase 0 content; không được trộn output trong cùng release.
- `src/data/preprocess.py`: có NFC, punctuation và whitespace helpers; chỉ nên tái sử dụng sau khi tách policy thành các phép biến đổi cấu hình được và có provenance.
- `src/medrag/data/metadata.py`: language heuristic có thể dùng ở giai đoạn metadata sau; không chạy trong Phase 1 Task 1.

## 8. Thành phần cần chỉnh sửa hoặc không nên dùng trực tiếp

| Thành phần | Vấn đề | Hướng xử lý cho implementation Phase 1 |
|---|---|---|
| `src/data/preprocess.normalize_text()` | Mặc định lowercase; collapse toàn bộ whitespace; làm mất ranh giới đoạn và thay đổi casing của thuốc/viết tắt | Không gọi trực tiếp. Tạo normalizer bảo toàn paragraph boundary, mặc định không lowercase, ghi transformation flags/version |
| `r2ai/extract.py` | Vừa extract vừa chunk; lấy ID từ toàn bộ crawler manifests; fallback `latin-1`; output là chunks | Không dùng trong Phase 1 |
| `src/medrag/data/prepared.py` | Pilot in-memory giới hạn 1.000 source; tự đọc raw, extract và build passages | Không dùng cho full corpus; chỉ tham khảo write-once manifest/hash |
| `src/medrag/data/loader.load_crawled_source()` | Suy đường dẫn raw từ ID và re-extract HTML | Không dùng để lấy text Phase 1; canonical raw path phải theo ledger |
| `src/data/schema.py` / `src/data/loader.py` | Generic chunk schema dùng ID string/JSONL, không khớp document schema ViBioMIR | Không mở rộng cưỡng bức; thêm schema Phase 1 riêng |
| `materialize_*_content.py` | Tạo passages cùng lúc với document materialization | Không gọi lại; chỉ tham khảo document schemas/checkpoints |
| `release/document_status.parquet` | `usable` rộng hơn Phase 0 allowlist 9.814 ID | Chỉ dùng resolver `content_group` sau khi đã inner join allowlist Phase 0 |

Không cần chỉnh sửa bất kỳ module hiện hữu nào trong Task 1.

## 9. Thành phần còn thiếu cần bổ sung ở task triển khai Phase 1

Đề xuất bổ sung tối thiểu, tách biệt với code crawler/retrieval:

- `src/medrag/data/phase1/schema.py`: schema canonical clean document, version hóa rõ.
- `src/medrag/data/phase1/selection.py`: load/validate allowlist 1.075.599 ID và resolve canonical content shard.
- `src/medrag/data/phase1/normalization.py`: các transform không mất thông tin, có unit test cho Việt/Anh/Trung, thuốc, đơn vị và Unicode.
- `src/medrag/data/phase1/validation.py`: count, uniqueness, exact ID–URL, text checksum, source checksum, excluded-set và staging guards.
- `scripts/workflows/vibiomir/build_phase1_corpus.py`: streaming runner, resume theo shard, write-once output.
- Tests tương ứng dưới `tests/workflows/vibiomir/` hoặc `tests/v4/`.

Task triển khai không được import chunk builder, BM25, FAISS, NER, ICD-10 hoặc taxonomy modules.

## 10. Checkpoint và manifest hiện có

Các cơ chế tốt đang có:

- HTML materialization chia batch 1.000 row, ghi `documents/part_*.parquet`, `passages/part_*.parquet` và `parts/part_*.json`.
- Sidecar giữ first/last input ID, số row/status, SHA-256 document/passages; resume chỉ chấp nhận checkpoint khớp.
- PubMed materialization xác minh SHA-256 input part và SHA-256 của output part trước khi skip.
- Recovery plan có write-once-or-verify; output Parquet/JSON được ghi qua `.partial` rồi `os.replace`.
- Release manifest giữ checksum cho đủ 922 cặp content file.
- Backup Phase 0 đã kiểm tra 847.112 raw paths bắt buộc hiện diện trong archive; `backup_manifest.json` có `status: complete`.

Thiếu đối với Phase 1:

- chưa có namespace/output manifest Phase 1;
- chưa có checkpoint ràng buộc với checksum allowlist Phase 0 và checksum release manifest;
- chưa có record-level normalization provenance;
- chưa có guard bắt buộc loại 9.814 review IDs và 10 staging recovery IDs;
- chưa có end-to-end assertion output đúng 1.075.599 ID.

## 11. Đường dẫn đầu vào và đầu ra đề xuất

### Input read-only bắt buộc

```text
outputs/vibiomir_phase0_20261014/source_lock.json
outputs/vibiomir_phase0_20261014/audit/audit_manifest.json
outputs/vibiomir_phase0_20261014/audit/document_state.parquet
outputs/vibiomir_step2_20261013/release/release_manifest.json
outputs/vibiomir_step2_20261013/release/document_status.parquet
outputs/vibiomir_step2_20261013/content/**/documents*.parquet
outputs/vibiomir_official_verify_20261012/links_corpus_vi_en_zh.parquet
```

Raw roots chỉ dùng cho provenance/checksum có kiểm soát, không làm nguồn text chính.

### Output mới đề xuất

```text
outputs/vibiomir_phase1_20261014_v1/
  clean_documents/
    part-*.parquet
  checkpoints/
    part-*.json
  reports/
  phase1_manifest.json
  source_snapshot.json
```

Runner phải từ chối chạy nếu output version đã tồn tại, trừ chế độ resume khi toàn bộ input fingerprint/checkpoint khớp. Không tạo symlink hoặc ghi ngược vào `data/`, `outputs/vibiomir_step2_20261013/` hay `outputs/vibiomir_phase0_20261014/`.

## 12. Rủi ro ảnh hưởng dữ liệu cũ và biện pháp khóa

| Rủi ro | Mức | Guard bắt buộc |
|---|---|---|
| Dùng nhầm corpus legacy 4,39 triệu dòng | Critical | Pin path + SHA-256 official corpus trước mọi output |
| Lấy Step 2 `usable` thay Phase 0 `USABLE` | Critical | Phase 0 allowlist là outer gate; assert đúng 1.075.599 |
| Tự merge 10 staging recovery IDs | Critical | Hard exclusion + manifest assertion |
| Quét raw tree và kéo ID ngoài official corpus | Critical | Chỉ follow raw path từ canonical allowlist row |
| Chọn sai row trong 1.204 duplicate content rows | High | Resolve bằng `content_group` và release manifest priority đã khóa |
| Làm biến dạng medical text khi lowercase/collapse whitespace | High | Non-destructive policy; lưu original text hash và transformation version |
| Ghi đè Phase 0/Step 2 khi resume | Critical | Output namespace mới, write-once, `FileExistsError`, atomic rename |
| Nhầm nghĩa checksum raw HTML compressed/decompressed | High | Tách trường `raw_gzip_sha256` và `raw_body_sha256`; không đổi nghĩa field nguồn |
| Working tree dirty làm lẫn thay đổi | High | Chỉ sửa file Phase 1 mới; không reset/cleanup/refactor ngoài scope |
| Runtime hiện tại thiếu `pyarrow` | High | Chọn môi trường Python >=3.11 đúng `pyproject.toml`, cài dependency trong venv/project environment; không dùng `sudo`/system install |

Ở thời điểm audit, shell không có lệnh `python`; `python3` và `.venv-r2ai/bin/python` đều không import được `pyarrow`, trong khi `pyproject.toml` yêu cầu Python >=3.11 và `pyarrow>=15`. Đây là blocker vận hành cho task triển khai/validation, không làm mơ hồ allowlist đã khóa. Không có package hệ thống nào được cài trong audit này.

## 13. Gate trước khi triển khai tiếp

Phase 1 chỉ được bắt đầu khi runner có thể chứng minh, trước khi ghi clean data:

1. Source lock và official corpus/query hashes khớp.
2. Phase 0 audit artifact hash khớp manifest.
3. Ledger có 1.220.598 `doc_id` duy nhất, URL khớp official corpus.
4. Filter `phase0_status == "USABLE"` cho đúng 1.075.599 ID duy nhất.
5. Giao với `REVIEW_REQUIRED` bằng 0; giao với 10 staging IDs bằng 0.
6. Mỗi ID resolve được đúng một canonical document row theo `content_group`.
7. Text hash và URL của từng canonical row khớp Phase 0 ledger.
8. Output root mới, versioned và chưa tồn tại hoặc là checkpoint resume khớp tuyệt đối.

Nếu một gate thất bại, runner phải dừng trước khi tạo bất kỳ clean document part nào.


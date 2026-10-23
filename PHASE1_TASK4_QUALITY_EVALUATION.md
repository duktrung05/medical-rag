# Phase 1 Task 4 — Data Quality Evaluation

Ngày đánh giá: 14/10/2026

## Kết luận

Pilot v1 không đủ đại diện để nghiệm thu chất lượng vì 10.000/10.000 document đều
là PubMed. Tập review Task 4 gồm đúng 500 document:

- 400 document từ pilot v1.
- 100 document `REVIEW_REQUIRED` làm quarantine control, đọc từ canonical content
  shard; các row này không được đưa vào clean corpus.

Không phát hiện normalization làm mất nội dung, tên thuốc, liều lượng, số liệu hay
phủ định trong 400 pilot rows. Tuy nhiên Task 4 xác nhận ba thiếu sót cần refinement:

1. Pilot lấy N dòng đầu nên không có HTML và không kiểm được tiếng Việt/Trung.
2. `language_metadata` của PubMed là ngôn ngữ publication, không bảo đảm là ngôn
   ngữ abstract; 73/400 row không đồng thuận với ngôn ngữ nội dung.
3. Quality rules ban đầu có false positive với ký hiệu di truyền `A-->G`,
   `8q22-->qter` và từ “home” trong ngữ cảnh home care.

## Tập review và bằng chứng

Output versioned:

```text
outputs/vibiomir_phase1_quality_20261014_v1/
  review_sample.parquet
  review_sample.csv
  quality_summary.json
  errors.json
```

`review_sample.parquet` giữ full `source_text` và `normalized_text`; CSV giữ metadata,
verdict, signals và preview 500 ký tự để review nhanh. Mỗi row có checksum, language,
HTML/menu/ad/encoding signals và bốn preservation invariants.

Phương pháp:

- Xác minh checksum của mọi shard pilot trước khi lấy mẫu.
- Chọn deterministic sample, không thay đổi nguồn.
- So sánh exact lexical sequence sau NFC và bỏ whitespace.
- So sánh non-empty lines, medical numeric-unit values và negation tokens.
- Kiểm HTML, menu/footer, quảng cáo, mojibake và quality flags.
- Review thủ công các anomaly candidates, mẫu đại diện từng language/source stratum
  và cả bốn nhóm lý do quarantine.

## Kết quả pilot v1 — 400 document

| Kiểm tra | Kết quả |
|---|---:|
| Source | 400 PubMed, 0 HTML |
| Nội dung phát hiện là English | 400 |
| HTML thực sự còn sót | 0 |
| Mojibake/replacement character | 0 |
| Lexical preservation failure | 0 |
| Non-empty line preservation failure | 0 |
| Medical value preservation failure | 0 |
| Negation preservation failure | 0 |
| Normalization thay đổi trong sample | 0 |

Vì không có HTML trong pilot v1, kết quả `0 HTML contamination` không thể dùng để
kết luận corpus HTML sạch. Đây là coverage failure của pilot, không phải bằng chứng
chất lượng HTML.

### Heading, câu và đoạn

- Không có ký tự lexical nào bị mất giữa `source_text` và `normalized_text`.
- Chuỗi non-empty lines không đổi; không thấy heading/paragraph bị xóa.
- Pilot v1 sample không có thay đổi normalization, nên không quan sát được ca khó
  có Unicode space hoặc nhiều blank lines.
- Không thể kết luận extraction trước Phase 1 có bỏ nội dung từ HTML gốc hay không,
  vì Task 4 không đọc kho raw legacy và Phase 1 không re-extract.

### Thuốc, liều và số liệu

67/400 row baseline chứa numeric-unit patterns trong kiểm tra ban đầu; mọi value
được giữ nguyên. Exact lexical invariant cũng bảo đảm tên bệnh/thuốc và dấu phủ định
không bị normalization viết lại. Không chạy Medical NER theo giới hạn nhiệm vụ.

### Language Detection

Pilot v1 chưa có trường detected language. Khi detector độc lập được áp dụng trong
evaluation, 400/400 abstract được xác định là English. Metadata lại gồm `eng`, `chi`,
`ger`, `fre`, `jpn`, v.v.; kiểm tra nội dung cho thấy nhiều abstract đã được cung cấp
bằng tiếng Anh dù metadata mô tả ngôn ngữ publication. Vì vậy dùng trực tiếp
`language_metadata` làm content language là sai contract.

## Quarantine control — 100 document

Phân tầng theo lý do:

| Lý do Phase 0 | Mẫu | Kết quả review |
|---|---:|---|
| `duplicated_template_content` | 32 | Nhãn được xác nhận |
| `homepage_or_redirect_content` | 32 | Nhãn được xác nhận |
| `login_or_access_gate_content` | 32 | Nhãn được xác nhận |
| `hex_or_asset_noise` | 4 | Nhãn được xác nhận |

Trong lần kiểm tự động đầu, bốn duplicate-template rows chưa đủ signal menu theo
regex. Review thủ công xác nhận một row là author-bio template lặp và ba row Trung
là navigation/template content. Kết luận cuối: **100/100 quarantine labels trong
mẫu được xác nhận; 0 trường hợp xác nhận là gắn `REVIEW_REQUIRED` nhầm.** Không row
nào được tự động khôi phục.

## Danh sách lỗi cần sửa

| ID | Lỗi | Mức độ | Hướng sửa |
|---|---|---|---|
| Q-01 | Pilot v1 chỉ có PubMed | Critical cho QA | Sampling phân tầng source/language |
| Q-02 | Không có content language độc lập | High | Detector versioned, metadata chỉ là hint |
| Q-03 | PubMed metadata khác ngôn ngữ abstract | High | Lưu cả metadata và detected language/agreement |
| Q-04 | `-->` bị hiểu là HTML comment close | High | Không flag standalone `-->`; nhận actual tag form |
| Q-05 | Lặp từ “home” bị hiểu là navigation | Medium | Dùng ≥3 loại menu signals khác nhau |
| Q-06 | So sánh `90 mmHg` và `90 mmHg` báo đổi value | Medium | Canonical whitespace trước khi so numeric-unit |
| Q-07 | Vietnamese list text thiếu tín hiệu trong body | High | Detect trên `title + text`, mở rộng lexical cues |

Các lỗi Q-04–Q-06 là lỗi validator/evaluator, không phải lỗi dữ liệu nguồn hoặc
normalization. Không có bằng chứng để sửa/xóa nội dung document.

## Giới hạn đánh giá

- Checklist cấu trúc và normalization được áp dụng cho toàn bộ 500 rows; review
  semantic thủ công tập trung vào mọi anomaly và các strata đại diện.
- Không đối chiếu raw HTML, vì user đã giới hạn pipeline vào dataset cập nhật và
  cấm re-extraction.
- Chất lượng y khoa/lâm sàng không được nghiệm thu trong Task 4.

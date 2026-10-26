# Báo cáo thực thi vòng 5 — 24/10/2026

## Kết luận

- Champion BTC vẫn là S2, FINAL `0,2318`.
- R4 high-confidence được BTC chấm `0,2316` (FINAL `-0,0002`; chunks F2
  `-0,0004`). Receipt nằm cạnh ZIP R4 và đánh dấu NO-GO.
- R3 fine-tune có tác dụng xếp hạng trên cùng input S2: tăng validation ROC-AUC
  `0,5899 → 0,6929`, AP `0,1144 → 0,2199`, query-macro AP
  `0,2875 → 0,3965`.
- Không có policy chọn context/chunk nào qua gate development của vòng 5. Không
  tạo submission R5; S2 tiếp tục là bản nên dùng.

## Tác vụ và kết quả

### So sánh cùng input

Hai model BGE gốc và R3 được chấm trên đúng 72.000 chunk S2, với cùng query,
title, text, tokenizer và giới hạn 512 tokens. Validation chỉ tính trên các
query đã khóa trong split; đây là dữ liệu feedback lịch sử, không phải official
F2.

| Metric validation | BGE gốc | R3 fine-tuned |
|---|---:|---:|
| ROC-AUC | 0,5899 | **0,6929** |
| Average precision | 0,1144 | **0,2199** |
| Query macro AP | 0,2875 | **0,3965** |
| Query macro MRR | 0,3872 | **0,5436** |

Kết quả xác nhận fine-tune giúp phân biệt các chunk đã gán nhãn khi so cùng
input. Nó không tự chứng minh submission F2 sẽ tăng.

### Context dài và truncation

Trong S2, `64.079/72.000` cặp vượt 512 tokens. Pool context chính xác gồm
188.805 ứng viên trong danh sách documents S2; 72.000 chunk hiện hành được
replay khớp đúng.

R3 được chấm trên các cửa sổ 512 tokens trượt, stride 128, lấy max logit cho
mỗi context. Trên validation, so với first-512:

| Metric | First-512 | Max trên cửa sổ |
|---|---:|---:|
| AP | **0,2199** | 0,2091 |
| Query macro AP | 0,3965 | **0,3973** |
| Query macro MRR | 0,5436 | **0,5499** |

Do AP tổng giảm, max-window bị loại theo gate đã đặt. R3 first-512 được dùng để
replay selection trên pool S2; chính sách vẫn giữ nguyên document list.

### Selection replay

Baseline development: historical parent/exact `1014/812`, manual `44/43`.

- Policy thay tối đa một parent, margin 3 điểm: historical parent/exact
  `1020/815`, manual `46/45`, nhưng cần 298 replacements trên 340 query có
  nhãn.
- Margin 4 điểm: historical exact không tăng (`812`), dù số replacements còn
  208.
- Margin 5 điểm: historical parent/exact giảm còn `1013/811` với 99 thay đổi.
- Margin 6 điểm: không có gain exact (`812`) với 32 thay đổi.
- Policy thay context trong cùng parent làm historical exact giảm `812 → 772`;
  không đạt.

Vì các policy tăng exact đều cần thay quá nhiều kết quả, không có policy nào
qua development gate với guardrail thay tối đa 10% query có nhãn và yêu cầu
không giảm positives ở cả hai nhóm. Validation không được dùng để cứu một policy
đã không qua development. Kết quả cuối `selected_policy: null`,
`VALIDATION_GATE_FAIL`, không có ZIP R5.

## Artifacts

- Same-input evaluation: `outputs/v4/r5_same_input_evaluation_20261024/evaluation.json`
- Context candidate manifest: `outputs/v4/r5_s2_context_candidates_20261024/manifest.json`
- Sliding-window evaluation: `outputs/v4/r5_window_scoring_evaluation_20261024/evaluation.json`
- Selection policy và replay: `outputs/v4/r5_context_selection_20261024/`
- Điểm BTC R4 do người dùng cung cấp: `outputs/v4/r4_high_confidence_submission_20261024/scoring_receipt_user_20261024.json`

Điểm R5 chưa được đo bởi BTC vì gate local không chọn candidate. Mốc `0,4`
chưa đạt; kết quả này định vị được rằng reranker không phải nút thắt duy nhất:
candidate retrieval và chất lượng evidence/context vẫn cần cải thiện mà không
phá các chunk đang tốt.

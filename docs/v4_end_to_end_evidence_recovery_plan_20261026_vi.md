# Kế hoạch một candidate phục hồi document và evidence

Trạng thái: **PLAN — chưa chạy inference hoặc tạo submission mới**. Mốc so sánh
là cap-10: BTC FINAL `0,2324`, DOCS_F2 `0,2788`, CHUNKS_F2 `0,1860`;
1.200 queries, 140.095 documents và 72.000 chunks. Mục tiêu của vòng này là
**một candidate hoàn chỉnh trên 1.200 queries và một lần chấm BTC**, không
sweep depth/cap/window rồi chọn theo kết quả.

## Vì sao chọn thay đổi này

[Audit 87 manual evidence](v4_manual87_recovery_direction_20261026_vi.md)
xác nhận tất cả 87 evidence có trong source và indexed passages, nhưng output
cap-10 chỉ chứa đầy đủ 65. Trong 22 misses: 11 parent không vào candidate
pool, 9 document có mặt nhưng không được cấp chunk slot, 2 chunk chọn sai
context. Ngoài ra, một số parent có mặt nhưng passage/context được chọn không
chứa evidence. Vì thế candidate này sửa **cả đường từ parent đến context rồi
đến slot**. Các số manual là phép kiểm tra khả năng bảo toàn evidence, không
phải dự báo điểm BTC.

Ví dụ kiểm chứng: q610/305379 có passage đúng ở dense rank 142, nhưng bị cắt
ở mixed-item rank 406. R7 parent-floor đưa document vào rank 104, song vẫn
không cấp chunk. Chỉ sửa fusion sẽ không sửa CHUNKS_F2 cho ca này.

## Chính sách candidate duy nhất, khóa trước khi chạy

1. **Thu hồi parent.** Giữ toàn bộ candidate của cap-10. Với official query
   gốc, lấy document BM25, passage BM25 và dense ở depth `800`/branch; gộp
   theo parent **trước khi cắt** và giữ tối đa `800` unique parents. Không
   chạy query expansion hoặc sweep depth. Tái dùng đúng index/model đã khóa;
   parent ngoài cap-10 cần được rerank trước khi có thể vào output.

2. **Xếp document.** Chấm một source-grounded seed/parent bằng BGE reranker
   gốc, tái dùng exact cached scores khi input hash khớp. Rút còn top-200
   reranked parents, rồi union với documents cap-10 làm pool evidence. Không
   tự thêm document từ manual IDs hay historical positives. Khóa tie-break
   theo parent ID và input hashes.

3. **Tìm evidence trong parent.** Với mỗi parent trong pool, lấy các indexed
   passages của chính parent và giữ nguyên context cap-10 làm fallback. Sinh
   cửa sổ 192 tokens, stride 96, có offsets về source thật; chấm bằng cùng
   pinned reranker, áp cùng quy tắc cho parents cũ và mới. Chọn context xuất
   từ đoạn source liên tục quanh cửa sổ thắng, tối đa 5.500 ký tự; nếu hòa
   điểm giữ context cap-10. Khóa cách cắt context trước scoring, không dùng
   manual evidence text để chọn cửa sổ. Ghi số passage/window trước và sau
   mỗi lần cắt; không âm thầm bỏ passage đúng khỏi pool.

4. **Cấp documents và 60 chunk slots cùng nhau.** Bắt đầu từ output cap-10.
   Cho phép tối đa `10` chunk-parent replacements/query dựa trên evidence
   score so với victim; trong đó tối đa `3` entrants là documents mới. Một
   document mới chỉ được thêm vào `relevant_docs` khi nó nhận chunk slot;
   giữ toàn bộ documents cap-10 để tránh lặp lại control loss của R7. Mỗi
   document có tối đa một chunk. Chọn lại context cho parent đang giữ slot
   theo bước 3; không tăng số slots trên 60.

Đây là **một policy duy nhất**. Không đổi max swaps, depth, window length,
margin hoặc số documents mới sau khi xem manual/validation/BTC. DOCS_F2 có
thể giảm vì entrants sai làm giảm precision; giới hạn ba entrants/query làm
rủi ro này có biên rõ ràng. CHUNKS_F2 chỉ có cơ hội tăng nếu evidence scorer
chọn được parent và source span phù hợp hơn.

## Đường thực thi và chi phí

- Tạo namespace mới `outputs/v4/end_to_end_recovery_20261026/`; lưu effective
  policy, SHA-256 của index/queries/champion/model, pair IDs, scores, source
  offsets, document/slot changes và ZIP. Không sửa S2 hoặc cap-10 artifacts.
- Full-index CPU scan của R11 vượt 4 CPU-hours ngay trên 94 queries. Để chạy
  đủ 1.200 queries, dùng SQL-aggregate BM25 read-only và batched dense search
  trên chính FP32 matrix hiện có; dùng GPU khi runtime sẵn sàng. Preflight
  **kỹ thuật** trên q53/130/380 phải khớp frozen branch prefixes và S2 fusion.
  Sai parity thì sửa backend, không chuyển sang một retrieval policy khác.
- Lập manifest và trần số pairs trước khi chấm BGE. Dùng pinned CUDA FP16
  runtime và exact-batch anchors như cap-10; checkpoint per query/worker.
  GPU chưa sẵn sàng trên host hiện tại thì giữ package ở trạng thái ready;
  không dùng CPU logits để trộn với cached FP16 scores.
- Tạo đúng một ZIP cho 1.200 queries. Validate source substring, parent
  membership, 60 chunks/query, max 200 documents/query, duplicate policy,
  full query coverage, CRC và ZIP dưới 100 MB. Nếu context mở rộng làm ZIP
  vượt cap, dùng một truncation rule đã khóa quanh scored window và giữ
  source-grounded text; không chọn rule theo manual labels.

## Một lần kiểm kết quả, không mở vòng pilot mới

Sau khi có full candidate, đối chiếu 87 manual pairs và 10 retained controls
với cap-10 theo query-parent identity. Báo gross gains/losses tại document,
slot và exact-context levels; không biến historical preservation thành gold.
**Chỉ submit khi** ZIP/grounding PASS, giữ toàn bộ 10 controls trong documents,
final manual containment đạt ít nhất `70/87` (cap-10 là `65/87`) và mất tối đa
`2/65` evidence manual đang được giữ. Các ngưỡng này là điều kiện tiết kiệm
một lượt BTC, không phải ước lượng F2. Nếu fail, kết thúc candidate với
nguyên nhân cụ thể; không tự mở policy thứ hai.

Nếu đủ điều kiện, chấm **đúng một lần** cùng BTC phase với cap-10. Lưu receipt
gắn SHA-256 ZIP. Quyết định giữ candidate dựa trên FINAL và riêng DOCS_F2,
CHUNKS_F2; chỉ gọi là cải thiện điểm khi BTC thực đo. Mốc thành công:
FINAL `>0,2324` và ít nhất một trong DOCS_F2 `>0,2788` hoặc CHUNKS_F2
`>0,1860`. Nếu BTC không tăng, giữ cap-10 làm champion và ghi lại một
failure mode, không suy từ manual proxy rằng scorer sai.

Đầu ra bắt buộc: implementation + tests contract, full run manifest,
per-query diff, validated submission ZIP và báo cáo quyết định với hoặc
không có BTC receipt. Kế hoạch này là bước cải thiện cục bộ có căn cứ;
nó không cam kết đạt FINAL `0,4`.

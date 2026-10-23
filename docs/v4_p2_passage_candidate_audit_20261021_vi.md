# P2: Kiểm tra passage selection và pair budget (21-10-2026)

## Phạm vi đã thực hiện

Khóa lại sáu query P0 và thử ba kế hoạch pair không dùng historical labels để chọn passage:

- **P2-A0:** parent-first hiện tại, 200 parents × 1 passage.
- **P2-A1:** 200 parents × 1 passage, chọn passage có dense similarity cao nhất trong các passage của parent đã hiện diện trong fused candidates; nếu parent không có dense passage, chọn passage có fused score cao nhất. Tie-break cố định bằng passage ID.
- **P2-A2:** lấy 190 parents đầu theo fused parent rank, mỗi parent một passage như A0; dành 10 pairs còn lại cho passage thứ hai của 10 parents có alternative passage được dense đánh giá cao nhất. Toàn bộ lựa chọn được quyết định trước rerank, tổng đúng 200 pairs/query.

Pair plan chỉ đọc traces và cached scores P0, không gọi retrieval/model. Có 6 cặp passage mới duy nhất chưa có score; chúng được chấm **một lần** bằng đúng reranker revision P0 trên CPU, lưu trong bản sao score cache. Sau đó cả 18 query/arm được replay từ cache, mỗi score được kiểm bằng query hash, parent ID, passage ID và input hash. A0 khớp toàn bộ selected/context output đã khóa; mọi arm xuất đúng 60 chunks/query và qua source-grounding validation. Không gọi retrieval mới.

| Arm | Pair/query | Scored parents/query | Passage khác A0 | Exact anchors / 37 | Mean LCS proxy |
|---|---:|---:|---:|---:|---:|
| P2-A0 | 200 | 200 | 0 | 21 | 0.649946 |
| P2-A1 | 200 | 200 | 38 | 21 | 0.650139 |
| P2-A2 | 200 | 190 | 10 passage thứ hai/query | 20 | 0.622919 |

Ban đầu P2-A1 thiếu 4 arm-specific scores, P2-A2 thiếu 3; hai arm dùng chung một cặp chưa chấm, nên chỉ cần **6 model calls mới**. Số model calls không tính lại các cặp đã có cache. Cả ba arm có 200 pairs/query; P2-A2 đánh đổi 10 parents để có 10 passage thứ hai.

| Query | A0 exact / LCS | A1 exact / LCS | A2 exact / LCS |
|---|---:|---:|---:|
| 53 | 6 / 0.787 | 6 / 0.837 | 6 / 0.787 |
| 130 | 1 / 0.266 | 1 / 0.266 | 0 / 0.123 |
| 380 | 5 / 1.000 | 5 / 1.000 | 5 / 1.000 |
| 754 | 3 / 0.600 | 3 / 0.600 | 3 / 0.600 |
| 771 | 4 / 0.571 | 4 / 0.571 | 4 / 0.571 |
| 1144 | 2 / 0.777 | 2 / 0.666 | 2 / 0.777 |

Ở A1 query 53, LCS của `128866` tăng từ `0.232` lên `0.686` nhưng vẫn chưa exact. Query 1144 lại giảm vì passage của positive `17620` đổi và proxy của nó từ `0.447` thành `0`; mean LCS gộp tăng rất ít đã che mất lỗi theo query này. Ở A2 query 130, parent `2911632` bị loại khỏi 190 scored parents dù ở A0 nó có exact context trong top-60. Trên toàn sáu query, lấy 190 parents đầu theo fused rank làm mất **9 A0 top-60 documents**, trong đó có `2911632`. Đây là chi phí parent coverage quan sát được.

## Hai trường hợp chọn sai passage

`53/128866` hiện ở final rank 5. A0 chọn passage mở đầu của bài về tức ngực/khó thở, có rerank score `5.543`, nhưng context không chứa historical anchor. Có hai passage khác mà context chứa anchor; một passage có cached score `4.723` và nếu chỉ thay seed này thì parent vẫn ở khoảng rank 17. P2-A1 và P2-A2 hiện chọn passage dense rank 6, cached score `5.074` và rank khoảng 10, nhưng context của passage đó vẫn không chứa anchor. Vì vậy quy tắc dense ưu tiên không tìm được passage mong muốn, dù passage đó đang có trong candidate set.

`1144/142195` hiện ở final rank 40. A0 chọn passage về phòng ngừa dậy thì sớm, cached score `3.834`. Passage còn lại có nội dung theo dõi triệu chứng, khi nào đưa trẻ đi khám và tư vấn; context của nó chứa historical anchor. Score cache của passage này là `3.094`: nếu chỉ thay seed thì parent ở khoảng rank 60. Nhưng dense rank của nó là 110 so với 31 của passage hiện tại, nên P2-A1 vẫn chọn passage cũ. P2-A2 cũng không cấp passage thứ hai cho parent này; ngay cả nếu cấp, policy chọn passage có rerank score cao nhất vẫn giữ passage cũ. Đây là bằng chứng trực tiếp rằng **thêm pair và lấy max rerank score không tự sửa lỗi passage choice**.

Những counterfactual theo anchor ở hai đoạn trên chỉ dùng để hiểu giới hạn và kiểm tra chi phí slot. Chúng không phải rule online; passage chứa anchor có thể vẫn kém liên quan hơn một passage khác.

## Truncation và chất lượng relevance

Query `130` có `178/200` pairs dài hơn giới hạn 512 tokens; năm query khác không có pair nào vượt giới hạn trong baseline P0. Với đúng tokenizer/revision P0, passage của `130/2968260` dài 538 pair tokens và bị cắt xuống 512: **507/543 ký tự passage vẫn nằm trong input**, chỉ 36 ký tự cuối bị cắt. Mô tả trẻ chảy nước mắt, cụm “viêm túi lệ sơ sinh”, câu trả lời phủ định và câu về thuốc nhỏ mắt đều nằm trước cutoff. Vì vậy, trường hợp này không mất nội dung lâm sàng chính do truncation. Passage `130/553119` chỉ 474 pair tokens nhưng vẫn bị xếp thấp; cần xem relevance/ngôn ngữ/ngữ cảnh thay vì mặc định tăng token limit.

Đánh giá passage cần ghi nhãn độc lập về mức trả lời câu hỏi, độ đúng ngữ cảnh bệnh nhân và giá trị xét nghiệm. Ví dụ `130/2968260` cùng chủ đề mắt nhưng mô tả người bệnh khác và câu trả lời phủ định bệnh được hỏi; `754/2713777` ghi CA125 40.41 thay vì 64.80 và thiếu bước xử lý; `1144/142195` có passage thứ hai nói trực tiếp hơn về theo dõi trẻ. Anchor containment và LCS chỉ là proxy để theo dõi thay đổi, không phải nhãn relevance.

Đã tạo packet **406 query–context items** từ chunks của ba arm và toàn bộ passage ứng viên của bảy diagnostic parents, gồm cả passage không arm nào chọn. Dữ liệu được khử trùng lặp và ẩn arm, score, rank, historical anchors khỏi file dành cho người đánh giá. Rubric chấm `0` (không trả lời hoặc sai đối tượng), `1` (cùng chủ đề nhưng thiếu câu trả lời/ngữ cảnh), `2` (trả lời trực tiếp, đúng ngữ cảnh), kèm cờ patient/numeric/contradiction/grounding. `outputs/v4/p2_relevance_review_packet_20261021_v2/` hiện chỉ là **template chưa được người đánh giá chấm**; `item_map_for_analysis.json` chỉ mở sau khi hoàn thành nhãn độc lập. Chưa có metric relevance độc lập để quyết định promote.

## Quyết định và điều kiện tiếp theo

Giữ A0. P2-A1 dense-priority chưa tăng exact anchors, không tìm đúng hai passage mục tiêu và làm giảm LCS ở một query. P2-A2 theo phân bổ 190+10 giảm exact anchors và parent coverage. Không promote hai arm. Cần đánh giá relevance trên development/holdout độc lập trước khi thử policy passage mới; historical anchor containment và trần oracle 28/37 không phải nhãn relevance online. Official CHUNKS_F2 chưa đo.

Artifacts: `outputs/v4/p2_passage_candidate_replay_20261021/` (pair allocation, target contexts, replay 18 query/arm, selected chunks, validation và SHA-256), `outputs/v4/p2_missing_pair_scores_20261021/` (sáu score mới, bản sao cache, model revision), `outputs/v4/p2_truncation_20261021/` (token cutoff chính xác) và `outputs/v4/p2_relevance_review_packet_20261021_v2/` (blinded review template). Harness: `scripts/v4/audit_p2_passage_candidates.py`, `scripts/v4/score_p2_missing_pairs.py`, `scripts/v4/audit_p2_truncation.py`, `scripts/v4/prepare_p2_relevance_review.py`.

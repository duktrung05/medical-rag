# P1: Replay chunk allocation (21-10-2026)

## Phạm vi và cách chạy

Replay dùng đúng 6 query khóa từ P0 (`53, 130, 380, 754, 771, 1144`) và cached scores trong P0; không chạy retrieval hoặc model. Mỗi query giữ nguyên 200 `relevant_docs`/scored parents. Các arm chỉ thay đổi 60 documents được cấp chunk, tối đa một chunk mỗi document. A0 được đối chiếu chính xác với chunks/context của P0. Historical positives chỉ được dùng để tính metric sau khi allocation đã hoàn tất.

Pool ghi một record cho mỗi scored parent, gồm query/doc ID, final/fused/dense/BM25 rank, passage ID, rerank score, query coverage và chunk-selection score. `R` là min-max rerank score trong 200 parents của query. `F` là min-max của trung bình rank strength từ fused, dense và BM25 (BM25 rank cap 400; dense cap 200; branch thiếu đóng góp 0). `C` là tỷ lệ token query duy nhất xuất hiện trong passage sau NFKC/lowercase. A2 thử ba bộ trọng số dễ giải thích: `(0.5, 0.3, 0.2)`, `(0.4, 0.3, 0.3)`, `(0.6, 0.2, 0.2)` cho `alpha, beta, gamma`. A3 chọn 40 theo hybrid `(0.5,0.3,0.2)`, 10 theo dense rank, 10 theo coverage; dedupe theo `doc_id`, rồi backfill bằng hybrid nếu quota thiếu.

## Kết quả tổng hợp

Có 37 historical positive anchors trên 6 query, nhưng đây là labels lịch sử chưa đầy đủ nên các tỷ lệ dưới đây chỉ dùng để chẩn đoán, không phải precision chính thức.

| Arm | Exact anchors / 37 | Mean LCS proxy | LCS ≥ 0.4 | Mean query coverage | Chunks |
|---|---:|---:|---:|---:|---:|
| A0 baseline | 21 (56.8%) | 0.650 | 25 | 0.527 | 360 |
| A1 rerank | 21 (56.8%) | 0.650 | 25 | 0.527 | 360 |
| A2 `(0.5,0.3,0.2)` | 13 (35.1%) | 0.417 | 16 | 0.666 | 360 |
| A2 `(0.4,0.3,0.3)` | 11 (29.7%) | 0.345 | 13 | 0.699 | 360 |
| A2 `(0.6,0.2,0.2)` | 15 (40.5%) | 0.471 | 18 | 0.656 | 360 |
| A3 `40 hybrid + 10 dense + 10 coverage` | 13 (35.1%) | 0.434 | 17 | 0.643 | 360 |

A0 đạt parity hoàn toàn với P0. A1 cho tập chunk giống A0, nên chỉ xác nhận final rank đã tương đương sắp theo rerank trong pool hiện tại; không tạo cải thiện. A2/A3 đều tăng query coverage nhưng giảm exact-anchor containment và LCS proxy. Chỉ A0/A1 qua quality gate không giảm cả hai metric tổng thể và theo từng query. Không có cơ sở để thay baseline bằng hybrid/quota trên pilot này.

Trade-off theo query cũng cho thấy mức giảm không đồng đều: ở query 130, A0 có 1 exact anchor nhưng cả A2/A3 đều còn 0; ở query 380, A0 có 5 so với 1 ở A2/A3; query 53 giảm từ 6 xuống 5 với A2/A3. Một số query giữ nguyên số exact nhưng vẫn có thể giảm LCS, vì vậy chỉ nhìn số positives sẽ bỏ sót tổn thất chất lượng passage.

## Bảy positive bị cắt khỏi top-60

Năm ca được P0 đánh dấu exact-recoverable từ current seed là `(53,756661)`, `(130,553119)`, `(130,2968260)`, `(754,2713777)`, `(771,484565)`. Cả năm đều không được chọn ở A0, A1, ba biến thể A2 hay A3; do đó các allocation thử nghiệm chưa cứu được ca exact-recoverable nào. Hai ca còn lại (`130,546076` và `771,494605`) cũng không tạo exact recovery: hai cấu hình A2 đầu chọn cả hai ở một phần arm, nhưng LCS lần lượt chỉ là `0.869` và `0.208`, không chứa exact anchor; A2 `(0.6,0.2,0.2)` chỉ chọn `771,494605`, còn A3 không chọn hai ca này.

Các lần đổi slot có chi phí rõ ràng. A2 `(0.5,0.3,0.2)` làm mất 8 positive exact anchors đang có trong A0; `(0.4,0.3,0.3)` làm mất 10; `(0.6,0.2,0.2)` làm mất 6; A3 làm mất 8. Trong khi đó, rescued exact anchors bằng 0 ở tất cả các arm. Ví dụ query 130, A2 `(0.5,0.3,0.2)` cấp slot cho doc `546076` và đẩy ra 13 docs vốn nằm trong A0 top-60, trong đó có positive `2911632` vốn có exact context. Chi tiết document bị thay thế, historical-positive status, rank và score nằm trong `displaced_documents.json` và `allocation_changes.csv`.

## Giới hạn cứu bằng allocation

Trong bảy ca, năm exact-recoverable vẫn nằm trong pool 200 scored parents nhưng chưa arm nào đưa được vào top-60; hai ca khác không có exact-context khả dụng trong scored passage hiện tại. Theo phân tích oracle đã khóa ở P0, trần exact context là 26 anchors nếu chỉ đổi allocation trên scored seeds hiện có; 28 nếu được chọn passage khác trong cùng budget 200 cặp rerank; và 31 nếu thay đổi retrieval/fusion rồi allocation. Đây là trần oracle dựa trên labels hiện có, không phải kết quả có thể đạt bằng policy hiện tại.

## Kết luận P1

Harness và artifact replay đạt correctness: đúng 200 scored parents và 200 `relevant_docs` mỗi query, 60 chunks/query, một chunk/document, passage được chọn có cached score hợp lệ, contexts được kiểm tra nguồn, A0 khớp P0, không gọi retrieval/model. A3 điền đúng quota 40/10/10 ở cả sáu query và không cần backfill.

P1 chưa có allocation mới đạt quality gate. Giữ A0 làm baseline; không đưa A2/A3 lên cấu hình chính. Sáu query này chỉ dùng cho chẩn đoán và tạo giả thuyết; cần đánh giá policy đã chọn trên development/holdout lớn hơn trước khi kết luận generalization hoặc tối ưu tiếp trọng số.

## Artifact và tái lập

Thư mục kết quả: `outputs/v4/p1_chunk_allocation_20261021_v2/`. Gồm `arm_comparison.json`, `per_query_metrics.json`, `selected_chunks.jsonl`, `allocation_changes.csv`, `rescued_positives.json`, `displaced_documents.json`, `validation.json`, `manifest.json`, `input_hashes.json` và `scored_parent_pool.jsonl`. Manifest lưu SHA-256 của input, implementation và artifacts.

Harness: `scripts/v4/replay_chunk_allocation.py`.

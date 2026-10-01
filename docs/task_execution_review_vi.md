# Kết quả thực hiện 7 task cải thiện workflow

Ngày thực hiện: 2026-09-30. Nguồn baseline: commit `8a20eef4b8be44e20f644075f1c39494ea87d9f5`; các thay đổi hiện tại chưa được commit.

## Task 01 — Baseline

Đã sao lưu năm config vào `configs/archive/baseline_20260930/`; bốn stage có rankings, metrics, manifest và run identity riêng trong `outputs/baseline_reference/baseline_20260930/`. Replay CPU qua `python -m scripts.replay_baseline --baseline baseline_20260930` kiểm tra checksum và khớp metrics cả bốn stage.

| Stage | Recall@5 | Recall@10 | Recall@100 | MRR@100 | p95 latency ms | Zero hit @10 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 0.6919 | 0.7525 | 0.8837 | 0.5839 | 137.7 | 547 |
| Dense | 0.7661 | 0.8163 | 0.9222 | 0.6504 | 71.9 | 406 |
| Hybrid | 0.7719 | 0.8226 | 0.9249 | 0.6498 | 192.3 | 392 |
| Hybrid + reranker | 0.8864 | 0.9063 | 0.9249 | 0.8091 | 1086.5 | 207 |

Document Recall@K, các Recall@K còn lại, zero-hit ở mọi cutoff, VRAM, SHA config/corpus/rankings, model revision và depth nằm trong baseline JSON. Tokenizer revision không có trong artifacts cũ. Hash config trong `run_identity` của hybrid/rerank khác manifest và config hiện tại; cả ba giá trị được giữ lại để kiểm toán. Latency hybrid/rerank là tổng stage cache cộng stage hiện tại, không phải phép đo API trực tiếp.

## Task 02 — Evaluator và winner

Evaluator xử lý prediction thiếu như prediction rỗng, từ chối query ID thừa/duplicate và kiểm tra ID corpus/parent khi có map. Báo cáo tách Retrieval Quality và Final Selection Quality. Điểm composite được ghi là `internal_macro_f2`; `macro_f2` còn là alias tương thích cho artifacts cũ. Script report có hai lựa chọn winner, không tự sửa config.

Kết quả prediction fixed Top5 hiện tại: reranker thắng theo cả hai objective trên validation. `internal_macro_f2=0.5150`, Document F2 `0.5375`, Chunk F2 `0.4925`. Đây là metric nội bộ trên source-context labels, không phải điểm chính thức.

## Task 03 — Dynamic selection

Đã thêm `configs/vimed_dynamic_selection.yaml` với chunk bounds 0–20, document bounds 0–10. Runtime và calibration dùng chung `select_prediction`; parent documents chịu giới hạn cứng và parent có thể thay document điểm thấp hơn. Pipeline giữ diagnostics theo query; Inspector trả diagnostics ở chế độ live. Submission builder hỗ trợ cap và ghi số document cha thêm/bị loại.

Replay cấu hình dynamic trên 2.210 rankings cache cho cardinality khác nhau: chunk từ 0 đến 18, document từ 0 đến 10. Mỗi prediction và selection diagnostics được lưu trong `outputs/selection_calibration/dynamic_runtime_replay/`. Đây là replay score cache, không gọi model.

## Task 04 — Calibration offline

Đã chạy calibration từ `outputs/vimed_validation/rerank/rankings.jsonl`. Score thresholds lấy từ percentile reranker; relative deltas lấy từ phân phối khoảng cách top1. Search dùng coordinate search, nên config báo cáo là kết quả tốt nhất trong grid đã khảo sát, không khẳng định cực đại toàn cục.

| Selector | Internal Macro F2 | Avg chunks | Avg docs |
|---|---:|---:|---:|
| Fixed Top1 | 0.8170 | 1.00 | 1.00 |
| Fixed Top3 | 0.6469 | 3.00 | 3.00 |
| Fixed Top5 | 0.5150 | 5.00 | 5.00 |
| Fixed Top10 | 0.3365 | 10.00 | 10.00 |
| Best dynamic found | 0.8170 | 1.00 | 1.00 |

Best dynamic tìm được cũng trả trung bình 1 chunk và 1 document, nên validation hiện tại chưa cho thấy lợi ích F2 từ trả nhiều kết quả hơn. Cấu hình dynamic vẫn hỗ trợ cardinality thay đổi; không nên dùng kết quả tune này như bằng chứng rằng nó tốt hơn Top1. Năm fold cố định được lưu để xem độ ổn định mô tả; vì chọn cấu hình từ toàn validation, fold score không phải kiểm định độc lập.

## Task 05–06 — Error và candidate miss

Trên 2.210 query, reranker top10 có 41 miss; selection bỏ 85 positive vẫn còn trong ranking; final prediction miss cả document lẫn chunk ở 72 query. Các nhóm error có thể chồng lấp. Có 166 query không có positive trong hybrid top100. Trong positive rows đó: 10 chỉ có trong BM25 cache, 35 chỉ có trong dense cache, 121 vắng khỏi cả hai top100. Rank sâu hơn cache là unknown. 45 positive được một nhánh tìm thấy nhưng bị loại khỏi hybrid top100; không có positive nào được cả hai nhánh tìm thấy rồi bị RRF loại trong tập candidate miss này.

## Task 07 — Fusion ablation

Replay RRF k=60 top100 khớp ranking cache. Candidate Recall@100 hiện tại là 0.9249, mất 166 query/positive source labels theo giao thức một positive/query. Union có ordering theo source rank đạt 0.9253 ở top100; tăng pool lên top200 đạt 0.9452 với weighted RRF BM25 0.75/dense 0.25. Đây là candidate recall thuần; chưa đánh giá reranker/final F2 trên candidates mới và chưa đổi config winner.

## Kiểm thử và giới hạn môi trường

`python -m pytest -q --ignore=tests/integration/runtime/test_api.py --ignore=tests/integration/runtime/test_inspector.py`: **112 passed, 3 skipped**. Ba test dùng PyTorch được skip vì environment Python 3.14 hiện không có `torch`. `compileall` cho `src`, `scripts`, `tests` đạt.

API/Inspector test chưa xác nhận được: `TestClient.__enter__` treo trước khi app lifespan/route chạy; test API đơn lẻ hết timeout sau 15 giây. Đây là giới hạn của test harness/runtime hiện tại, không phải route assertion failure. Không chạy lại BGE-M3 hoặc reranker. Metric validation được tune trên cùng split và nhãn relevance chưa đầy đủ, nên cần xác nhận thêm bằng nhãn người và test split chỉ sau khi khóa cấu hình.

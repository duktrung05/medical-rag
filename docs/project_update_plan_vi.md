# Kế hoạch cập nhật Medical RAG

Ngày: 2026-09-30

## Mục tiêu

Đưa các cải tiến workflow hiện có thành một phiên bản có thể tin cậy để chọn cấu hình retrieval và final selection. Chỉ promote thay đổi khi chất lượng được xác nhận trên nhãn phù hợp, có so sánh với baseline đã freeze và có số đo latency.

## Trạng thái hiện tại

- Baseline ngày 2026-09-30 đã được lưu và replay CPU khớp metric.
- Evaluator, dynamic selector, calibration offline, error analysis và fusion ablation đã được thêm.
- Thay đổi hiện đang nằm trong working tree, chưa commit.
- Calibration hiện tại đạt `internal_macro_f2=0.8170` với Top1, nhưng relevance label mỗi query chỉ có một source context suy ra và selector đã được chọn trên cùng validation. Đây chưa phải bằng chứng đủ để đưa Top1 thành cấu hình production.
- Fusion pool top200 tăng candidate recall từ `0.9249` lên `0.9452` so với current RRF top100. Chưa đo reranked ranking hay final F2 trên pool mới.
- Artifact cache chỉ sâu đến top100 mỗi retriever; các rank sâu hơn không xác định.

## Nguyên tắc thực hiện

1. Giữ nguyên baseline frozen; mọi thử nghiệm ghi vào thư mục run riêng.
2. Không dùng test split để chọn threshold, fusion hoặc winner.
3. Ghi rõ `internal_macro_f2`; không gọi là official score.
4. Đo riêng retrieval quality, final prediction quality, cardinality và latency.
5. Không promote cấu hình chỉ dựa trên candidate recall hoặc một metric validation đã dùng để tune.

## Giai đoạn 0 — Chốt trạng thái thay đổi hiện tại

**Việc làm**

- Review diff và bảo đảm các file baseline, manifest, run identity không bị thay đổi ngoài ý muốn.
- Chạy lại bộ unit/integration phù hợp với runtime hiện tại; xử lý hoặc ghi nhận riêng vấn đề TestClient treo và ba test cần PyTorch.
- Commit thay đổi workflow thành một mốc có thể tham chiếu trước khi bắt đầu experiment tiếp theo.

**Hoàn thành khi**

- Replay baseline vẫn khớp checksum và metric.
- Test có thể chạy được có kết quả được lưu; giới hạn môi trường được ghi thành issue rõ ràng.
- Có commit SHA cho implementation task 01–07.

## Giai đoạn 1 — Cải thiện ground truth và giao thức đánh giá

**Ưu tiên cao nhất**

**Việc làm**

- Rà soát source context được dùng làm positive hiện tại; tạo tập relevance có nhãn người cho document và chunk.
- Cho phép nhiều positive document/chunk cho mỗi query và biểu diễn rõ query không có positive.
- Chia query thành calibration và holdout validation cố định trước khi tune; giữ test split đóng cho đánh giá cuối.
- Báo cáo macro và micro Precision/Recall/F2, số positive/query, coverage nhãn và confidence interval hoặc bootstrap nếu kích thước tập cho phép.

**Hoàn thành khi**

- Bộ nhãn có guideline, version, provenance và thống kê coverage.
- Macro F2 được tính trên nhiều positive với ID và parent consistency đã kiểm tra.
- Có holdout validation không tham gia chọn cấu hình.

## Giai đoạn 2 — Xác nhận fusion và candidate depth

**Việc làm**

- Dùng cache hiện tại để giữ RRF top100 làm đối chứng và đánh giá top150/top200, union và các RRF đã thử.
- Trên tập relevance đã cải thiện, đo candidate Recall@50/100/150/200 và số query được phục hồi/regressed.
- Chọn tối đa hai candidate policy để chạy reranker; ghi candidate count và latency theo query.
- Khi có GPU capacity, rerank candidates mới và so sánh reranked Recall/MRR cùng final Doc/Chunk F2 với baseline.

**Hoàn thành khi**

- Có kết quả end-to-end trên cùng query và cùng reranker cho baseline top100 và candidate policy được đề xuất.
- Có số đo latency p50/p95, memory/VRAM nếu có, và final metric trên holdout validation.
- Chỉ tăng depth nếu cải thiện final metric đủ lớn so với chi phí latency đã thống nhất.

## Giai đoạn 3 — Đánh giá dynamic selection và calibration

**Việc làm**

- Dùng cùng `select_prediction` runtime/calibration path; kiểm tra document aggregation và parent consistency.
- So sánh fixed Top1/3/5/10 với dynamic selector trên calibration split, sau đó chạy một lần trên holdout validation.
- Tune chunk và document bounds độc lập; lưu cardinality distribution, empty prediction rate, parent additions/removals, precision/recall/F2.
- Kiểm tra ổn định theo topic và nhóm query, không chỉ điểm trung bình.

**Hoàn thành khi**

- Dynamic selector tạo cardinality thay đổi theo query và có diagnostics.
- Cấu hình thắng trên calibration giữ được cải thiện trên holdout validation so với baseline selection.
- Nếu không giữ được cải thiện, giữ fixed selection hiện tại và ghi rõ dynamic selector chưa được promote.

## Giai đoạn 4 — Dùng error analysis để chọn hướng cải thiện

**Việc làm**

- Theo từng nhóm lỗi `CANDIDATE_MISS`, `FUSION_DROPPED`, `RERANKER_MISS`, `SELECTION_MISS`, `DOC_HIT_CHUNK_MISS`, lấy mẫu query để kiểm tra nhãn và nội dung.
- Tách unknown do cache depth khỏi miss đã xác nhận.
- Ưu tiên một thay đổi cho từng bottleneck có bằng chứng lớn nhất; chạy ablation riêng và so sánh với baseline.

**Hoàn thành khi**

- Mỗi experiment có giả thuyết, tập query ảnh hưởng, metric chính, latency và quyết định giữ/bỏ.
- Không gộp thay đổi retriever, fusion, reranker và selector trong cùng một ablation.

## Giai đoạn 5 — Promotion và đánh giá cuối

**Việc làm**

- Chọn winner retrieval và winner final submission độc lập theo objective đã khóa trước.
- Freeze config, code SHA, corpus SHA, model/tokenizer revision, dataset/split và mọi depth/threshold.
- Chạy test split đúng một lần sau khi khóa cấu hình; không quay lại tune theo kết quả test.
- Xuất report so với frozen baseline và lưu artifacts trong thư mục run/version riêng.

**Hoàn thành khi**

- Có kết quả final trên test, có baseline comparison và provenance đầy đủ.
- Config được đề xuất tái lập được từ manifest; mọi thay đổi có thể truy ngược đến experiment.

## Thứ tự ưu tiên đề xuất

| Ưu tiên | Hạng mục | Lý do |
|---|---|---|
| P0 | Chốt commit và xác nhận baseline replay | Tạo điểm bắt đầu ổn định cho experiment |
| P1 | Bổ sung/kiểm tra nhãn relevance nhiều positive | Metric hiện tại bị giới hạn bởi source-context label suy ra |
| P2 | Rerank top150/top200 trên cùng query | Candidate recall tốt hơn chưa chứng minh final quality tốt hơn |
| P3 | Đánh giá selector trên holdout validation | Top1 hiện cao nhưng có rủi ro optimistic do tune trên cùng split |
| P4 | Chọn bottleneck bằng error analysis rồi ablate riêng | Tránh thay đổi nhiều tầng mà không biết nguyên nhân |
| P5 | Freeze winner và chạy test cuối | Chỉ dùng test sau khi toàn bộ quyết định đã khóa |

## Kết quả cần có sau mỗi experiment

- `experiment_id`, commit SHA, config/corpus/model revisions.
- Split và query IDs đã dùng; xác nhận không tune trên test.
- Retrieval metrics, final Doc/Chunk metrics, `internal_macro_f2`, zero-hit và candidate coverage.
- Cardinality trung bình/phân vị, latency p50/p95, VRAM nếu có.
- So sánh với frozen baseline, lỗi được phục hồi/regressed, quyết định promote hoặc reject.


# P3/P4: independent relevance và passage-first challenger (22-10-2026)

## Trạng thái và quyết định

Giữ P2-A0 làm baseline. Chưa promote P2-A1/A2, chưa chạy P2-A3 hoặc tuning trọng số. Kết quả P2 cho thấy anchor containment không đồng nhất với việc passage trả lời đúng câu hỏi; packet 406 items hiện **chưa được chấm**. Vì vậy chưa có kết luận độc lập về evidence precision, within-parent passage accuracy hay reranker separation.

Đã tạo hai luồng thực nghiệm tách biệt: (1) khóa nhãn blinded trước khi mở mapping, (2) B0 passage-first sử dụng passage BM25 và dense index hiện có, fuse theo passage identity rồi rerank tối đa 200 pairs. B0 giữ tối đa một exported chunk/document và 60 chunks/query, document IDs dẫn xuất từ passage đã score. A0 production không thay đổi.

## P3: nhãn độc lập

Packet gốc: `outputs/v4/p2_relevance_review_packet_20261021_v2/`. Người đánh giá chỉ xem `review_items.jsonl`, `rubric.json` và bản sao `annotation_template.csv`; không xem `item_map_for_analysis.json`, A0/A1/A2, scores, ranks hoặc historical anchors. Chấm rubric 0/1/2, kiểm tra riêng ngữ cảnh bệnh nhân, số liệu, mâu thuẫn và grounding. Một reviewer chấm toàn bộ 406 items; reviewer thứ hai chấm độc lập 102 items đã chọn trước bằng SHA-256 trong `outputs/v4/p3_secondary_review_assignment_20261022/`. Lưu hai CSV riêng, giải quyết bất đồng trong CSV adjudicated, sau đó khóa file bằng SHA-256 **trước khi** chạy analysis.

```bash
python3 scripts/v4/evaluate_p3_blinded_relevance.py lock \
  --packet outputs/v4/p2_relevance_review_packet_20261021_v2 \
  --primary /path/to/reviewer_1.csv \
  --secondary /path/to/reviewer_2.csv \
  --secondary-assignment outputs/v4/p3_secondary_review_assignment_20261022/assignment.json \
  --adjudicated /path/to/adjudicated.csv \
  --output outputs/v4/p3_relevance_locked

python3 scripts/v4/evaluate_p3_blinded_relevance.py analyze \
  --lock-dir outputs/v4/p3_relevance_locked \
  --replay outputs/v4/p2_passage_candidate_replay_20261021
```

`lock` kiểm tra 406 item IDs, đầy đủ nhãn reviewer 1 và adjudicated, đúng 102 IDs reviewer 2 được phân công trước, rubric hợp lệ và hashes packet; không mở mapping, kể cả để hash. Reviewer 2 dùng `secondary_review_items.jsonl` và bản sao `secondary_annotation_template.csv`. `analyze` xác minh toàn bộ hashes trước khi đọc mapping, rồi tính evidence precision@60 theo arm/query, agreement và Cohen kappa của phần double review, phân bố rerank score theo relevance 0/1/2, AUC score relevance 2 so với 0/1, và within-parent accuracy **chỉ trên các passage đã được review**. Chưa được suy ra recall toàn corpus từ packet chọn bởi A0/A1/A2; muốn đo recall cần pool relevant passages độc lập.

Đã khóa **query IDs**, chưa khóa nhãn, thành 60 development + 60 holdout trong `outputs/v4/p3_query_split_20261021/`. Chọn ngẫu nhiên xác định bằng SHA-256 trong ba nhóm độ dài query, loại toàn bộ sáu pilot, không đọc historical anchors. Holdout chỉ dùng một lần sau khi cố định policy trên development. Hai tập này chưa có relevance labels và chưa phải benchmark chất lượng.

## P4: B0 passage-first

`src/medrag/core/passage_first.py` tạo B0 riêng: passage BM25 + dense top-200 mỗi nhánh, RRF theo `(doc_id, passage_id)` không parent aggregation trước rerank, chọn tối đa 200 passage pairs, score một lần mỗi pair, nhóm về parent sau score, xuất tối đa 60 source-grounded chunks. `shared_pair_plan` lập kế hoạch ngân sách chung tối đa 200 unique pairs/query khi A0 và B0 cùng chạy; arm B0 standalone có budget 200 riêng và **không được** gọi là dual-path 200 total.

CPU replay chỉ dùng sáu P0 branch traces, không retrieval/model mới. Mỗi query có 200 B0 pairs; chỉ **134–160 unique parents**, thay vì 200 parents của A0. Pair overlap A0/B0 là **131–158/query**; **42–69 B0 pairs/query** chưa có score A0, tổng 349/1.200 pairs. Toàn bộ 1.200 B0 pairs từ replay đã được kiểm tra passage ID, parent ID và source offsets trên registered full index. Đây là kiểm tra candidate survival/cost, **không phải** evidence quality hoặc complete rerank replay. Artifact: `outputs/v4/p4_b0_candidate_replay_20261021_v2/`.

Với 200 passage pairs, B0 có thể chỉ tạo 134–160 document IDs trong sáu pilot; điều này có nguy cơ giảm document retrieval so với A0 có 200 scored parents. B0 standalone chưa đáp ứng yêu cầu giữ document quality. Cần đánh giá document ranking độc lập và thử dual-path có **ngân sách inference chung** trước khi cân nhắc thay thế A0.

`scripts/v4/run_p4_b0_full_index.py` là runner B0 standalone cho frozen development/holdout IDs, dùng full index và reranker đã pin revision; sau mỗi query kiểm `validate_selected`, 200 pairs, <=60 chunks, source/context grounding, và ghi latency thực. Chưa chạy **rerank end-to-end**: GPU driver không truy cập được; bộ PyTorch/Transformers/NumPy có thể nạp từ local `outputs/pubmed_runtime_deps_20261012/`, nhưng rerank 200 pairs/query trên CPU chưa phải phép đo nên thực hiện trước khi có relevance labels. Không lấy latency A0 của 20-query full-index smoke làm latency B0.

Đã chạy một probe retrieval-only trực tiếp trên full index 2.951.224 passages bằng BGE-M3 CPU và passage BM25 cho ba development queries chưa gán nhãn. Các query IDs `39`, `74`, `119` có BM25 lần lượt `13,37`, `23,05`, `48,04` giây; dense search `7,52`, `6,95`, `6,87` giây. Mỗi query có 200 B0 candidate pairs được source-validated. Artifact: `outputs/v4/p4_b0_full_index_retrieval_probe_20261022/results.json`. Ba query quá ít để suy ra p95 sản xuất; số đo không gồm rerank, context expansion hoặc output serialization. B0 **end-to-end latency**, document quality và independent relevance vẫn chưa đo.

Probe có thể chạy lại bằng `scripts/v4/probe_p4_b0_full_index_retrieval.py` với `HF_HOME=.cache/huggingface TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONPATH=outputs/pubmed_runtime_deps_20261012 OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 .venv-phase1/bin/python` và `--output` trỏ đến thư mục mới. CPU/GPU, cache và tải hệ thống có thể làm thay đổi thời gian chạy.

Khi có reviewer labels và runtime model, so sánh A0/B0 trên development theo relevance mức 2, loss/gain theo từng query, parent/document quality và chi phí full index. Chỉ mở holdout sau khi khóa architecture/policy. Official CHUNKS_F2 vẫn cần evaluator chính thức. Không dùng oracle anchors 21→28/37 làm tiêu chí chọn architecture.

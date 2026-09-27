# Review và kế hoạch cải thiện medical-rag

Ngày review: 2026-09-27. Phạm vi: code trong workspace hiện tại, bộ test, cấu hình và artifacts validation. Không sửa code, không chạy lại toàn bộ benchmark GPU, không xác minh Docker hoặc UI trực quan trong lần này. Workspace có sẵn nhiều thay đổi staged/unstaged; kết quả áp dụng cho trạng thái hiện tại.

## Kết quả kiểm tra

- `.venv/Scripts/python.exe -m pytest -q`: **120 passed, 1 failed**, 1 warning, 28,78 giây.
- Test fail: `tests/test_workflow_integrity.py::test_readme_local_links_exist`; README trỏ tới `docs/vimed_validation_benchmark_vi.md` không tồn tại.
- Ruff hiện hành báo 317 findings; phần lớn là style, imports và typing. Kiểm tra riêng `--select F` có 7 findings: 5 F821 ở annotation `torch` trong `src/retrieval/dense_encoding.py`, 2 imports không dùng trong tests. File dùng postponed annotations nên F821 này không chứng minh inference đang lỗi; cần TYPE_CHECKING import hoặc khai báo type phù hợp.
- Tính lại Recall từ rankings cho đủ 2.210 câu ở cả bốn stages: khớp các số trong báo cáo hiện có. Đây là kiểm tra artifacts, không phải inference mới.

| Stage | Recall@5 | Recall@10 | Recall@100 |
|---|---:|---:|---:|
| BM25 | 69,19% | 75,25% | 88,37% |
| Dense BGE-M3 | 76,61% | 81,63% | 92,22% |
| Hybrid RRF | 77,19% | 82,26% | 92,49% |
| Hybrid + reranker | 88,64% | 90,63% | 92,49% |

## Nhận định và phát hiện

Hệ thống hiện là retrieval và selection, có UI đánh giá; chưa có luồng sinh câu trả lời bằng LLM. Kiến trúc đã tách data/retrieval/reranking/evaluation, có model revision cố định và kiểm tra checksum. Đầu tư tiếp nên dựa trên lỗi retrieval và chất lượng đánh giá.

1. **P0 — Đưa regression gate về trạng thái xanh.** Sửa link README hoặc khôi phục báo cáo đã được xác minh; xem lại docs/workflow_review_vi.md vì ghi 121 pass trong khi trạng thái hiện tại có test fail. Không coi con số test lịch sử là kết quả hiện tại.
2. **P1 — Độ tin cậy của nhãn đánh giá.** Qrels chỉ có context nguồn suy ra từ QA; context khác chưa được gán nhãn vẫn có thể phù hợp. Recall hiện tại là khả năng tìm context nguồn, không đủ để kết luận chất lượng trả lời y khoa. Corpus chứa context train/validation/test theo giao thức closed corpus; điều này không tự động là leakage, nhưng cần audit câu hỏi trùng/near-duplicate và không tune bằng test labels.
3. **P1 — Recall của candidate pool.** Hybrid top 100 còn thiếu positive cho 166/2.210 câu; reranker chỉ đổi thứ tự nên không thể sửa nhóm này. Có 251 câu miss ở top 5, trong đó 166 miss toàn pool và 85 có positive trong pool nhưng ngoài top 5. Nhóm topic `2` có rerank Recall@5 khoảng 80,90%, thấp hơn tổng thể; cần tra metadata trước khi đặt tên chủ đề.
4. **P1 — Calibration chưa tương đương runtime.** `src/selection/calibrate.py` tạo prediction từ selection trực tiếp, còn `src/pipeline.py` gọi `enforce_parent_consistency`. Khi document được thêm sau selection, số lượng document và F2 có thể khác calibration. Config hiện cố định chunk/doc min=max=5; sau bổ sung document cha, relevant_docs có thể vượt 5. Phải xác định giới hạn đầu ra và metric mục tiêu trước khi tối ưu.
5. **P2 — Resume chưa chịu được dòng JSONL ghi dở.** `scripts/benchmark_vimed_validation.py:read_rows` parse mọi dòng; crash giữa lần ghi có thể làm resume thất bại. Nên phục hồi riêng dòng cuối không hoàn chỉnh, giữ backup và từ chối lỗi ở giữa file. Viết manifest/metrics qua file tạm rồi atomic replace.
6. **P2 — Hiệu năng và concurrent serving.** API giữ lock quanh toàn query; cách này bảo vệ trạng thái dùng chung nhưng serialize requests. `last_candidate_scores` là mutable state của pipeline. Nên trả diagnostics trong kết quả riêng cho từng query trước khi mở rộng concurrency. P95 rerank trong artifacts khoảng 1.086 ms được cộng từ các stage cache; chưa phải đo end-to-end live dưới tải.
7. **P2 — Chất lượng code và CI.** Lint chưa sạch; cần cấu hình lint được lưu trong repo và áp dụng dần. Ưu tiên findings correctness/import, không đổi hàng loạt chỉ để giảm số style findings. Chưa thấy workflow CI trong danh sách file đã kiểm tra.

## Skill nên tập trung

Ở đây skill được hiểu là năng lực kỹ thuật cần phát triển cho dự án.

| Ưu tiên | Skill | Bài tập trực tiếp trên dự án | Đầu ra |
|---|---|---|---|
| 1 | Information Retrieval evaluation và thiết kế thí nghiệm | Phân loại lỗi, relevance judging, leakage audit, paired bootstrap, tập tune và tập kiểm chứng độc lập | Bộ đánh giá đáng tin và quyết định dựa trên số liệu |
| 2 | Data engineering / xử lý văn bản tiếng Việt | Đo truncation 512 tokens, phân tích chunk dài, query không dấu/viết tắt, quan hệ doc–chunk | Corpus và query processing được version hóa |
| 3 | Retrieval tuning | Sweep candidate depth, RRF k, sparse/dense depth; đo union coverage trước và sau fusion | Pareto quality/latency và config được chọn |
| 4 | Python testing và backend engineering | Fault injection cho resume, concurrency test, thống nhất calibration với pipeline | Test hồi quy thực tế và runtime đáng tin |
| 5 | ML serving / MLOps | Đo cold/warm latency, p50/p95, throughput, VRAM; test container | Kịch bản deploy tái lập và báo cáo tải |

Phân bổ thời gian đề xuất: 40% evaluation, 25% data/retrieval, 20% testing/backend, 15% serving. Đây là ưu tiên công việc theo bằng chứng của dự án, không phải đánh giá trình độ cá nhân của người phát triển.

## Plan 4 tuần

### Tuần 1 — Baseline và nhãn đánh giá

- Sửa test fail và annotation/import findings; thống nhất lệnh lint, test trong CI.
- Lưu baseline với config/corpus/model/code identity, metric mục tiêu và phần cứng.
- Review toàn bộ 166 miss trong pool; lấy thêm mẫu từ 85 lỗi xếp hạng và nhóm đúng làm đối chứng.
- Gán nhãn khoảng 200–300 câu, với rubric relevance 0/1/2 và subset được hai người chấm để kiểm tra độ nhất quán.
- Phân biệt benchmark context nguồn với benchmark relevance; quyết định tối ưu Recall@5, Recall@10 hay F2 theo mục tiêu thật.
- Tiêu chí hoàn tất: test hiện tại đều pass, lỗi có phân loại, tập tune/holdout được khóa trước tuning; không dùng test labels để chọn config.

### Tuần 2 — Tìm đủ ứng viên

- Thử depth 100/200/300 và RRF k 20/60/100; thay một nhóm yếu tố mỗi lần.
- Tách hai loại miss: không có positive trong union sparse+dense và có trong union nhưng bị fusion cắt bỏ.
- Đo tỷ lệ context vượt 512 tokens. Chỉ thử chunk nhỏ/overlap nếu lỗi truncation có bằng chứng; giữ mapping từ context nguồn sang subchunks và cập nhật qrels/checksum tương ứng.
- Thử normalization tiếng Việt/viết tắt trên nhóm lỗi cụ thể, có ablation đối chứng.
- Mục tiêu thử nghiệm: nâng candidate Recall@100 từ 92,49% lên khoảng 94% hoặc hơn; đây là mục tiêu, không cam kết. Với pool lớn hơn cần báo Recall@200/300 riêng, không gọi là cải thiện @100.
- Tiêu chí hoàn tất: bảng quality/latency theo topic, kiểm chứng cải thiện trên holdout, config và artifacts mới không ghi đè baseline.

### Tuần 3 — Xếp hạng và selection

- Đánh giá rerank depth 30/50/100/200 và max_length theo tài nguyên thực tế.
- Giữ candidate pool cố định để tách tác dụng reranker khỏi tác dụng retrieval.
- Dùng cùng luồng selection, document aggregation và parent consistency trong calibration và runtime; thêm test cho trường hợp document cha được bổ sung.
- So sánh fixed k=3/5/10 với threshold theo raw logits trên tập tune. Không coi raw logit là xác suất; nếu cần confidence phải có calibration và nhãn đủ tốt.
- Mục tiêu thử nghiệm: Recall@5 khoảng 90% hoặc giữ chất lượng gần baseline với latency thấp hơn; báo cả Recall/Precision/F2 theo độ đầy đủ nhãn.
- Tiêu chí hoàn tất: policy được chọn theo metric đã khóa và có kết quả holdout, không chọn chỉ vì điểm trên tập tune cao.

### Tuần 4 — Độ ổn định và triển khai

- Viết tests crash/resume: dòng cuối ghi dở, corruption giữa file, checksum/config thay đổi.
- Tách diagnostics khỏi trạng thái pipeline; kiểm thử request đồng thời trước khi nới lock.
- Đo live end-to-end với concurrency 1/2/4, cold/warm, p50/p95, throughput và VRAM; chọn mức tải phù hợp GPU.
- Kiểm tra Docker build/start/health/search với data/index/cache được mount, thêm readiness và log thời gian từng stage.
- Tiêu chí hoàn tất: resume an toàn, không trộn diagnostics giữa request, có bằng chứng container chạy thật và báo cáo tải.

## Backlog sau plan

Fine-tune bằng hard negatives chỉ sau khi có nhãn và holdout đủ tin cậy. Thêm answer generation/citations/abstention nếu mục tiêu chuyển sang RAG trả lời; khi đó phải đánh giá groundedness và chất lượng câu trả lời riêng. Chưa ưu tiên agent orchestration hay đổi model liên tục vì chưa có bằng chứng chúng giải quyết nút thắt hiện tại.

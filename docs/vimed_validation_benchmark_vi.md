# Benchmark retrieval ViMed — task 2

Đã đo đủ **2210 câu validation**; 5 câu thiếu context bị loại. Corpus cố định 17.955 chunks; không chạy lại test split.

| Cấu hình | Recall@1 | Recall@5 | Recall@10 | Recall@100 | MRR@100 |
|---|---:|---:|---:|---:|---:|
| bm25 | 49.05% | 69.19% | 75.25% | 88.37% | 0.5839 |
| dense | 55.66% | 76.61% | 81.63% | 92.22% | 0.6504 |
| hybrid | 54.80% | 77.19% | 82.26% | 92.49% | 0.6498 |
| rerank | 74.93% | 88.64% | 90.63% | 92.49% | 0.8091 |

## Cấu hình được chọn: rerank

Tiêu chí đã đặt trước: Recall@10, sau đó MRR@100, sau đó độ trễ. Cấu hình đóng băng ở `configs/vimed_selected_validation.yaml`.

## Giao thức và giới hạn

BM25 và dense lấy 100 candidates mỗi nhánh; RRF k=60 giữ 100; reranker xếp lại đủ 100. BGE-M3 dùng CLS, FP16, max_length=512; reranker BGE v2 M3 dùng raw logits, FP16, batch=8, max_length=512. Revision của cả hai model được khóa trong config.

Ground truth là context nguồn của từng câu QA, chưa phải nhãn relevance đầy đủ. Context khác có thể đúng nhưng chưa được gán nhãn. Corpus gồm context từ các split, phù hợp đánh giá retrieval trong corpus đóng; không thể suy ra chất lượng tổng quát ngoài corpus. Không dùng câu hỏi/đáp án làm nội dung index.

Latency BM25/dense là thời gian pipeline sau warmup; hai lượt chạy đồng thời nên chịu tranh chấp CPU. Latency hybrid/rerank là tổng các lượt đã cache cộng lượt hiện tại, không phải latency API đo trực tiếp. Recall và MRR dùng thứ hạng thực tế, độc lập với cách đo latency.

Union BM25+dense có Recall candidate 94.52%; sau RRF top100 còn 92.49%. Reranker giữ nguyên tập candidates nên Recall@100 bằng hybrid.

## Kiểm tra

112 tests pass; 1 cảnh báo deprecation Starlette/AnyIO. Mỗi lượt đủ số câu, checksum cache hợp lệ, không duplicate/unknown ID hoặc score vô hạn; metrics được tính lại từ rankings; reranker giữ nguyên candidates. Chi tiết mỗi topic, latency, VRAM và danh sách lỗi nằm trong `outputs/vimed_validation/`.

## T?i nguy?n v? th?i gian

RTX 4050 Laptop 6 GB; RAM 16 GB, c?n kho?ng 5,5 GB tr??c l??t reranker; disk c?n kho?ng 178 GB. Dense d?ng peak PyTorch allocated kho?ng 1,07 GiB; reranker 1,15 GiB (reserved 1,23 GiB). ??y l? b? nh? PyTorch, kh?ng g?m desktop v? driver. Reranker ch?y 2.210 c?u trong 1.729 gi?y (~28,8 ph?t), sau warmup. Th?i gian t?i model kh?ng t?nh v?o benchmark.

207 c?u ch?a t?m ???c context ngu?n trong top10: 166 c?u kh?ng c? context ngu?n trong candidates top100, 41 c?u c? nh?ng b? x?p sau top10. B??c c?i thi?n ti?p theo n?n th? t?ng pool candidates v? ki?m tra l?i topic 2 (Recall@10 82,55%) tr?n validation.

## Ch?y l?i

D?ng m?i tr??ng `.venv` c? CUDA; ??t `HF_HOME` tr? t?i `.cache/huggingface`. Model ?? ???c cache v? kh?a revision; c? th? ??t `HF_HUB_OFFLINE=1`.

```powershell
.venv/Scripts/python.exe -m scripts.prepare_vimed_validation
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage bm25 --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage dense --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage hybrid --timeout-seconds 600
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage rerank --timeout-seconds 7200
.venv/Scripts/python.exe -m scripts.report_vimed_validation
.venv/Scripts/python.exe -m pytest -q
```

L??t t??ng th?ch ?? ho?n th?nh s? ???c d?ng l?i. N?u ??i config ho?c d? li?u, d?ng th? m?c `--output` m?i; ch??ng tr?nh t? ch?i tr?n cache kh?c fingerprint. `--limit` ch? d?nh cho probe v? b?t bu?c c? th? m?c output ri?ng. Windows Application Control ch?n DLL sklearn trong sandbox c?a Codex; l??t model th?t ?? ch?y ngo?i sandbox v?i quy?n ???c c?p.

So v?i BM25 ? top10: kh?i ph?c 349 c?u, gi?m ch?t l??ng 9 c?u, c?ng b? s?t 198 c?u.


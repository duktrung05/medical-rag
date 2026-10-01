# Review luồng và chạy ViMedQA — 2026-09-26

## Trạng thái trước khi tích hợp ViMed

- Đã có schema, loader, validator, BM25 lưu index, dense exact/FAISS, hybrid RRF,
  BGE reranker, scoring document, selection threshold, calibration, evaluator và submission.
- Luồng thực thi: `QueryRecord → adapt_query/normalize → retriever → reranker tùy chọn
  → chunk selection → document aggregation/selection → parent consistency`.
- `src/service.py` tạo pipeline chung cho CLI/API. `src/api.py` trả ID kết quả;
  chưa hiển thị text, score hoặc ground truth để kiểm tra retrieval.
- Query expansion có code nhưng chưa được nối vào luồng `adapt_query → pipeline`.
- Tantivy vẫn báo `NotImplementedError`; BM25 đang dùng implementation nội bộ.
- Có kết quả BM25 trên MedQuAD dev 2.333 câu và hybrid smoke 32 câu/96 chunks.
  Smoke không chứng minh chất lượng trên toàn bộ dữ liệu hay tiếng Việt.
- README cũ nói các backend chưa triển khai đầy đủ, không phản ánh code hiện tại.
- ViMedQA có parquet local nhưng chưa có converter/index/evaluation/UI riêng.

## Đã bổ sung

1. `scripts/preparevimed.py`: đọc config `all` để tránh gộp trùng các thư mục topic;
   lấy context từ train, validation, test; loại trùng bằng context NFC; gom bài theo URL.
2. Chỉ index context; không index câu hỏi, answer hoặc keyword. Title được giữ để
   hiển thị, không đưa vào BM25 text. Một context là một chunk; chưa tái chunking.
3. Câu hỏi test là query; context nguồn là positive suy ra. Corpus có test context
   vì mục tiêu là tìm tài liệu trong kho đóng, không huấn luyện mô hình trên test.
4. `configs/vimed_bm25.yaml`: BM25 word, 100 candidates, pipeline chọn 5 kết quả.
5. `scripts/evaluate_vimed.py`: Recall@1/3/5/10/20/50/100, MRR@100, latency,
   breakdown theo topic và rankings cho từng câu; không chỉnh tham số trên test.
6. `src/inspector.py` và `src/web/inspector.html`: UI dùng pipeline hiện tại;
   hiển thị candidates, điểm, kết quả được chọn, nguồn và context/answer tham chiếu.
   Sửa nội dung câu hỏi hoặc nhập câu mới thì không còn áp nhãn dataset.

## Cách chạy

Từ thư mục gốc, dùng Python có pandas, pyarrow, pydantic, yaml, fastapi, uvicorn:

```powershell
python -m scripts.prepare_vimed
python -m scripts.evaluate_vimed
python -m uvicorn src.inspector:create_app --factory --host 127.0.0.1 --port 8000
```

Mở <http://127.0.0.1:8000>. Nếu mở UI trước khi benchmark hoàn tất, reload sau
khi có `outputs/vimed_bm25/metrics.json` để hiển thị metrics và lọc câu trượt.

Dữ liệu sau chuyển đổi: 17.955 context, 1.935 bài, 2.213 câu test có context.
4/2.217 câu test thiếu context được loại khỏi đánh giá; manifest ghi số bỏ qua.
File rankings giữ top 100 để phân tích lỗi, metrics giữ tổng hợp.

Kết quả BM25 word trên 2.213 câu: Recall@1 = 50,07%, Recall@5 = 70,85%,
Recall@10 = 77,45%, Recall@100 = 89,15%, MRR@100 = 0,5943.
Thời gian retrieval trung bình = 66,61 ms trong lần chạy local này.
Recall@10 theo topic: body-part 85,43%, disease 81,86%, drug 69,33%,
medicine 75,36%. Nhóm drug nên được ưu tiên phân tích lỗi.

## Giới hạn khi đọc điểm

Đây là benchmark retrieval suy ra từ QA, chưa phải qrels được đánh giá đầy đủ.
Context khác vẫn có thể trả lời đúng nhưng không nằm trong positive nguồn.
Recall đo khả năng tìm đúng context nguồn; score BM25 không phải xác suất.
ViMedQA hiện chỉ kiểm tra tiếng Việt, chưa đánh giá retrieval đa ngôn ngữ.

## Bước tiếp theo

1. Xem câu trượt top 10 trong UI, phân loại: từ đồng nghĩa, câu hỏi diễn đạt khác,
   context quá ngắn/dài, nhiều context cùng bài, nhãn QA chưa đầy đủ.
2. Tạo query/ground truth validation bằng cùng mapping context; dùng validation
   để chọn tokenizer, dense model, fusion, reranker và threshold.
3. Build dense index trên đúng corpus này; đo dense và hybrid RRF trên validation.
   Chỉ thêm reranker khi đã biết candidate recall đủ tốt.
4. Khóa cấu hình rồi đánh giá test cho các lần so sánh được ghi nhận rõ ràng;
   không dùng các lỗi test để chỉnh tham số rồi báo như test độc lập.
5. Bổ sung relevance review nhiều positive và benchmark vi/en/zh để kết luận
   về khả năng retrieval y khoa đa ngôn ngữ.

## Kiểm tra vận hành

Đã chạy converter, build index, benchmark và kiểm tra HTTP health, HTML,
search với câu test. Endpoint trả 10 candidates và tìm context nguồn ở hạng 4
cho câu đầu tiên. Browser automation không có browser khả dụng trong phiên,
nên chưa xác nhận layout và click UI bằng trình duyệt thật.

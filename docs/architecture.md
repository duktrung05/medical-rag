# Bộ khung dự án

## Luồng hiện tại

`JSONL corpus → kiểm tra dữ liệu → DemoRetriever → RetrievalPipeline → PredictionRecord`

Hai cách gọi dùng chung `src/service.py` để khởi tạo pipeline:

- Batch: `python -m scripts.retrieve` đọc queries JSONL và ghi predictions JSONL.
- HTTP: `src.api:create_app` nạp corpus một lần lúc khởi động, nhận query qua `/search`.

`configs/demo.yaml` là cấu hình runtime có kiểm tra kiểu và từ chối khóa lạ.
Đường dẫn corpus tính từ thư mục chứa YAML. Đường dẫn CLI tính từ thư mục chạy lệnh.
API mặc định đọc `configs/demo.yaml`; đổi qua biến môi trường `R2AI_CONFIG`.

## Phạm vi của bản khung

DemoRetriever chỉ so khớp token, dùng để kiểm tra nối các thành phần. Nó không có
khả năng hiểu y khoa hoặc đối chiếu ngữ nghĩa giữa các ngôn ngữ. Các file mẫu hiện
có dùng cho kiểm thử phần mềm, không phải dữ liệu huấn luyện hay benchmark chính thức.
Điểm demo không được dùng để kết luận chất lượng thi đấu.

Các cấu hình `baseline_*.yaml`, BM25, dense, indexing và reranker có sẵn là những
điểm mở rộng đang phát triển; chưa được nối vào runtime mới. Cấu hình demo cố ý
không chấp nhận các backend đó để tránh chạy nhầm thành hệ thống đã hoàn thiện.

## Bổ sung sau khi có dữ liệu

1. Viết adapter đưa dữ liệu BTC về ChunkRecord và QueryRecord; giữ nguyên ID.
2. Triển khai backend theo `BaseRetriever.search(query, top_k)` và đăng ký tại
   `build_pipeline`; mở rộng schema cấu hình cùng lúc.
3. Nối reranker qua `BaseReranker`; cấu hình riêng ngưỡng chọn cho mỗi backend.
4. Xác nhận định dạng submission, quy tắc cha-con và cách tính điểm theo đề chính
   thức. Các quy tắc trong repository hiện là quy ước nội bộ cần đối chiếu.
5. Bổ sung tập đánh giá, ghi nhận thực nghiệm và chỉ thêm fine-tuning khi cần.

Không có bước training, tải mô hình hoặc gọi dịch vụ ngoài trong chế độ demo.

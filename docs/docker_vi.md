# Chạy ViMed Retrieval Lab bằng Docker

## Yêu cầu

- Docker Desktop chạy Linux containers, backend WSL2 trên Windows và driver NVIDIA hỗ trợ GPU trong WSL2. Linux cần NVIDIA Container Toolkit. Xem [GPU trên Docker Desktop Windows](https://docs.docker.com/desktop/features/gpu/) và [GPU trong Compose](https://docs.docker.com/compose/how-tos/gpu-support/).
- Chạy lệnh tại thư mục gốc repo. Máy hiện tại dùng RTX 4050 6 GB; UI chỉ chạy một worker để tránh nhân bản model trên GPU.
- Giữ dữ liệu/index/model và benchmark đã tạo ở các thư mục bên dưới. Image chỉ chứa code và dependencies; các dữ liệu lớn được mount từ máy.

```text
data/vimed/chunks.jsonl
data/vimed/validation/{samples.jsonl,manifest.json}
artifacts/vimed_bm25/
artifacts/vimed_dense_bge_m3_cls/
outputs/vimed_validation/{bm25,dense,hybrid,rerank}/
.cache/huggingface/hub/
```

Hai model BGE-M3 và BGE-reranker-v2-m3 đã được cache trong workspace hiện tại. Docker dùng lại cache này, mặc định offline. Không cần mount `.venv` Windows vào container Linux.

## Chạy nhanh

Mở Docker Desktop, chờ engine sẵn sàng:

```powershell
docker version
docker compose up -d --build
docker compose logs -f ui
```

Mở **http://localhost:8001**. Lần build đầu tải CUDA Torch và dependencies nên có thể lâu; model được nạp khi truy vấn live lần đầu. Benchmark đã lưu có thể xem ngay sau startup.

Nếu port 8001 đang được server Python cũ sử dụng, chọn port khác:

```powershell
$env:UI_PORT = '8002'
docker compose up -d --build
```

Khi đó mở http://localhost:8002. Health endpoint: `/health`.

## Kiểm tra GPU trong container

```powershell
docker compose exec ui python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

Lệnh phải trả về `CUDA: True`. Cấu hình dense/reranker hiện dùng FP16 và CUDA, không tự chuyển sang CPU. Dockerfile dùng [image PyTorch chính thức](https://hub.docker.com/r/pytorch/pytorch/tags?name=2.7.1-cuda12.8-cudnn9-runtime), khóa digest, chứa sẵn Torch 2.7.1 + CUDA 12.8. Không còn tải Torch wheel bằng pip trong build; `docker/constraints.txt` ngăn dependencies thay phiên bản Torch. Image hỗ trợ Linux AMD64; Compose đã chọn platform này.

## Khi build báo lỗi pip cài Torch

Dockerfile mới đã bỏ bước `pip install torch`.

Lỗi đã ghi nhận: cuDNN wheel 726,9 MB bị tải dở ở khoảng 515,6 MB, dẫn đến SHA256 mismatch. Đây là lỗi ở bước tải dependency, trước khi code dự án chạy. Base image PyTorch chứa sẵn Torch/cuDNN và được kiểm tra theo digest, nên không còn bước tải wheel đó trong pip.

Pip được cấu hình timeout 120 giây, retry kết nối 10 lần và resume download 10 lần qua `PIP_RESUME_RETRIES` (xem [tài liệu pip](https://pip.pypa.io/en/stable/cli/pip/#cmdoption-resume-retries)). Nếu download bị ngắt, pip có thể tải tiếp phần còn thiếu; kiểm tra hash vẫn được giữ nguyên.

Chạy lại với log đầy đủ:

```powershell
docker compose --progress plain build ui
docker compose up -d
```

Không cần xóa model cache hay prune Docker. Lần đầu cần tải base image khoảng 4,2 GB compressed; Docker tải và kiểm tra các layers. Nếu còn lỗi, giữ phần log `ERROR` phía trên `exit code: 1` để phân biệt download timeout, DNS/TLS, dependency hoặc hết dung lượng.

## Lưu dữ liệu, cập nhật và dừng

Nhãn relevance được lưu ở `outputs/vimed_validation/human_labels.jsonl` trên máy. Dừng hoặc thay container không làm mất file này. Corpus/index được mount read-only. Cache model và outputs được giữ trên máy.

```powershell
docker compose up -d --build   # rebuild sau khi sửa code
docker compose stop          # dừng
docker compose start         # chạy lại
docker compose down          # gỡ container, giữ các thư mục bind mount
```

Nếu mang sang máy khác, copy cả các thư mục dữ liệu liệt kê ở trên. Nếu thiếu model cache, bật tải model (cần mạng):

```powershell
$env:HF_HUB_OFFLINE = '0'
docker compose up -d
```

Compose không tự chuẩn bị corpus hay build index. Nếu thiếu thư mục bind mount, Compose sẽ báo lỗi thay vì tạo một thư mục rỗng. Các scripts chuẩn bị/build có trong repo; image giữ `scripts/` để có thể chạy batch, nhưng cần mount dữ liệu/index với quyền ghi khi build chúng.

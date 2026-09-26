# Task 1 — encoder theo model và build dense ViMed

## Quy ước encoding

| Profile | Pooling | Query prefix | Passage prefix |
|---|---|---|---|
| `bge_m3` | CLS token đầu tiên | rỗng | rỗng |
| `e5` | mean với attention mask | `query: ` | `passage: ` |
| `generic` | mean mặc định, cho phép cấu hình | rỗng mặc định | rỗng mặc định |

`auto` nhận diện tên `bge-m3` hoặc model E5; với đường dẫn/model alias không
nhận diện được, chỉ định profile rõ ràng. Profile BGE-M3/E5 từ chối các override
không đúng quy ước để tránh âm thầm chạy sai encoder.

Quy ước BGE-M3 đối chiếu với [implementation chính thức](https://github.com/FlagOpen/FlagEmbedding/blob/master/FlagEmbedding/inference/embedder/encoder_only/m3.py)
và [model card](https://huggingface.co/BAAI/bge-m3).

Build, runtime factory, dense benchmark và smoke script đều dùng cùng profile.
Manifest format 2 ghi profile, pooling, prefix, precision và revision model/tokenizer.
Cache format 1 vẫn được đọc nếu quy ước khớp; BGE cache mean pooling cũ bị từ chối.
Không thể chỉ sửa manifest của cache cũ: bắt buộc encode lại passages.

## Tài nguyên và môi trường

Máy kiểm tra có RTX 4050 Laptop 6 GB VRAM, khoảng 4,7 GB VRAM trống,
16 GB RAM tổng và khoảng 2,7–3 GB RAM trống; ổ đĩa trống khoảng 191 GB.
PyTorch ban đầu là CPU-only. Bản CUDA được chuẩn bị trong `.venv`, không cài
vào Python hệ thống. Model BGE-M3 revision đã pin có sẵn trong cache dự án.

Thiết lập build khởi đầu: CUDA, FP16, batch 4, max_length 512. Output vector
vẫn là float32 để lưu và tìm kiếm. FP16 có thể tạo sai khác số học nhỏ so với FP32.
512 tokens là giới hạn thực thi hiện tại; task này chưa tối ưu chunking/truncation.

## Build có timeout

Từ thư mục gốc:

```powershell
$env:HF_HOME = "$PWD/.cache/huggingface"
$env:HF_HUB_OFFLINE = "1"
.venv/Scripts/python.exe -m scripts.build_dense_index --config configs/vimed_dense_bge_m3.yaml --limit 128 --output-index artifacts/vimed_dense_probe --timeout-seconds 300
.venv/Scripts/python.exe -m scripts.build_dense_index --config configs/vimed_dense_bge_m3.yaml --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.verify_dense_index --config configs/vimed_dense_bge_m3.yaml
.venv/Scripts/python.exe -m pytest -q
```

Probe ghi throughput, peak allocated/reserved VRAM và thời gian để chọn timeout
cho build đầy đủ. Timeout bao gồm cả import/load model. Nếu quá hạn, tiến trình
worker bị dừng và supervisor trả exit code 124.

Probe thực tế đạt: 128 context encode trong 2,093 giây (~61,16 context/giây),
peak allocated 1.182.187.520 bytes (~1,10 GiB), reserved ~1,13 GiB.
Ước tính full encoding ~294 giây; chọn timeout 1.200 giây để có dư địa.
Trong môi trường sandbox của phiên này, DLL scikit-learn bị Application Control
chặn khi Transformers import; chạy probe ngoài sandbox theo approval đã thành công.

Worker dùng thư mục `.building`, công bố index bằng rename sau khi ghi đầy đủ.
Index đích đã có dữ liệu sẽ không bị ghi đè. Nếu lần trước thất bại, kiểm tra
thư mục `.building` và dùng đường dẫn output mới khi chạy lại.

## Điều kiện hoàn tất

- Unit/regression tests đạt: CLS khác mean, mask padding, prefix theo model,
  cùng profile cho build/query, từ chối stale cache, deadline/failure exit code.
- Build đủ 17.955 context bằng model thật trên GPU.
- Kiểm tra số lượng/ID/hash/dimension; embeddings hữu hạn và có norm gần 1.
- Encode lại passages khớp cache trong sai số FP16 cho phép.
- Query embedding khớp tham chiếu raw transformer CLS; batch/single retrieval nhất quán.
- Kết quả xác minh lưu ở `outputs/vimed_dense_build/verification.json`.

Đây là xác minh encoding/index. So sánh Recall giữa BM25/dense/hybrid trên
validation thuộc task 2, chưa phải điều kiện chất lượng được đo trong task này.

## Kết quả hoàn tất task 1

- PyTorch `.venv`: `2.7.1+cu128`, CUDA compute kiểm tra thành công trên RTX 4050;
  `pip check`: không có dependency hỏng.
- Full suite sau thay đổi cuối: **103 passed**, một warning deprecation Starlette/AnyIO.
- Full index: **17.955 × 1.024**, profile BGE-M3/CLS, prefix rỗng,
  model FP16 trên CUDA, lưu vectors float32.
- Encoding: **195,844 giây**; load + build: **206,859 giây**;
  supervisor tổng: **212,2 giây**, timeout **1.200 giây**.
- Peak allocated VRAM: **1.196.008.448 bytes (~1,11 GiB)**;
  peak reserved: **1.231.028.224 bytes (~1,15 GiB)**.
- Kiểm tra model thật: **passed**; tất cả vectors hữu hạn và normalized;
  cache/re-encode max absolute error **0,00024414** trong sai số FP16;
  CLS query/reference max error **0**; batch/single top 5 nhất quán cho ba query smoke.

Index: `artifacts/vimed_dense_bge_m3_cls/`.
Báo cáo: `outputs/vimed_dense_build/verification.json` và `task1_result.json`.
Task 2 chưa được chạy.

## Khôi phục tải PyTorch CUDA

Nếu pip tải file lớn bị cắt, utility `scripts/fetch_torch_cuda.py` tải wheel
Windows CPython 3.12 Torch 2.7.1 CUDA 12.8 từ `download.pytorch.org` theo ranges.
Chỉ tạo file wheel hoàn chỉnh nếu size và SHA256 khớp checksum chính thức.
Các phần nằm trong `.cache/pip/torch_cuda_parts` để có thể tiếp tục lần sau.

```powershell
.venv/Scripts/python.exe -m scripts.fetch_torch_cuda
.venv/Scripts/python.exe -m pip install --no-deps '.cache/pip/torch-2.7.1+cu128-cp312-cp312-win_amd64.whl'
```

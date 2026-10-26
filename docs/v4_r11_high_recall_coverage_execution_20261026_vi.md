# R11 — kết quả canary và quyết định budget

Ngày: 2026-10-26. Trạng thái: **STOP_BUDGET tại canary; chưa chạy full cohort**.

## Đã hoàn tất

- Cohort lock PASS: 87/87 manual query–parent pairs được xác minh lại bằng
  source SHA-256 và evidence round-trip trên source hiện hành.
- Cap-10 có 76/87 manual parents; còn 11 ngoài champion document lists.
- Khóa thêm 10 retained controls; 3 control queries trùng tập manual, nên tổng
  cohort là 94 query.
- BGE-M3 revision, index fingerprint và input hashes được lưu trong
  `outputs/v4/r11_high_recall_coverage_20261026/cohort_lock.json`.
- Raw branch prefixes và frozen S2 mixed fusion khớp chính xác trên cả ba
  preflight queries `53, 130, 380`. Thử dense FP32 cũng giữ exact top-200 trên
  cả ba query.

## Canary cost

Canary depth 4.000 mất 1.454,60 CPU-giây cho raw và normalized variants của ba
query. Từ đó projection cho 94 query là:

| Run | Projected CPU seconds | Budget |
|---|---:|---:|
| Raw-only | 22.792,9 | 14.400 |
| Raw + normalized | 45.577,5 | 14.400 |

Raw-only vẫn vượt cap khoảng 1,58 lần; hai variants vượt khoảng 3,17 lần. Peak
RSS ghi nhận 15,98 GB. Vì vậy runner không mở retrieval cho toàn cohort. Đã
lưu ba canary traces trong `traces/` (FP64 reference) và `traces_fast/` (FP32
dense); không có reranking, submission hay official F2 run.

Projection dựa trên ba preflight queries cố định và là ước lượng chi phí, không
phải kết quả coverage. Gate thu hồi manual parents/control vì vậy chưa được đo.
Artifact quyết định là `outputs/v4/r11_high_recall_coverage_20261026/decision.json`.

## Handoff

R11 cần một trong hai thay đổi để tiếp tục mà vẫn giữ mục tiêu đo đủ 94 query:

1. Cho phép ngân sách lớn hơn dựa trên projection đã đo; hoặc
2. Chuẩn bị một dense retrieval path có cùng index/model identity và chứng minh
   exact prefix parity trước khi chạy full cohort.

Các trace canary và cohort lock được giữ nguyên để không phải lặp lại phần đã
hoàn tất. `STOP_BUDGET` không có nghĩa retrieval quality đã fail; coverage gate
chưa được đánh giá.

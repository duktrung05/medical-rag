# Kế hoạch thu thập phần URL Long Châu còn thiếu

Ngày lập: 14/10/2026, giờ Việt Nam. Phạm vi: corpus ViBioMIR revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`, hostname `nhathuoclongchau.com.vn`.

## Kết quả thực thi ngày 14/10

Pilot kết nối thật chạy lúc 22:25, dùng wave `longchau_20261014_live`. `robots.txt` trả về được, nhưng URL bài viết đầu tiên (official ID `204679`) trả HTTP 403. Crawler dừng ngay; **1/10 URL đã thử, 9/10 chưa gửi request, 0 tài liệu mới usable**. Cổng pilot không đạt, vì vậy chưa chạy bất kỳ lô mở rộng nào. Danh sách còn thiếu vẫn là **77.571 URL**. Kết quả nằm ở `outputs/crawl_failure_recovery/recovered_raw/pilot_longchau_20261014_live/rate_limited/batch_00000/crawl_summary.json` và `RECOVERY_SUMMARY.json`.

Từ kết quả HTTP 403, bộ lọc retry mới giữ toàn bộ hostname Long Châu khỏi các wave tự động tiếp theo; chỉ tiếp tục khi có cơ sở truy cập hợp lệ hoặc phương án dữ liệu chính thức được xác nhận. Pilot đầu tiên trong sandbox bị lỗi kết nối tới `robots.txt`, không chạm trang bài viết; 10 kết quả giả này đã được tách khỏi bảng kết quả hoạt động và lưu trong `outputs/crawl_failure_recovery/sandbox_diagnostics/20261014_robots_connect_error/`.

## Mốc hiện tại

| Chỉ số | Số lượng |
|---|---:|
| Official ID thuộc Long Châu | 79.598 |
| Tài liệu đã usable trong release bước 2 | 2.027 |
| URL còn thiếu, nhóm `rate_limited` | 77.571 |
| URL đã qua cooldown tại lúc kiểm tra 22:02 ngày 14/10 | 77.571 |

Nguồn: `outputs/vibiomir_step2_20261013/STEP2_REPORT.md`, `outputs/crawl_failure_recovery/eligible/FAILED_URLS_ELIGIBLE.parquet`. Lần crawl gần nhất ghi 404 thành công và 19 phản hồi HTTP 429 trong batch 500 URL; batch dừng vì các phản hồi bị chặn liên tiếp. Đã thử 1 worker, giãn 3 giây. Mốc cooldown 24 giờ là chính sách nội bộ trong `audit_failed_urls.py`, không phải cam kết của website rằng sẽ chấp nhận request sau mốc đó.

## Trình tự thực hiện

1. **Đóng băng đầu vào.** Dùng danh sách 77.571 ID–URL trong `FAILED_URLS_ELIGIBLE.parquet`, lọc đúng hostname và revision; đối chiếu với `release/document_status.parquet` và mọi kết quả recovery mới. Chỉ đưa ID còn thiếu vào hàng đợi. Không tải lại 2.027 ID đã usable hoặc URL đã được recovery thành công. Ghi checksum danh sách và số lượng trước khi gửi request.
2. **Kiểm tra quy tắc truy cập và chạy pilot.** Kiểm tra lại `robots.txt` bằng chính user agent của crawler trước lượt chạy. Tạo một wave riêng chỉ cho nhóm `rate_limited`; `dry-run` phải hiện đúng 10 URL pilot và không gửi request. Chạy pilot với 1 worker, tối thiểu 10 giây giữa request cùng domain, giữ nguyên giới hạn `Retry-After` nếu server trả về. Raw, extraction và kết quả nằm trong `outputs/crawl_failure_recovery/`.
3. **Chốt cổng pilot.** Chỉ mở rộng khi cả 10 URL đã được xử lý, ít nhất 8 tài liệu usable, không có HTTP 429, 403, 428, access challenge hoặc robots denied. Cổng hiện có của `recover_failed_urls.py` kiểm tra ngưỡng 8/10 và một số trạng thái bị chặn; phải bổ sung kiểm tra HTTP 403/428 và `access_blocked` trước khi mở rộng. Nếu có phản hồi giới hạn/chặn, dừng domain, lưu thời điểm thử và đặt cooldown mới; không thử đổi IP, user agent hay tăng concurrency để vượt giới hạn.
4. **Mở rộng có kiểm soát.** Sau pilot đạt, chạy lần lượt các lô 50, 100, 250 rồi tối đa 500 URL. Mỗi lô dùng 1 worker và giãn ít nhất 10 giây. Kiểm tra kết quả sau từng lô; chỉ tăng cỡ lô nếu không có 429/challenge/robots denial, tất cả ID có trạng thái cuối và tỷ lệ usable ít nhất 95%. Dừng ngay khi có 429 hoặc dấu hiệu bị chặn; cập nhật cooldown từ lần thử mới nhất trước khi lập wave tiếp theo. Nếu nhiều lô ổn định, giữ lô 500 và tiếp tục cho đến khi hết URL đủ điều kiện.
5. **Đối soát và hợp nhất.** Sau mỗi lô, kiểm tra raw checksum, HTML thực, extraction, ID–URL chính thức, không trùng ID và không sửa dữ liệu đã usable. Materialize tài liệu mới vào vùng staging. Chỉ hợp nhất release rồi cập nhật chunks/index khi các phép kiểm tra qua; xuất coverage Long Châu gồm `planned`, `attempted`, `usable`, `rate_limited`, `blocked`, `unavailable`, `deferred`, cùng danh sách ID còn thiếu và `next_retry_at`.

## Việc phải sửa trước khi chạy diện rộng

- `recover_failed_urls.py` có pilot 10 URL, nhưng `run` hiện lập kế hoạch cho **toàn bộ** URL của nhóm đã qua pilot. Cần thêm giới hạn số URL/lô hoặc controller wave để triển khai nấc 50 → 100 → 250 → 500 mà không đưa 77.000+ URL vào một lượt.
- Cần thêm cổng dừng ngay khi có HTTP 429/challenge/403/428, thay cho ngưỡng nhiều lần bị chặn liên tiếp của crawler hiện tại; ghi số URL chưa gửi request là `deferred`.
- Cooldown trong allowlist hiện được tính từ các attempt trước lúc audit. Sau một pilot/lô mới có 429, cần ghi lịch sử attempt và tính lại `next_retry_at`; không dùng lại thời điểm cũ để retry ngay.
- Runner phải checkpoint từng lô, resume đúng ID chưa xử lý và kiểm tra lại kết quả trước khi gửi request. Không tính URL bị hoãn vì circuit là đã tải thành công.
- Không chạy lệnh `run --wave ...` của recovery hiện tại cho cả nhóm Long Châu trước khi có giới hạn lô và cooldown động. Với khoảng cách tối thiểu 10 giây, riêng 77.571 request đã cần ít nhất khoảng **9 ngày** nếu chạy liên tục, chưa tính thời gian mạng, retry và các lần dừng do 429. Không thể hứa ngày hoàn tất khi chưa có pilot ổn định.

## Nơi kiểm tra kết quả

- Lịch sử pilot và các lô: `outputs/crawl_failure_recovery/recovered_raw/`, `recovered_extracted/`, `RECOVERY_RESULTS.parquet`.
- Trạng thái và số còn thiếu: `outputs/crawl_failure_recovery/RECOVERY_REMAINING_FAILED.parquet`.
- Release hiện tại để đối soát: `outputs/vibiomir_step2_20261013/release/document_status.parquet` và `missing_documents.parquet`.

`robots.txt` chính thức được kiểm tra ngày 14/10/2026 tại <https://nhathuoclongchau.com.vn/robots.txt>; quy tắc có thể thay đổi, vì vậy crawler vẫn phải kiểm tra lại vào lúc thực thi.

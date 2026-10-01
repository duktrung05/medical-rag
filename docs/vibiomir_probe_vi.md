# ViBioMIR — HTTP probe 20 URL

Ngày: 2026-10-01. Lệnh: `python -m scripts.probe_vibiomir_urls` và một lượt `--retry-failed` sau khi bổ sung xử lý robots redirect HTTP→HTTPS cùng hostname.

Selection được ghi vào `data/vibiomir/probe/selected_urls.json` trước network request. Chọn 8 domain lớn nhất, 6 domain quanh trung vị và 6 domain nhỏ nhất theo `domain_url_count`; trong mỗi domain lấy mẫu có `selection_hash` nhỏ nhất. Có đúng 20 domain khác nhau.

Kết quả: **12/20 thành công (60%)**; **8 robots_unavailable**, 0 robots_denied. Tất cả success có content type `text/html`. Tổng byte đã tải: **758,063**.

Lần đầu: **39,91 giây**, peak RSS **139,28 MiB**. Lượt retry failed: **143,54 giây**, peak RSS **141,53 MiB**. Hai lệnh đều exit 0; thời gian đợi cấp quyền nằm ngoài thời gian chạy lệnh.

Robots trả 404 được coi là cho phép; timeout, 403, 5xx/521 hoặc redirect robots ngoài phạm vi an toàn khiến page request bị bỏ qua. Trước mỗi page redirect, fetcher kiểm robots cho origin đích. Raw chỉ lưu khi status success, dạng `raw/<official_id>.bin`.

## 20 URL đã chọn và status cuối

| # | Group | Domain | Official ID | Status | Robots HTTP | Page HTTP | Content type | Bytes | URL gốc |
|---:|---|---|---:|---|---:|---:|---|---:|---|
| 1 | large | cnkang.com | 1690344 | success | 200 | 200 | text/html | 58,893 | http://www.cnkang.com/dzjk/mrbj/hfyy/mljf/kz/200805/107758.html |
| 2 | large | 120ask.com | 1006158 | success | 200 | 200 | text/html | 68,217 | http://www.120ask.com/question/108511271.htm |
| 3 | large | familydoctor.com.cn | 3715486 | success | 200 | 200 | text/html | 65,707 | https://www.familydoctor.com.cn/fuke/info/201111/98228110720.html |
| 4 | large | ask.39.net | 2381039 | success | 404 | 200 | text/html | 53,803 | https://ask.39.net/question/94728413.html |
| 5 | large | zysjonline.com | 4266246 | robots_unavailable | 403 | — | — | 0 | https://zysjonline.com/books/jiatingyixuebaike_zijiuhujiu/191783/ |
| 6 | large | a-hospital.com | 3103309 | success | 200 | 200 | text/html | 36,251 | https://www.a-hospital.com/w/%E7%A5%9E%E4%BB%99%E9%81%97%E8%AE%BA |
| 7 | large | suckhoecongdongonline.vn | 348999 | success | 404 | 200 | text/html | 42,218 | https://suckhoecongdongonline.vn/cay-ngay-huong-va-nhung-bai-thuoc-dan-gian-dieu-tri-benh |
| 8 | large | zhongyibaodian.net | 4099606 | success | 404 | 200 | text/html | 16,297 | https://zhongyibaodian.net/a/21772.html |
| 9 | medium | heart.39.net | 863461 | robots_unavailable | 301 | — | — | 0 | http://heart.39.net/a/210501/8877628.html |
| 10 | medium | article.iiyi.com | 2133559 | robots_unavailable | 521 | — | — | 0 | https://article.iiyi.com/detail/33789.html |
| 11 | medium | suckhoeviet.org.vn | 565962 | success | 200 | 200 | text/html | 97,941 | https://suckhoeviet.org.vn/ha-giang-theo-duoi-muc-tieu-tro-thanh-vung-trong-diem-quoc-gia-ve-duoc-lieu-8019.html |
| 12 | medium | baoquangtri.vn | 89564 | success | 404 | 200 | text/html | 51,172 | https://baoquangtri.vn/suc-khoe/202603/bo-y-te-de-xuat-tiep-tuc-cat-giam-thu-tuc-hanh-chinh-ve-kham-chua-benh-d0f7a12/ |
| 13 | medium | pmc-ecm-healthblog.beta.pharmacity.io | 322617 | robots_unavailable | 403 | — | — | 0 | https://pmc-ecm-healthblog.beta.pharmacity.io/ung-thu-dai-trang-nguyen-nhan-dau-hieu-va-phuong-phap-dieu-tri/ |
| 14 | medium | man.39.net | 865318 | robots_unavailable | 301 | — | — | 0 | http://man.39.net/a/2010224/1162428.html |
| 15 | small | ask.familydoctor.com.cn | 766443 | robots_unavailable | 403 | — | — | 0 | http://ask.familydoctor.com.cn/q/3909357.html |
| 16 | small | mega.vietnamplus.vn | 199048 | success | 200 | 200 | text/html | 62,242 | https://mega.vietnamplus.vn/theo-chan-nguoi-dan-bien-vien-mien-tay-xu-nghe-xuyen-rung-hai-loc-troi-5412.html |
| 17 | small | yanglao.familydoctor.com.cn | 2131080 | robots_unavailable | 403 | — | — | 0 | http://yanglao.familydoctor.com.cn/zt/scsjtfz.html |
| 18 | small | fk.99.com.cn | 836542 | success | 200 | 200 | text/html | 101,027 | http://fk.99.com.cn/news/410997.html |
| 19 | small | special.vietnamplus.vn | 583 | success | 200 | 200 | text/html | 104,295 | http://special.vietnamplus.vn/benh_vien_nhan_ai |
| 20 | small | ypk.familydoctor.com.cn | 4030514 | robots_unavailable | — | — | — | 0 | https://ypk.familydoctor.com.cn/267067/ |

## Domain bị chặn hoặc lỗi ở bước robots

| Domain | Robots HTTP | Error type | Diễn giải |
|---|---:|---|---|
| zysjonline.com | 403 | robots_http_error | robots.txt returned HTTP 403 |
| heart.39.net | 301 | robots_redirect | Robots redirect left its origin or exceeded limit |
| article.iiyi.com | 521 | robots_http_error | robots.txt returned HTTP 521 |
| pmc-ecm-healthblog.beta.pharmacity.io | 403 | robots_http_error | robots.txt returned HTTP 403 |
| man.39.net | 301 | robots_redirect | Robots redirect left its origin or exceeded limit |
| ask.familydoctor.com.cn | 403 | robots_http_error | robots.txt returned HTTP 403 |
| yanglao.familydoctor.com.cn | 403 | robots_http_error | robots.txt returned HTTP 403 |
| ypk.familydoctor.com.cn | — | RemoteProtocolError | Server disconnected without sending a response. |

## Kiểm tra

- `79 passed` gồm tests fetcher/probe và hai bộ tests inventory/sampling; Ruff cho file mới pass.
- Đọc lại Parquet xác nhận đúng 20 ID, 20 URL và 20 domain duy nhất; mỗi dòng khớp selected manifest.
- SHA-256 và số byte của 12 raw file khớp metadata. Tám dòng robots_unavailable có `attempt_count=0`, không có raw file và không có page HTTP status.
- `data/vibiomir/` được ignore trong Git. Không trích xuất HTML/PDF, chunk hoặc gọi model.

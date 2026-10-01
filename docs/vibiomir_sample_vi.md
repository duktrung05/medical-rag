# ViBioMIR — kết quả chọn 500 URL mẫu

Ngày thực hiện: 2026-10-01. Chạy `python -m scripts.sample_vibiomir_urls` từ thư mục gốc repository.

Script kiểm SHA256 corpus trước khi chọn; dùng `domain_stats.csv` để chia quota và quét Parquet theo batch. Trong mỗi domain, chọn các SHA256 nhỏ nhất của chuỗi official ID, ký tự NUL và URL gốc. Không gửi HTTP request.

Corpus: **4,394,718 URL**; sample: **500 URL**, bao phủ **97/97 domain**. Mẫu không có ID hoặc URL trùng.

Corpus SHA256: `dee1153f5b9d9010dea18e1b7724b4915ff847b711d036bec8b933486802a971`. Sample Parquet SHA256: `ce811cdabefce0b2bdcb5b678408e5f525c5dfeb299a0c90bd30c9dca3ba8332`. Thuật toán: `domain_quota_sha256_v1`.

Sáu domain có dưới 5 URL, nên base thực tế là **465 slot**. **35 slot** còn lại được cấp lần lượt theo quy mô domain. Quota distribution: 3 domain nhận 1 URL, 2 domain nhận 2, 1 domain nhận 3, 56 domain nhận 5, 35 domain nhận 6.

Hai lần chạy thật đều exit 0. Lần 1: **27,79 giây**, peak RSS **190,38 MiB**. Lần 2: **29,80 giây**, peak RSS **205,38 MiB**. Cả hai output giống nhau theo SHA256.

Kiểm thử: `tests/workflows/vibiomir/test_sample_vibiomir_urls.py` cùng `tests/workflows/vibiomir/test_prepare_vibiomir.py`: **47 passed**. Ruff và `git diff --check` pass. Đã đọc lại mẫu và quét corpus lần hai để xác nhận cả 500 cặp official ID/URL xuất hiện đúng một lần.

## Phân bổ cho 20 domain lớn nhất

| Domain | URL trong corpus | Quota | Base | Extra |
|---|---:|---:|---:|---:|
| cnkang.com | 963,438 | 6 | 5 | 1 |
| 120ask.com | 918,479 | 6 | 5 | 1 |
| familydoctor.com.cn | 447,453 | 6 | 5 | 1 |
| ask.39.net | 337,375 | 6 | 5 | 1 |
| zysjonline.com | 243,347 | 6 | 5 | 1 |
| a-hospital.com | 169,339 | 6 | 5 | 1 |
| suckhoecongdongonline.vn | 154,503 | 6 | 5 | 1 |
| zhongyibaodian.net | 146,700 | 6 | 5 | 1 |
| suckhoedoisong.vn | 85,823 | 6 | 5 | 1 |
| nhathuoclongchau.com.vn | 79,598 | 6 | 5 | 1 |
| zydcd.com | 78,680 | 6 | 5 | 1 |
| thanhnien.vn | 72,111 | 6 | 5 | 1 |
| wujue.com | 53,341 | 6 | 5 | 1 |
| youlai.cn | 46,119 | 6 | 5 | 1 |
| laodong.vn | 31,282 | 6 | 5 | 1 |
| baby.39.net | 27,621 | 6 | 5 | 1 |
| phunusuckhoe.giadinhonline.vn | 27,577 | 6 | 5 | 1 |
| vinmec.com | 26,129 | 6 | 5 | 1 |
| bingli.iiyi.com | 24,990 | 6 | 5 | 1 |
| medlatec.vn | 24,762 | 6 | 5 | 1 |

## 20 dòng đầu của sample_urls.parquet

Thứ tự của file: domain tăng dần, rank trong domain tăng dần, sau đó ID tăng dần.

| ID | Domain | Rank | Reason | URL gốc |
|---:|---|---:|---|---|
| 1006158 | 120ask.com | 0 | domain_base | http://www.120ask.com/question/108511271.htm |
| 2655755 | 120ask.com | 1 | domain_base | https://www.120ask.com/question/13608457.htm |
| 2829477 | 120ask.com | 2 | domain_base | https://www.120ask.com/question/3567962.htm |
| 1260622 | 120ask.com | 3 | domain_base | http://www.120ask.com/question/40450608.htm |
| 1005760 | 120ask.com | 4 | domain_base | http://www.120ask.com/question/108510645.htm |
| 2638481 | 120ask.com | 5 | large_domain_extra | https://www.120ask.com/question/1168390.htm |
| 3103309 | a-hospital.com | 0 | domain_base | https://www.a-hospital.com/w/%E7%A5%9E%E4%BB%99%E9%81%97%E8%AE%BA |
| 3131943 | a-hospital.com | 1 | domain_base | https://www.a-hospital.com/w/%E8%B0%83%E7%BB%8F%E8%BF%87%E6%9C%9F%E6%B1%A4 |
| 3129770 | a-hospital.com | 2 | domain_base | https://www.a-hospital.com/w/%E8%A5%84%E6%A8%8A%E5%B8%82%E5%86%9B%E5%B7%A5%E5%8C%BB%E9%99%A2 |
| 3022578 | a-hospital.com | 3 | domain_base | https://www.a-hospital.com/w/%E5%8F%A4%E8%8F%8C |
| 3008568 | a-hospital.com | 4 | domain_base | https://www.a-hospital.com/w/%E5%85%B0%E5%B7%9E%E8%A5%BF%E5%9B%BA%E5%8C%BA%E5%A6%87%E5%B9%BC%E4%BF%9D%E5%81%A5%E9%99%A2 |
| 3074193 | a-hospital.com | 5 | large_domain_extra | https://www.a-hospital.com/w/%E6%B0%94%E8%85%B9 |
| 2133559 | article.iiyi.com | 0 | domain_base | https://article.iiyi.com/detail/33789.html |
| 2135298 | article.iiyi.com | 1 | domain_base | https://article.iiyi.com/detail/411477.html |
| 2134279 | article.iiyi.com | 2 | domain_base | https://article.iiyi.com/detail/409946.html |
| 2135736 | article.iiyi.com | 3 | domain_base | https://article.iiyi.com/detail/412123.html |
| 2134242 | article.iiyi.com | 4 | domain_base | https://article.iiyi.com/detail/409863.html |
| 2381039 | ask.39.net | 0 | domain_base | https://ask.39.net/question/94728413.html |
| 2265666 | ask.39.net | 1 | domain_base | https://ask.39.net/question/52655528.html |
| 2394453 | ask.39.net | 2 | domain_base | https://ask.39.net/question/99645393.html |

Artifacts: `data/vibiomir/sample_urls.parquet`, `data/vibiomir/sample_summary.json`. Cả hai cùng Parquet nguồn nằm ngoài Git theo `.gitignore`.

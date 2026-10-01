# ViBioMIR — kiểm kê dữ liệu local

Ngày thực hiện: 2026-10-01. Dataset: `AIGuruTinix/ViBioMIR`.

Chạy từ thư mục gốc repository:

```bash
python -m scripts.prepare_vibiomir
python -m pytest -q tests/test_prepare_vibiomir.py
```

CLI hỗ trợ `--query-file`, `--corpus-file`, `--output-dir`, `--batch-size` và `--keep-www`. Mặc định đọc `data/raw/vibiomir/{query,links_corpus}.parquet`, xuất vào `data/vibiomir/`, batch 50.000 dòng.

Query được đọc đầy đủ bằng PyArrow. Corpus được đọc theo batch; SQLite tạm lưu các cặp official ID/URL, đếm trùng và thống kê hostname. SQLite dùng cache 8 MiB, tắt mmap và sort trên disk; toàn bộ corpus không được đưa vào DataFrame hoặc list trong RAM. SQLite tạm được xóa khi hoàn tất.

Schema phải đúng thứ tự hai cột: query có `id: integer`, `query: string`; corpus có `id: int64`, `url: string`. Nullability trong schema được ghi lại và null trong dữ liệu được đếm riêng. Không giả định IDs liên tục hoặc bắt đầu từ 1.

**Quy tắc đếm và xuất report**

- Duplicate ID/URL là số lần xuất hiện dư sau lần đầu, bỏ qua giá trị null. So sánh URL nguyên văn, bao gồm case/whitespace/path/query string.
- `invalid_urls` bao gồm tất cả URL null, rỗng hoặc không hợp lệ; reason tối thiểu là `null_url`, `empty_url`, `unsupported_scheme`, `missing_hostname`. Lỗi parser được ghi thêm `malformed_url`.
- Hostname dùng `urlsplit`, lowercase, bỏ dấu chấm cuối và mặc định bỏ `www.`. Đây là hostname, không phải phép gộp các subdomain theo registered domain.
- Domain count đếm mọi dòng URL HTTP(S) hợp lệ, bao gồm URL trùng; sắp xếp count giảm dần rồi hostname tăng dần.
- `duplicate_urls.csv` có `url,id,occurrence_count`: mỗi occurrence là một dòng, giữ đủ mapping official ID kể cả ID null/trùng. Không gom hàng triệu IDs vào một list.
- SHA256 được tính theo block 1 MiB.
- Các report được tạo trong thư mục tạm cùng filesystem, rồi rename từng file sau khi quét hoàn tất; inventory được publish cuối. Không có giao dịch atomic cho cả bốn file đồng thời. Lỗi schema/quét trước bước publish giữ nguyên report cũ.
- Khi chạy lại cùng dữ liệu và tùy chọn vào cùng output directory, `created_at` ban đầu được giữ nếu toàn bộ nội dung tính toán không đổi; reports deterministic theo byte. Thời gian chạy/peak RSS chỉ in ở summary.
- Exit code: `0` hợp lệ (URL lỗi riêng lẻ vẫn xuất report); `1` nếu query null/rỗng hoặc ID null/trùng; `2` nếu schema, input hoặc thao tác xử lý bị lỗi.

**Kết quả dữ liệu thật**

| Metric | Query | Corpus |
|---|---:|---:|
| Số dòng thực tế | 1.200 | 4.394.718 |
| Row groups | 1 | 44 |
| ID min | 1 | 583 |
| ID max | 1.200 | 4.420.561 |
| ID null | 0 | 0 |
| Duplicate ID | 0 | 0 |
| Query null / rỗng | 0 / 0 | — |
| URL null / rỗng / invalid | — | 0 / 0 / 0 |
| Duplicate URL | — | 0 |
| Unique hostnames sau gộp www | — | 97 |

Query SHA256: `ce13508b893bbd6b9169679d540630bc37eb5950cad08b4d5e0e5d7063670c08`.

Corpus SHA256: `dee1153f5b9d9010dea18e1b7724b4915ff847b711d036bec8b933486802a971`.

Lần chạy đầu exit `0`, thời gian **28,42 giây**, peak process RSS **181,56 MiB** trên Linux/Python 3.14.6. Peak RSS được đo bằng `resource.getrusage(RUSAGE_SELF).ru_maxrss`. Đây là số đo của lần chạy này, có thể khác theo máy và batch size.

Lần chạy xác nhận thứ hai cũng exit `0`: **28,75 giây**, peak RSS **183,57 MiB**. SHA256 của cả bốn output giữ nguyên giữa hai lần chạy. Đã kiểm tra lại SHA256 hai file nguồn, thứ tự domain CSV, tổng domain count, header/count của các report lỗi/trùng và việc dọn SQLite tạm.

**20 hostname lớn nhất**

| Hostname | URL count |
|---|---:|
| cnkang.com | 963.438 |
| 120ask.com | 918.479 |
| familydoctor.com.cn | 447.453 |
| ask.39.net | 337.375 |
| zysjonline.com | 243.347 |
| a-hospital.com | 169.339 |
| suckhoecongdongonline.vn | 154.503 |
| zhongyibaodian.net | 146.700 |
| suckhoedoisong.vn | 85.823 |
| nhathuoclongchau.com.vn | 79.598 |
| zydcd.com | 78.680 |
| thanhnien.vn | 72.111 |
| wujue.com | 53.341 |
| youlai.cn | 46.119 |
| laodong.vn | 31.282 |
| baby.39.net | 27.621 |
| phunusuckhoe.giadinhonline.vn | 27.577 |
| vinmec.com | 26.129 |
| bingli.iiyi.com | 24.990 |
| medlatec.vn | 24.762 |

**Artifacts và kiểm thử**

- `data/vibiomir/inventory.json`: schema, SHA256, counts, bounds và validation errors.
- `data/vibiomir/domain_stats.csv`: 97 hostname, tổng URL count 4.394.718.
- `data/vibiomir/invalid_urls.csv`: chỉ header vì không có URL lỗi.
- `data/vibiomir/duplicate_urls.csv`: chỉ header vì không có URL trùng.
- Unit tests: **32 passed**; chỉ dùng fixtures nhỏ trong `tmp_path`. Có kiểm tra duplicate xuyên batch/row group, integer ID lớn, schema sai, null/rỗng, malformed URL, deterministic output, bounded hash reads, corpus streaming và giữ output cũ khi scan bị ngắt.
- Ruff cho script/tests mới và `git diff --check`: pass.

File triển khai: `scripts/prepare_vibiomir.py`, `tests/test_prepare_vibiomir.py`, `.gitignore` và tài liệu này. Parquet nguồn và thư mục output đã được ignore trong Git; không commit dữ liệu. Các thay đổi không liên quan trong working tree được giữ nguyên.

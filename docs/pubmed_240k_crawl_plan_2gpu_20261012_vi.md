# Kế hoạch thu thập 240.000 PubMed mới — 2 RTX A6000

Ngày lập: 12/10/2026. Phạm vi: đúng 240.000 URL PubMed mới của ViBioMIR revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`. Đây là kế hoạch và bộ đầu vào đã chuẩn bị; chưa chạy collector, GPU job hoặc thay runtime.

## 1. Kết quả cần bàn giao

Thu thập bản ghi PubMed theo official ID–URL, lưu raw XML, trích title/abstract và metadata, đối soát trạng thái đủ 240.000 ID. Có thể chạy hai encoder để tạo vector document bổ sung sau khi text đạt kiểm tra. Giữ phần crawl/chunks cũ; phạm vi này không bao gồm recovery Long Châu/39.net hoặc chấm lại toàn bộ 1.200 queries.

PubMed chỉ cần HTTP API để lấy dữ liệu, không dùng GPU để tải. Hai GPU được dành cho embedding. Theo yêu cầu tránh ảnh hưởng server, khởi đầu **4 worker chuẩn bị dữ liệu mỗi GPU**, tổng 8 worker; chỉ nâng lên 8 mỗi GPU sau khi đo được tài nguyên còn dư. **Mỗi GPU có một encoder/model**, không nhân thành 4–8 bản sao model trên mỗi card.

## 2. Đầu vào đã chuẩn bị

Thư mục: `outputs/pubmed_240k_plan_2gpu_20261012/`.

- `pubmed_queue_240000.parquet`: official ID, official URL, PMID, fetch batch ID, GPU shard và trạng thái `planned`.
- `pubmed_pilot_1000.parquet`: 1.000 dòng trải đều trên danh sách official ID đã sort; phân tầng abstract/năm/ngôn ngữ bổ sung sau khi có response.
- `plan_manifest.json`: checksum, revision, số PMID/ID duy nhất và phân bố shards.
- `planned_config.json`: cấu hình đề xuất; collector chưa triển khai đọc/chạy file này.

Đã xác nhận 240.000 ID và 240.000 PMID duy nhất, URL đều thuộc `pubmed.ncbi.nlm.nih.gov`. Chia 1.200 API batch, mỗi batch 200 PMID; hai shard embedding cân bằng 120.000 documents/shard trước khi loại metadata-only/failed. Giữ official ID và PMID thành hai trường riêng.

Nguồn: `outputs/vibiomir_official_verify_20261012/official_missing_locally.parquet`. Khi bắt đầu triển khai, kiểm tra lại metadata Hugging Face và checksum. Nếu corpus đổi thì cập nhật queue theo delta trước khi chạy; không đổi scope âm thầm trong cùng run.

## 3. Cấu hình ít ảnh hưởng server

| Thành phần | Khởi đầu | Giới hạn/điều chỉnh |
|---|---|---|
| GPU | 2 RTX A6000, một encoder/card | Chọn đúng hai physical IDs còn khả dụng; không mặc định GPU 0/1 trống |
| Worker chuẩn bị dữ liệu | 4/GPU, tổng 8 | Tối đa 8/GPU khi kiểm tra CPU/RAM cho phép; không tự tăng |
| Fetch worker | 4 async worker dùng chung | Tách khỏi GPU; 1 request/giây tổng, burst 1 |
| API batch | 200 PMID | Khoảng 1.200 batch trước retry |
| GPU batch | 32 documents | Giảm 16 khi thiếu VRAM; tối đa 64 sau pilot |
| Encoder | BGE-M3 fp16, model/tokenizer pin revision | Document representation; không encode toàn bộ chunks trong lượt này |
| Độ dài encoder đề xuất | 384 tokens | Chốt bằng pilot, ghi truncation; cần thống nhất khi tích hợp index |
| VRAM budget | 12 GiB/GPU | Cấu hình giới hạn allocator nếu triển khai hỗ trợ; batch nhỏ không bảo đảm mức utilization cố định |
| CPU budget | Tối đa 8 cores và tối đa 50% tổng cores host, lấy mức nhỏ hơn | Tính cả encoder/preparation; giảm pool nếu host nhỏ hoặc đang bận |
| RAM budget job | 8 GiB | Đọc/ghi theo batch, queue hữu hạn, không giữ toàn bộ XML/text trong RAM |
| Threads thư viện | OMP/MKL = 1; PyTorch = 1/encoder | Tắt tokenizer parallelism để không nhân thread ngoài pool |
| Ưu tiên | `nice=10`, I/O ưu tiên thấp | Chỉ áp dụng cho tiến trình của job; không đổi cấu hình server/global GPU |
| Theo dõi | Mỗi 30 giây | CPU, RAM khả dụng, GPU memory, I/O, request rate, lỗi |

Giảm hoặc hoãn batch mới khi CPU toàn host >70%, RAM khả dụng <8 GiB hoặc <15% RAM host, hay GPU được chọn còn <16 GiB trước khi nạp model. Khi GPU đang có workload khác, chọn lại thiết bị khả dụng hoặc chạy fetch/extraction trước; giữ embedding ở trạng thái chờ. Các ngưỡng là cấu hình khởi đầu và cần được thực thi trong launcher/controller, không phải điều kiện đã được bảo đảm bởi một file JSON.

Dùng một pool chuẩn bị dữ liệu chung cho parse/tokenize/prefetch, tránh đồng thời tạo thêm pool extraction 8 worker và hai pool GPU 8 worker nữa. Phase tải/trích xuất có thể hoàn tất trước phase GPU để giảm tải cùng lúc. Khi đủ headroom, pipeline theo shard nhỏ nhưng giữ tổng CPU/RAM budget không đổi.

Trần request là cho **tất cả** fetch workers, không phải 1 request/giây/worker. NCBI mặc định áp dụng 3 request/giây khi không có key và 10 khi có key, theo IP/key; cấu hình kế hoạch chủ động giữ 1 request/giây kể cả có key. [NCBI](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/). Phải tính cả traffic NCBI từ job khác dùng chung IP/key.

## 4. Trình tự triển khai

### Bước A — Collector/state và pilot

Triển khai collector PubMed riêng, không gọi crawler HTML cũ cho 240.000 URL. Dùng EFetch POST: `db=pubmed`, `retmode=xml`, `id=<PMIDs>`, thêm tool/email; key tùy chọn từ môi trường. NCBI hỗ trợ tập UID và khuyến nghị POST cho danh sách vượt khoảng 200 UID. [E-Utilities](https://eutilities.github.io/site/Quick_Start/eu_quick/).

State dùng SQLite WAL với một writer và hàng đợi writer có giới hạn. Khóa document theo revision/official ID; khóa attempt theo run/batch/attempt. Ghi timestamp thực tế. Raw XML được gzip và ghi `.tmp` → kiểm checksum/kích thước → rename → checkpoint. HTTP 200 phải đi kèm response đúng định dạng, parse hợp lệ và đối soát PMID.

Pilot tải 1.000 bản ghi, rà thủ công ít nhất 30 gồm structured abstract, Unicode và thiếu abstract nếu gặp. Kiểm tra mapping, response XML lỗi, batch trả thiếu PMID, ngắt/chạy lại, rate limiter và bộ nhớ trước khi mở đủ queue.

### Bước B — Tải đủ 1.200 batch và xử lý retry

Chạy 4 async workers, rate 1 request/giây tổng, timeout 60 giây; hàng đợi ready tối đa 8 batch. Tối đa 16 MiB/response, tối đa 64 MiB raw responses đang xử lý; vượt ngưỡng thì chia batch và thử lại, không cắt response rồi ghi success.

Ghép records bằng PMID trả về, không ghép theo thứ tự. Với PMID thiếu: ghi riêng, requeue tập thiếu hoặc tách batch nhỏ/single-record để chẩn đoán. Duplicate records trong response được phát hiện trước khi publish.

Retry lỗi network/timeout/5xx tối đa 4 attempts/batch với backoff khởi đầu 5/15/45 giây có jitter. 429 tuân thủ `Retry-After`, giảm rate và không giữ worker khi chờ. Khi nhiều batch liên tiếp lỗi, pause nguồn/cooldown thay vì phát thêm requests. Tối đa hai lượt phục hồi dài có khoảng nghỉ khởi đầu 30 phút; lỗi còn lại phải có nguyên nhân và trạng thái chưa hoàn tất rõ ràng.

Resume kiểm raw/checksum/state, không coi tồn tại tên file là xong. Batch XML tốt đã lưu không tải lại; parse lại offline nếu parser thay đổi/hỏng checkpoint. Không đánh dấu cả batch done nếu chưa xử lý hết requested PMIDs.

### Bước C — Trích title/abstract và kiểm tra nội dung

Parser hỗ trợ `PubmedArticle` và bản ghi sách nếu gặp. Giữ thứ tự, label của AbstractText, văn bản inline, ký hiệu y sinh và Unicode. Với nội dung dịch/OriginalAbstract có sẵn, lưu riêng cùng language/provenance; không ghép các abstract khác nhau thành một text không rõ nguồn.

Lưu title, abstract đầy đủ, PMID, DOI/PMCID khi có, language, loại record, raw checksum và parser version. Phân biệt `usable_abstract`, `metadata_only`, `missing_record`, `parse_failed` và retry đang chờ. Không có abstract là đặc điểm dữ liệu, không tự điền nội dung hoặc tính như abstract đã thu được.

Lượt này lấy citation/abstract qua PubMed; chưa tải full text PMC hoặc publisher. Full text nếu cần là một kế hoạch riêng có xác nhận phạm vi nội dung theo BTC. Giữ raw/XML và toàn văn abstract để không mất dữ liệu khi encoder giới hạn token.

Ghi documents theo shard khoảng 5.000 dòng, zstd; publish atomic mỗi shard. Không để một crash làm mất toàn bộ 240.000 records.

### Bước D — Embedding tùy theo đầu ra cần bàn giao

Chỉ khi text đạt kiểm tra mới mở hai GPU. Chia theo shard queue, mỗi encoder nhận khoảng 120.000 IDs tối đa. Metadata-only để bảng riêng; mặc định không tính vào index nội dung. Nếu về sau chọn policy index title-only, lưu flag và báo coverage riêng.

Một encoder/GPU + 4 worker chuẩn bị dữ liệu/GPU; shared CPU budget kiểm soát toàn job. Batch 32, document representation rõ ràng và text hash trước tokenization, model/tokenizer revision cố định. Lưu số token gốc/được encode/truncated. Checkpoint khoảng 2.000 documents; output vector/ID đi cùng nhau, merge theo official ID và kiểm uniqueness/finite/normalization.

Không dùng `r2ai/index.py` nguyên trạng để encode toàn corpus mới: script hiện nhận một device và đọc toàn bộ chunks file. Bổ sung shard input/output, streaming document reader và launcher hai device cho lượt PubMed. Giữ toàn văn/sections để tạo chunks phục vụ retrieval về sau; không coi vector document là thay thế chunks.

Lượt crawl này kết thúc ở bộ PubMed delta đã nghiệm thu (và vector nếu bật). Việc hợp nhất vào chunks, BM25/dense toàn scope và chuyển runtime là bước riêng trong kế hoạch cập nhật tổng thể; không chạy rerank toàn bộ queries để hoàn tất việc thu thập PubMed.

### Bước E — Đối soát và bàn giao

Mỗi official ID có đúng một trạng thái hiện tại; tổng các trạng thái bằng 240.000. Sau hết retry, phân biệt unavailable và chưa xử lý xong; nếu còn pending/retry thì đợt thu thập chưa hoàn tất.

Với records đã nhận: mapping ID–PMID–URL đúng, raw checksum khớp, text/provenance đọc được. PMID thiếu không được im lặng bỏ qua. Metadata-only không được cộng vào usable-abstract coverage. Vector chỉ chứa IDs đủ điều kiện theo policy và khớp text hash.

## 5. Artifacts thực thi cần tạo

Đầu ra dự kiến dưới `data/vibiomir/pubmed/<revision>/<run_id>/`:

- `raw/batch_<id>.xml.gz`, checksum và request/response metadata.
- `state.sqlite`, `fetch_attempts.parquet`, `missing_pmids.parquet`.
- `documents/part_*.parquet` và `metadata_only.parquet`.
- `vectors/shard_<0|1>_*` + official IDs nếu chạy embedding.
- `quality_review.csv`, `coverage.json`, `resource_usage.csv`.
- `release_manifest.json`: source/queue hashes, API/config, model/parser versions và output checksums.

Các artifacts thực thi này chưa tồn tại trong lượt lập kế hoạch; chỉ queue/pilot/config/plan manifest đã được chuẩn bị. Test collector cần kiểm partial batch, mapping lộn thứ tự, lỗi XML trong HTTP 200, metadata-only, atomic resume, duplicate attempts và limiter nhiều worker.

## 6. Thời gian và điểm chốt ETA

| Giai đoạn | Ước lượng trước pilot |
|---|---|
| Viết collector/state, kiểm tra và pilot | Khoảng 0,5–1 ngày công triển khai |
| Tải 240.000 records | Khoảng 1–3 giờ nếu API ổn định, gồm overhead và retry thông thường |
| Parse/quality/shard | Khoảng 0,5–2 giờ với budget CPU đã nêu |
| Encode document representations trên 2 GPU | Khoảng 0,5–1,5 giờ, chưa benchmark A6000 với cấu hình này |
| Đối soát và đóng delta | Khoảng 0,5–1 giờ |

1.200 requests ở trần 1 request/giây có sàn khoảng 20 phút, không phải ETA hoàn tất. Tốc độ còn phụ thuộc latency, response size, queue và retry. Dự trù **3–6 giờ runtime sau khi collector sẵn sàng**, với hai phase parse/GPU ít chồng tải; nếu API có cooldown/recovery thì lâu hơn. **Chỉ lấy raw + text** có thể hoàn tất khoảng 1,5–4 giờ; GPU không cần bật cho phần này.

Sau pilot, ghi records/request, fetch/parse docs/s, encoder docs/s, RAM/VRAM peak và tỷ lệ retry. Chốt ETA bằng số còn lại/throughput thực, không suy ra thời gian crawl từ số GPU. Không tăng worker hoặc request rate tự động để đạt ETA.

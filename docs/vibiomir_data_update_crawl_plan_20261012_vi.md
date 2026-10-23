# Kế hoạch cập nhật, crawl và chuẩn hóa ViBioMIR

Ngày lập: 12/10/2026, giờ Việt Nam. Trạng thái: kế hoạch triển khai; chưa chạy crawl hoặc đổi runtime trong lượt lập kế hoạch.

Mục tiêu: toàn bộ danh mục, dữ liệu thu thập, text, chunks và index cùng trỏ tới một phiên bản corpus chính thức; thu hồi dữ liệu thiếu có kiểm soát; đo được phần chưa thu thập và nguyên nhân của từng URL.

## 1. Mốc dữ liệu và phạm vi

Đã kiểm tra lại API Hugging Face trong lượt lập kế hoạch. Bản hiện tại vẫn là revision `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`, cập nhật ngày 12/10/2026 lúc 20:52:14 giờ Việt Nam.

[Corpus chính thức tại revision cố định](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR/tree/71be0e1239a19dfd7a717e2bc13e3bec21bd0334).

| Dữ liệu đã xác minh | Số lượng |
|---|---:|
| Corpus chính thức | 1.220.598 URL |
| Query, giống từng byte với local | 1.200 |
| Cặp ID–URL chung với raw local | 980.598 |
| URL chính thức thiếu trong raw local, toàn bộ là PubMed | 240.000 |
| URL raw local ngoài danh mục hiện tại | 3.414.120 |
| Documents chính thức có chunks, BM25 ID và dense ID | 833.892 |
| Documents chính thức chưa có chunks/index | 386.706 |
| Documents trong index cũ ngoài danh mục hiện tại | 2.712.046 |
| Chunks cũ nằm trong danh mục hiện tại | 4.261.536 |

Nguồn chính: `outputs/vibiomir_official_verify_20261012/verification.json`. Số liệu chunks/index dựa trên quét đầy đủ ID, chưa chứng minh chất lượng text.

Nguồn chi tiết trạng thái crawl: `outputs/evidence/linux_url_scope_20261012/current_scope_live_states.csv` và `domain_live_coverage.csv`. Đây là snapshot sớm hơn trong ngày; phải làm mới trạng thái trước khi đưa URL vào hàng đợi thực thi.

Mọi con số trong kế hoạch gắn với revision trên. Nếu Hugging Face có revision khác khi triển khai, đối chiếu lại danh mục và lập delta trước khi bắt đầu; không trộn hai revision trong một release.

## 2. Thứ tự công việc và điều kiện nghiệm thu

| Bước | Công việc | Phụ thuộc | Điều kiện chuyển bước |
|---|---|---|---|
| P0 | Chốt corpus, lập trạng thái và phạm vi | — | Chính xác 1.220.598 ID–URL; 1.200 query; checksum đúng |
| P1 | Sửa scope, resume, lịch retry và ghi dữ liệu | P0 | Thử ngắt/chạy lại không mất URL, không bỏ qua file hỏng |
| P2 | Kiểm tra và tái sử dụng dữ liệu còn trong danh mục | P0, P1 | Raw/text/chunk đạt kiểm tra, có nguồn gốc và phiên bản |
| P3A | Thu thập 240.000 bản ghi PubMed | P0, P1, pilot | Mapping PMID–official ID đúng, batch thiếu được đưa lại vào queue |
| P3B | Phục hồi URL Việt/Trung thiếu dữ liệu | P0, P1, pilot từng domain | Có trạng thái cho từng URL; tài liệu hợp lệ đi qua extraction |
| P4 | Chuẩn hóa text/chunks, hợp nhất theo revision | P2; nhận delta từ P3 | Không ID ngoài corpus; không mất text âm thầm |
| P5 | Build BM25/dense và tạo lại candidates | P4 | Index IDs khớp tập documents đạt chất lượng |
| P6 | Kiểm tra đầu cuối, đóng release và chuyển runtime | P5 | Qua điều kiện nghiệm thu; có phiên bản cũ để rollback |

P3A/P3B là hai pipeline độc lập. Có thể chạy đồng thời bằng tiến trình riêng nếu đủ tài nguyên; cùng dùng một cơ sở trạng thái. Extraction chạy theo từng batch hoàn tất, không phải chờ crawl toàn bộ. Dùng một build đầy đủ cuối đợt để nghiệm thu; chỉ build trung gian khi cần baseline so sánh.

## 3. P0 — Chốt corpus và tạo sổ trạng thái

Lưu snapshot vào `data/raw/vibiomir/releases/<revision>/` với tên chính thức `links_corpus_vi_en_zh.parquet`, `query.parquet`, metadata và SHA-256. Truyền đường dẫn/revision rõ ràng cho từng stage; đường dẫn `links_corpus.parquet` hiện tại cần được loại khỏi mặc định cứng hoặc được cấu hình trỏ tới snapshot đã nghiệm thu.

Giữ official ID nguyên bản. Không renumber, không giả định ID liên tục. Giữ `official_url` nguyên văn; URL chuẩn hóa, redirect và `final_url` là các trường phụ. File mới không có cột ngôn ngữ: lưu riêng ngôn ngữ phát hiện từ nội dung và cách phát hiện; host chỉ cung cấp gợi ý.

Tạo `document_state.parquet` hoặc SQLite với khóa `(corpus_revision, official_id)`. Mỗi ID có đúng một trạng thái hiện tại, nhưng giữ lịch sử attempts riêng. Các trường tối thiểu:

- Official ID/URL, hostname, domain dùng chia ngân sách, PMID nếu có.
- Trạng thái fetch/extraction/chunk/index riêng, `attempt_count`, `last_attempt_at`, `next_retry_at`, `last_error`, `run_id`.
- Raw path, raw checksum/kích thước, MIME, HTTP status, final URL, dấu hiệu truncation/challenge.
- Text checksum, extractor version, encoding, content kind, language; chunker/tokenizer revision; index/model version.

Quét lại manifest, kiểm tra file raw, doc IDs của chunks/index. Manifest cũ thiếu timestamp nên lịch sử thứ tự tên file chỉ là suy luận; ghi rõ `timestamp_unknown`, rồi sử dụng timestamp thật cho attempts mới.

Xuất các tập phân biệt: `reuse_candidate`, `needs_extract`, `needs_fetch`, `needs_quality_review`, `outside_current_corpus`. Nhóm `needs_quality_review` có thể phát sinh từ 833.892 documents đã có; không coi số documents đã index là số documents chắc chắn đạt chất lượng.

## 4. P1 — Sửa các lỗi vận hành trước khi crawl lớn

Các điểm dưới đây được đọc trực tiếp từ code hiện có:

| Vị trí | Hiện trạng | Việc cần làm |
|---|---|---|
| `r2ai/crawl.py:load_targets` | Đọc cứng corpus cũ | Nhận corpus/queue/revision; chỉ xử lý ID–URL đúng snapshot |
| `r2ai/crawl.py:already_done` | File `.gz` tồn tại được coi là đã hoàn tất | Kiểm tra gzip, checksum, nội dung và mapping trước khi bỏ qua |
| `r2ai/crawl.py:DomainLimiter.wait` | Các coroutine có thể cùng thức dậy sau khoảng chờ | Giữ khóa khi đặt lịch request; kiểm thử tốc độ thực tế theo domain |
| `r2ai/crawl.py:CircuitBreaker` | Mở circuit tới hết lượt, hàng loạt URL bị ghi `circuit_open` | Cooldown, half-open probe, hàng đợi hoãn và thời gian retry; không coi đây là URL đã được request |
| `r2ai/deadhosts.py` | Host trong danh sách được bỏ qua cả khi retry failed | Thêm ngày kiểm tra/hạn hiệu lực; probe lại trước khi tiếp tục bỏ qua |
| `r2ai/crawl.py:fetch_one` | Chạm 6 MiB có thể cắt body nhưng vẫn ghi `ok` | Phân biệt hoàn tất/too_large/truncated; chỉ raw hoàn tất được dùng |
| `r2ai/crawl.py:fetch_one` | Ghi raw trực tiếp; schema thiếu thời gian và checksum | Ghi `.tmp`, đóng file/kiểm checksum, rename rồi checkpoint; thêm provenance |
| `r2ai/extract.py:main` | Gom tất cả ID từng có manifest `ok` | Nhận danh mục ID được phép và queue delta; không extraction ngoài scope |
| `r2ai/extract.py:process` | Cap chunks và ngưỡng tối thiểu có thể bỏ nội dung | Lưu toàn bộ text; ghi số token bị loại; nội dung thiếu đi vào review |
| `r2ai/langbalance.py` | Host heuristic không có nhóm EN riêng | Thêm EN/unknown từ metadata và kiểm tra nội dung; đo lại thống kê trước khi dùng correction |

Chuẩn hóa trạng thái: `pending`, `in_progress`, `raw_valid`, `extract_failed`, `quality_rejected`, `indexed`, `retry_scheduled`, `unavailable`, `blocked`, `robots_denied`, `metadata_only`. Giữ trạng thái network riêng với trạng thái nội dung. HTTP 200 không đủ chứng minh tải được bài viết.

Chính sách retry đề xuất: lỗi timeout/network/5xx dùng exponential backoff có jitter; 429 tôn trọng `Retry-After` và hạ tốc độ domain; 404/410 xác nhận rồi ghi unavailable; challenge/403 chuyển sang phục hồi domain; lỗi robots tạm thời được hoãn để kiểm tra lại. Tối đa 3 retry ngắn và 2 lượt phục hồi có khoảng chờ là điểm khởi đầu, điều chỉnh sau pilot. Không giữ worker trong thời gian chờ dài.

Bổ sung test có ý nghĩa: file hỏng nhưng còn tên, ngắt giữa write và manifest, URL đổi nhưng ID giữ nguyên, batch trả thiếu PMID, duplicate attempts, circuit half-open và tốc độ request khi nhiều worker. Đây là kiểm tra thay đổi crawler/state, không phải kiểm tra lại những phép đếm đã xác minh.

## 5. P2 — Tái sử dụng có kiểm tra

Lọc candidate tái sử dụng theo cặp ID–URL chính thức. 833.892 documents/4.261.536 chunks là điểm xuất phát tối đa từ bản hiện có, không phải cam kết tất cả đạt chất lượng.

Đọc raw trong scope để kiểm tra gzip, body bị cắt, trang challenge, error page, redirect sang trang chủ và encoding. Có raw hợp lệ thì extraction lại khi cần; chỉ fetch lại raw thiếu/hỏng/nội dung không dùng được. Lấy mẫu thủ công phân tầng theo domain, VI/ZH, độ dài và phương pháp extraction; phát hiện lỗi mang tính hệ thống thì rà toàn bộ nhóm tương ứng.

Giữ chunk/text cũ đã đạt chất lượng làm baseline. Khi thay extractor hoặc chunker, tạo version mới và checksum mới; không sửa text nhưng vẫn dùng cache reranker/embedding cũ. Các archive ngoài scope được tách khỏi runtime; không cần xóa để hoàn thành cập nhật corpus.

## 6. P3A — Thu thập PubMed

Tập đầu vào là `outputs/vibiomir_official_verify_20261012/official_missing_locally.parquet`: đúng 240.000 ID–URL. Parse PMID từ URL và lưu mapping official ID → PMID; PMID không thay official ID.

Ưu tiên EFetch với `db=pubmed`, `retmode=xml`, POST và batch khởi đầu 200 PMID. Nếu toàn bộ PMID duy nhất, có khoảng 1.200 batch trước retry. API NCBI hỗ trợ lấy tập UID và khuyến nghị POST khi danh sách vượt khoảng 200 UID. [NCBI E-Utilities](https://eutilities.github.io/site/Quick_Start/eu_quick/).

Khởi đầu 2 request/giây tổng cho tiến trình không có key; nếu có key, có thể thử 5 request/giây. Limiter phải tính tổng các worker/máy dùng chung IP/key. Mức mặc định NCBI là 3 request/giây khi không có key và 10 khi có key. [Chính sách API key NCBI](https://ncbiinsights.ncbi.nlm.nih.gov/2017/11/02/new-api-keys-for-the-e-utilities/). Cấu hình `tool`, email liên hệ và key qua môi trường; key là tùy chọn, không cản việc bắt đầu pilot.

Mỗi batch giữ raw XML, checksum, requested/returned/missing PMIDs. Ghép bằng PMID trong response, tuyệt đối không ghép theo vị trí. Batch HTTP 200 nhưng thiếu bản ghi vẫn phải requeue ID thiếu; response XML lỗi không được coi là thành công.

Extraction riêng cho XML: title, AbstractText theo label và đúng thứ tự, văn bản inline, thông tin ngôn ngữ/PMID/DOI/PMCID. Giữ abstract thành text gốc có section; metadata là trường riêng. Không đưa XML qua extractor HTML hiện tại.

PubMed cung cấp citation và abstract; full text nằm ở nguồn khác/PMC khi có. [PubMed Help](https://pubmed.ncbi.nlm.nih.gov/help/#finding-the-full-text-article). Bản ghi không có abstract được ghi `metadata_only`, không bịa phần nội dung thiếu. So sánh API với trang official trên mẫu để đánh giá khác biệt nội dung. Trước khi đưa full text PMC vào dữ liệu chấm, kiểm tra hướng dẫn BTC về phạm vi văn bản/nguồn; lưu PMC thành lớp riêng với provenance. Pilot abstract/XML vẫn triển khai độc lập với phần mở rộng này.

Pilot đề xuất: 1.000 PMID phân tầng theo năm/ID; đối chiếu thủ công 30 bản ghi gồm structured abstract, không abstract và Unicode. Nghiệm thu 100% mapping PMID–official ID; mọi PMID thiếu có nguyên nhân/retry; không HTML challenge lọt vào text. Sau đó chạy đủ 240.000 và đối soát toàn bộ.

## 7. P3B — Phục hồi các website Việt/Trung

Thứ tự theo lượng thiếu và khả năng tái sử dụng. Các nhóm domain dưới đây không chồng lặp:

| Nhóm | Documents còn thiếu chunks/index | Hành động |
|---|---:|---|
| PubMed, xử lý ở P3A | 240.000 | API batch/XML |
| `nhathuoclongchau.com.vn` | 78.075 | Pilot chậm; xử lý rate/challenge, sau đó crawl phục hồi |
| `ask.39.net` | 32.760 | Kiểm tra extraction raw đã có và ngân sách chung toàn bộ `39.net` |
| Domain còn lại | 35.871 | Xếp theo gap, lỗi và kết quả pilot |
| Tổng | 386.706 | Đối soát trước khi enqueue |

Snapshot crawl cho thấy 222 documents trong corpus có raw nhưng chưa có chunks. Xử lý nhóm này trước vì không cần tải lại nếu raw hợp lệ. **222 là tập con của các nhóm trên**, không cộng vào 386.706.

Snapshot ghi 139.638 documents thiếu index có trạng thái cuối `circuit_open`. Đây là dấu hiệu scheduler đã hoãn/bỏ qua theo host, không phải bằng chứng từng URL không tồn tại. Kiểm tra trạng thái này và danh sách dead hosts trước khi retry hàng loạt.

Pilot 100–300 URL/domain ưu tiên, trải trên các nhóm path. Điểm khởi đầu: Long Châu 1 request đang chạy và khoảng cách 2 giây; nhóm `39.net` 1–2 request đang chạy, khoảng cách 1–2 giây dùng chung cho tất cả subdomain. Hạ tốc độ khi phát hiện 429/challenge, mở rộng sau khi pilot ổn định. Đây là cấu hình thử, không phải giới hạn đã xác minh của website.

Ưu tiên các nhóm nhỏ có raw hợp lệ hoặc host khỏe trước; không để host đang cooldown chiếm worker. Với HTML cần JS, chỉ đưa vào queue render khi đã kiểm tra trang gốc cần render và phương thức truy cập cho phép; tránh bật browser cho toàn bộ 1,22 triệu URL. Giữ robots và trạng thái chặn; ghi rõ phần không truy cập được.

Mỗi domain báo cáo riêng: queued, attempted thật, raw_valid, extracted, indexed, rate_limited, blocked, unavailable, deferred; throughput tính theo raw/text hợp lệ, không theo HTTP 200.

## 8. P4 — Chuẩn hóa text và chunks

Tách bảng `documents` chứa text đầy đủ khỏi bảng `chunks`. Giữ raw/text gốc cùng checksum. Clean tối thiểu khoảng trắng và markup; giữ câu, số, đơn vị, ký hiệu y sinh, dấu tiếng Việt và nội dung Trung/Anh. Không dịch hoặc sinh lại text làm evidence.

HTML ưu tiên main article, thêm adapter theo template đối với domain lớn. Kiểm tra boilerplate, menu/related articles, text rỗng, mojibake, title/body bất hợp lý. PDF/XML dùng extractor đúng loại. Ngưỡng text ngắn tạo nhãn review, không tự loại title/abstract ngắn hợp lệ.

Lưu các document ID có nội dung trùng; khử trùng storage bằng content hash vẫn phải giữ mapping official ID. Không gộp khác ID thành một document chỉ vì text giống nhau. Chunk trùng trong cùng document có thể xử lý theo policy rõ ràng và có thống kê.

Baseline migration giữ chunk/text cũ đã đạt kiểm tra. Đối với dữ liệu mới/re-extract, đo token thật bằng tokenizer BGE-M3 có revision cố định; chunk theo câu/đoạn và có offsets vào text lưu. Cấu hình thử 384 tokens, overlap 64; so với cấu hình hiện tại trên pilot rồi mới chốt. BGE-M3 có tokenizer cho nhiều ngôn ngữ; không chỉ đếm khoảng trắng để sizing Chinese. Cấu hình thử không phải chunk size chính thức của BTC.

Kiểm tra độ dài theo encoder/reranker thực tế, chừa chỗ cho title/query/special tokens. Đo truncation ở cả document representation và chunk reranker. Khi cap chunks ảnh hưởng nội dung, ghi rõ coverage/tail bị loại; giữ toàn văn để xử lý tiếp. Không drop đoạn dưới ngưỡng hoặc vượt cap âm thầm.

Khóa chunk gồm official ID, document/text version, chunker version và ordinal. Xuất adapter schema hiện dùng `(doc_id, chunk_index, title, text, n_words)` để pipeline tương thích; sidecar giữ token count thật, offsets và provenance. Kiểm tra full key uniqueness, nonempty text, offsets khớp và IDs trong corpus.

## 9. P5 — Index và cache theo cùng phiên bản

Build BM25 mới trên documents/chunks đạt chuẩn trong scope để tính lại vocab, document frequency và độ dài corpus. Không chỉ sửa file companion IDs của BM25 cũ.

Dense có thể tái sử dụng vector nếu chứng minh cùng official ID, đúng text representation hash, model/tokenizer revision, normalization và cấu hình truncation. Nếu thiếu provenance hoặc thay text, encode lại. Tái sử dụng phải lọc cả hàng vector và ID tương ứng, rồi xây index mới. Encode riêng documents bổ sung; batch GPU dựa trên pilot VRAM, không đưa một mức cố định chưa đo.

Đảm bảo `set(index_doc_ids) == set(eligible_document_ids)` theo policy từng index; không có IDs ngoài corpus. Kiểm tra số hàng/chiều vector, finite values, normalization, sort/mapping và khả năng lookup text.

Tạo lại BM25/dense candidates, fusion, shortlist chunks và reranker cache. Khóa cache phải chứa corpus revision, document/chunk checksum, model/settings; query giống nhau không đủ để reuse candidates khi corpus đổi.

Mọi retrieval phải loại ID ngoài scope **trước top-k**. Cách lấy top-k từ index cũ rồi lọc sau có thể thiếu candidates hợp lệ. Nếu backend không hỗ trợ lọc trước top-k, dùng index mới đã lọc.

Kiểm tra PubMed tiếng Anh qua các tầng candidate/reranker; giữ cấu hình correction ngôn ngữ tắt cho tới khi có đo lường mới. Không áp quota VI/EN/ZH chỉ để ép kết quả bằng tỷ lệ corpus.

## 10. P6 — Nghiệm thu và cập nhật tiếp theo

Điều kiện bắt buộc trước chuyển runtime:

1. Snapshot chứa chính xác 1.220.598 cặp ID–URL và 1.200 query, checksum đúng revision.
2. Mỗi official ID có một trạng thái hiện tại và lịch sử xử lý; không mất URL do circuit/resume. Nếu không thu được nội dung, giữ trạng thái và nguyên nhân rõ ràng.
3. Documents/chunks/index/candidates không chứa ID ngoài scope, ID trùng không hợp lệ hoặc mapping sai.
4. Raw đưa vào extraction phải hợp lệ; chunk đưa vào index không rỗng/challenge/truncated chưa xử lý; provenance nối được đến raw.
5. Tập ID của từng index bằng tập documents đủ điều kiện theo policy đã công bố; không dùng failed/metadata-only để giả tăng coverage nội dung.
6. Chạy toàn bộ 1.200 query; mọi result có URL/doc/chunk hợp lệ và không dùng cache phiên bản khác. Query không có kết quả phải có báo cáo nguyên nhân.
7. So sánh baseline cùng scope với bản bổ sung; báo cáo document coverage theo domain/ngôn ngữ, content quality và retrieval riêng. Nếu có qrels/feedback, đo trên nhãn đúng phạm vi và tách tập chọn cấu hình khỏi tập đánh giá.
8. Build vào thư mục version mới, đóng manifest/checksum/config/model revisions, rồi đổi runtime pointer có thể rollback. Corpus snapshot thống nhất với mọi file index/chunk.

Mục tiêu thu thập là bao phủ tối đa 1.220.598 documents, nhưng không cam kết website/abstract luôn tồn tại. Có thể đặt mốc vận hành 95% usable-document coverage sau pilot; đây là mục tiêu nội bộ, không phải chuẩn BTC. Bản bàn giao phải công bố số usable, metadata-only, unavailable, blocked và retry còn lại. Không kết luận coverage URL là gold recall.

Kiểm tra metadata Hugging Face trước mỗi đợt chạy/release. Khi revision thay đổi: diff ID–URL → thêm/đổi ID thì enqueue → URL giữ nguyên tái sử dụng theo chất lượng/freshness → ID bị loại tách khỏi active scope → rebuild index/caches cần thiết. Khi chỉ README đổi nhưng hash corpus/query giữ nguyên, cập nhật provenance mà không crawl/rebuild dữ liệu vô ích.

## 11. Đầu ra dự kiến và điểm triển khai

Các đầu ra sau **chưa được tạo/chạy** trong lượt này; đây là hợp đồng triển khai:

| Đầu ra | Nội dung |
|---|---|
| `release_manifest.json` | Revision, input/output hashes, config và model versions |
| `document_state` + `fetch_attempts` | Một trạng thái hiện tại/ID, lịch sử có timestamp |
| `crawl_queue.parquet` | Official ID–URL, priority, reason, retry time |
| `documents.parquet` | Text đầy đủ, kind/language, hashes, extraction provenance |
| `chunks_scope_<version>.parquet` + sidecar | Chunks tương thích runtime, offsets/token provenance |
| `missing_documents.parquet` | Toàn bộ ID chưa có text/index và nguyên nhân |
| BM25/dense release | Index và ID/vector mapping chỉ trong scope |
| `coverage_report.csv` | Phễu corpus→fetch→text→chunk→index theo domain/language |
| `quality_review.csv` | Mẫu rà thủ công, lỗi, quyết định tái sử dụng/tái xử lý |
| `migration_validation.json` | Kết quả các điều kiện nghiệm thu |

Ưu tiên sửa `r2ai/crawl.py`, `r2ai/deadhosts.py`, `r2ai/extract.py`; thêm collector/extractor PubMed và trạng thái version; sau đó sửa `r2ai/bm25index.py`, `r2ai/index.py`, language metadata và cache/runtime. Các scripts probe/pilot hiện có có thể dùng để kiểm tra nhỏ, nhưng không thay thế crawler production đang xử lý hàng triệu URL. Collector PubMed cần schema/raw adapter riêng.

## 12. Lịch dự kiến và cách ước lượng

Giả định có CPU cho fetch/extraction và GPU chạy dense/rerank theo lịch sẵn có. Chưa đo công suất/VRAM trong lượt lập kế hoạch.

| Mốc | Ước lượng ban đầu |
|---|---|
| Ngày 1 | P0/P1, queue/state, kiểm tra reuse, pilot PubMed/Long Châu/39.net |
| Ngày 2–3 | Thu PubMed, extraction theo batch, phục hồi các domain có pilot đạt |
| Ngày 3–5 | Tiếp tục recovery, hợp nhất text/chunks, build index và nghiệm thu |

Đây là lịch sơ bộ, phải điều chỉnh theo pilot và blocking của website. Ví dụ Long Châu có 78.075 URL thiếu: nếu tất cả cần fetch và tốc độ thực 0,5 URL/giây, riêng fetch có sàn khoảng 43,4 giờ chưa tính retry. Số URL cần fetch thực tế được xác định sau kiểm tra raw. PubMed 240.000 records/batch 200 tương đương khoảng 1.200 requests; thời gian thực còn download, parsing và retry.

Sau pilot ghi requests/giây, usable docs/giây, MB/doc, tokens/doc, CPU/GPU throughput và peak RAM. Ước lượng `remaining_docs / usable_docs_per_second` cho từng domain; tổng thời gian phụ thuộc nhánh chậm nhất và lịch GPU. Ước lượng storage từ bytes/chunks/vectors quan sát, kiểm dung lượng cho staging + release cũ trước batch lớn.

Việc triển khai đầu tiên nên hoàn thành P0/P1 và pilot, rồi công bố backlog đã làm mới cùng ETA đo được. Các host còn chặn kéo dài được báo cáo riêng, không chặn việc xây bản release từ những nguồn đã hoàn tất và đạt chuẩn.

## 13. Ước lượng khi dùng 4 RTX A6000

Cập nhật yêu cầu: đợt thu thập riêng 240.000 PubMed hiện chọn **2 GPU, 4 worker chuẩn bị dữ liệu/GPU, tối đa 8**. Kế hoạch thực hiện cho đợt này nằm ở [pubmed_240k_crawl_plan_2gpu_20261012_vi.md](pubmed_240k_crawl_plan_2gpu_20261012_vi.md). Phần 4 GPU dưới đây giữ làm ước lượng tham khảo cho kế hoạch tổng thể trước đó.

Người dùng xác nhận cấu hình dự kiến là 4 RTX A6000. Mỗi card có 48 GB VRAM theo [NVIDIA](https://www.nvidia.com/en-us/products/workstations/rtx-a6000/). Chưa benchmark trực tiếp cấu hình này; `nvidia-smi` trong môi trường phiên hiện tại không kết nối được driver, nên không xác nhận được GPU thực đang sẵn có.

Log `outputs/dense_fuse_crawl_20261007.log` ghi build dense trên một physical GPU: 3.545.938 documents, batch 128, embedding loop mất 3:03:49, tương đương khoảng 321,5 documents/giây. Loại GPU trong lượt đó chưa xác minh. Đây là số đo lịch sử, không phải benchmark RTX A6000.

Nếu throughput mỗi A6000 tương đương lượt cũ, giữ nguyên độ dài input/config và chia đều dữ liệu, hiệu quả 4 GPU giả định 70–85% thì encode 1.220.598 document representations mất khoảng 19–23 phút thuần GPU. Chỉ 386.706 documents bổ sung mất khoảng 6–7 phút thuần GPU. Các phép ngoại suy chưa tính đọc/chuẩn bị text, model startup, write/merge và khác biệt nội dung; dùng ngân sách 0,5–2 giờ cho stage dense document-level, rồi hiệu chỉnh bằng pilot. Đổi sang input dài hơn hoặc embedding toàn bộ chunks làm khối lượng thay đổi và phải tính lại.

Pipeline hiện dùng `device` đơn trong `r2ai/index.py`; có 4 card không tự làm code chạy trên 4 card. Cần chia shard input, một worker/model trên mỗi GPU, merge ID/vector đúng thứ tự và kiểm tra không thiếu/trùng. Rerank chia theo query/batch, không chạy lặp toàn bộ workload trên mỗi card.

| Phần việc | Ngân sách sơ bộ với cấu hình 4 GPU |
|---|---|
| Chuẩn bị scope, sửa pipeline và pilot | 0,5–1 ngày công triển khai |
| PubMed API/XML | 1–4 giờ khi collector sẵn sàng, điều chỉnh theo missing/retry |
| Website Việt/Trung còn thiếu | Khoảng 36–72 giờ nếu host truy cập được; chặn kéo dài có thể lâu hơn |
| Extraction/chunking phần bổ sung | Khoảng 2–8 giờ CPU, chạy theo batch song song với crawl; phụ thuộc CPU/I/O |
| Dense document-level | 0,5–2 giờ; budget dự phòng, chưa benchmark A6000 |
| BM25 và đối soát release | Khoảng 2–6 giờ CPU/I/O, tùy khối lượng phải tái xử lý |
| Rerank một lượt 1.200 queries | Budget 1–6 giờ nếu 200–400 candidate chunks/query; cần pilot theo model/token length |

Các hàng trên không cộng trực tiếp vì crawl, extraction và encode delta có thể chồng thời gian. Dự trù **3–5 ngày cho toàn bộ kế hoạch gồm sửa code và nghiệm thu**, hoặc **khoảng 2–4 ngày runtime sau khi pipeline sẵn sàng**, với CPU/I/O đủ và các domain phục hồi được. Bốn GPU giảm stage ML; bottleneck chính vẫn là tốc độ request và blocking của website. Không cam kết thu được nội dung cho 100% URL chỉ dựa vào số GPU.

Để chốt ETA: benchmark 10.000 document representations trên cả 4 card và khoảng 10.000 query–chunk pairs; chạy pilot web ít nhất đủ lâu để thấy rate/challenge. Báo throughput usable documents, GPU utilization, peak CPU RAM/I/O và tỷ lệ retry. Không thuê/chạy cả 4 GPU trong toàn bộ thời gian crawl nếu không có batch text sẵn để encode.

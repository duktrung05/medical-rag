# `r2ai` — pipeline cho R2AI2026 ViBioMIR

Package này thay thế pipeline ViMedAQA cũ trong `src/` (xem phân tích ở
[`docs/vibiomir_strategy_vi.md`](../docs/vibiomir_strategy_vi.md) về lý do).

## Môi trường

```bash
python3 -m venv --system-site-packages .venv-r2ai
.venv-r2ai/bin/pip install bm25s PyStemmer selectolax trafilatura rapidfuzz \
                           sentence-transformers faiss-cpu
export HF_HOME="$PWD/.cache/huggingface"
```

> **Máy dùng chung:** kiểm tra `nvidia-smi` trước khi chạy — thường chỉ 1 trong 4 card
> A6000 trống. Truyền `--device cuda:<n>` cho card rảnh.

## Dữ liệu gốc

```bash
mkdir -p data/raw/vibiomir && cd data/raw/vibiomir
curl -L -O https://huggingface.co/datasets/AIGuruTinix/ViBioMIR/resolve/main/query.parquet
curl -L -O https://huggingface.co/datasets/AIGuruTinix/ViBioMIR/resolve/main/links_corpus.parquet
```

## Chạy pipeline

```bash
# 1. Ứng viên từ text trong URL slug — không cần crawl, dùng để ưu tiên crawl
.venv-r2ai/bin/python r2ai/candidates.py build
.venv-r2ai/bin/python r2ai/candidates.py retrieve --top-k 60

# 2a. Crawl tập ứng viên (~52k) để có submission nhanh
.venv-r2ai/bin/python -u r2ai/crawl.py --source candidates \
    --concurrency 128 --per-domain 6 --delay 0.1 --retries 1 --circuit-threshold 30

# 2b. Crawl TOÀN BỘ 4,4M URL (chạy dài, resume được — cứ chạy lại nếu đứt)
#     --per-domain-cap 48 cho phép các domain TQ lớn (cnkang 963k, 120ask 918k)
#     chạy song song mạnh hơn; nếu không wall-time sẽ bị chúng quyết định.
#     Dự trù: ~950 GB HTML thô (~200 GB sau gzip), 8-35 giờ tuỳ băng thông.
.venv-r2ai/bin/python -u r2ai/crawl.py --source all \
    --concurrency 256 --per-domain 6 --per-domain-cap 48 \
    --delay 0.08 --retries 1 --circuit-threshold 30

# 3. Bóc nội dung + chunk hoá (--target-words là biến A/B quan trọng nhất)
.venv-r2ai/bin/python r2ai/extract.py --target-words 220 --workers 28 --out chunks.parquet

# 4. Retrieve + rerank + xuất submission
.venv-r2ai/bin/python -u r2ai/rank.py --chunks-file chunks.parquet \
    --k-docs 10 --k-chunks 10 --pool 200 --device cuda:1 --out sub_v1

# 5. BẮT BUỘC: validate trước khi upload
.venv-r2ai/bin/python r2ai/validate.py outputs/submissions/sub_v1.zip
```

Upload `outputs/submissions/sub_v1.zip` tại mục **My Submissions** trên
`leaderboard.aiguru.com.vn`.

## Hiệu chỉnh cutoff từ leaderboard

Sau mỗi lần nộp, ghi 7 chỉ số vào `outputs/submissions/log.csv`, rồi:

```bash
.venv-r2ai/bin/python r2ai/calibrate.py \
    --probe 5,0.40,0.35 --probe 10,0.28,0.50 --probe 30,0.13,0.70
```

Công cụ suy ra `|G|` (số doc liên quan thật mỗi query) từ cặp Precision/Recall và
tính `K` tối ưu cho F2 — thay vì mò.

## Final evidence selector sau reranker

`rank.py` mặc định vẫn xuất Top-K. Chế độ `threshold` chỉ chọn chunk có score
thật từ cross-encoder, cho phép trả rỗng và không bù chunk dưới ngưỡng để đủ K.
Score là **raw logit** với activation `Identity`, không phải xác suất. Chunk ở
đuôi khi dùng `--rerank-top` có `rerank_score=null` và không đủ điều kiện chọn.
Taxonomy chỉ quyết định thứ tự ưu tiên; không thay thế relevance score.

Chạy ranking một lần và lưu toàn bộ candidate pool cùng text gốc, score và
fingerprint đầu vào. Dùng tệp chunk/candidates đang có trong workspace:

```bash
.venv-r2ai/bin/python -u r2ai/rank.py \
    --chunks-file chunks_v4.parquet --candidates fused_candidates.parquet \
    --pool 200 --rerank-top 200 --device cuda:1 \
    --k-docs 10 --k-chunks 10 --out selector_baseline \
    --ranking-cache outputs/evidence/rankings.jsonl.gz
```

Model revisions được pin; embedding cache mới gắn với SHA-256 của chunk,
query/candidates và cấu hình encoder. Cache cũ chỉ có tên file được giữ lại;
lần chạy đầu bằng luồng mới sẽ tính embedding lại. Cache ranking chứa đủ dữ
liệu để replay, không cần đọc corpus hoặc import Torch.

Thử ngưỡng bằng CPU (giá trị `0` dưới đây chỉ là ví dụ **chưa hiệu chỉnh**):

```bash
python3 r2ai/select_evidence.py --rankings outputs/evidence/rankings.jsonl.gz \
    --selector threshold --chunk-threshold 0 --k-docs 10 --k-chunks 10 \
    --chunks-per-doc 1 --document-mode parents \
    --output outputs/evidence/threshold_example

.venv-r2ai/bin/python r2ai/validate.py \
    outputs/evidence/threshold_example/submission.zip
```

Các chế độ document:

- `parents`: chỉ xuất document cha của chunk được chọn. Đây là mặc định cho selector.
- `baseline`: giữ document Top-K để thử riêng ảnh hưởng của chunk filtering.
- `threshold`: lọc document bằng score lớn nhất của các chunk đã được rerank;
  `--doc-threshold` mặc định dùng chunk threshold. `--doc-delta` thêm relative gate.

`--chunk-delta` giới hạn khoảng cách với **score cao nhất thực tế**, kể cả khi
taxonomy đã đổi thứ tự. Giới hạn `--k-docs`, `--k-chunks`, `--chunks-per-doc` là
giới hạn trên; không có `min_k`. Text được giữ nguyên, chunk trùng text hoàn toàn
được loại trong chế độ threshold. Trong `select_evidence.py`, `--k-chunks 0` thực
sự trả không chunk; riêng CLI `rank.py` cũ vẫn dùng `0` để lấy K bằng `--k-docs`.
Có thể dùng `--selection-config` JSON để đặt giới hạn bằng 0 trong cả hai luồng.

Replay tạo `submission.zip`, `selection_config.json`, `summary.json` và
`diagnostics.jsonl` ghi lý do chọn/loại từng chunk. Chọn đường dẫn cache/output
mới cho mỗi thí nghiệm; replay không ghi đè thí nghiệm cũ.

### Nhãn dev/holdout và hiệu chỉnh

Chưa có nhãn local cho ViBioMIR thì xuất mẫu để gán nhãn thủ công:

```bash
python3 r2ai/calibrate_evidence.py --rankings outputs/evidence/rankings.jsonl.gz \
    --prepare-labels outputs/evidence/judgments.jsonl --sample-size 150 --seed 42
```

Mỗi dòng chứa query và toàn bộ candidates của query đó, với `relevant=null`.
Điền **tất cả** thành `true/false`, rồi chia thành hai file dev/holdout có query ID
không giao nhau. Recall từ mẫu này là **recall trong candidate pool**, không phải
recall toàn corpus. Công cụ từ chối nhãn chưa hoàn thành hoặc lệch text/candidate.

Nếu có ground truth đầy đủ, dùng JSONL theo dạng sau và khai báo đúng phạm vi:

```json
{"id": 1, "label_scope": "corpus", "relevant_docs": [123], "relevant_chunks": [{"doc_id": 123, "chunk_index": 0}]}
```

`chunk_index` phải thuộc đúng phiên bản chunk được lưu trong cache. Đây là
evaluator nội bộ theo ID chunk, không mô phỏng luật matching chunk text của
scorer chính thức. Nhãn chỉ cho candidate pool dùng `label_scope=candidate_pool`.

```bash
python3 r2ai/calibrate_evidence.py --rankings outputs/evidence/rankings.jsonl.gz \
    --ground-truth outputs/evidence/dev.jsonl \
    --holdout-ground-truth outputs/evidence/holdout.jsonl \
    --baseline-k-docs 10 --baseline-k-chunks 10 \
    --max-docs 5,10,20 --max-chunks 5,10,20 \
    --chunk-deltas none,1,2 --max-recall-drop 0.02 \
    --output outputs/evidence/calibration

python3 r2ai/select_evidence.py --rankings outputs/evidence/rankings.jsonl.gz \
    --selection-config outputs/evidence/calibration/best_config.json \
    --output outputs/evidence/selected
```

Grid gồm cả Top-K nhỏ hơn và các selector, chọn cấu hình tăng chunk precision,
giữ document/chunk recall trong mức giảm cho phép và không giảm internal macro F2.
Ngưỡng tự động lấy từ score **dev**, holdout chỉ kiểm tra cấu hình đã chọn.
Nếu không có phương án đạt điều kiện, `best_config.json` giữ baseline và báo
`no_improving_configuration`. Không bỏ query trả rỗng khi lấy macro trung bình;
precision/recall/F2 của prediction rỗng bằng 0.

Báo cáo gồm `all_results.csv`, `report.json`, `report.md`, `dev_details.jsonl`
và `holdout_details.jsonl` nếu có holdout. Kết quả dev ghi rõ positive bị loại và
false positive được bỏ so với baseline. Chỉ dùng cấu hình với đúng corpus/model
đã hiệu chỉnh; đối chiếu fingerprint và revision trong báo cáo.

`calibrate.py` cũ vẫn dành cho thí nghiệm K cố định. Không dùng công thức một K
để suy gold-set size khi selector trả số evidence khác nhau giữa các query.

### ZIP thử nghiệm từ baseline đã có

Để đo riêng tác động của chunk filtering, `test_evidence_submission.py` nhận ZIP
baseline thực tế, đối chiếu evidence với corpus gốc, chấm lại bằng reranker thật,
lưu cache/checkpoint rồi chạy selector và validator. Không cần tính lại embedding.
Chế độ document `baseline` giữ danh sách document của bản gốc; query có thể có
document được truy hồi nhưng không có chunk tương ứng trong pool chấm lại.

```bash
python -u r2ai/test_evidence_submission.py \
    --baseline outputs/submissions/vibiomir_test_best_20261006.zip \
    --chunks-file data/vibiomir/chunks_v4.parquet \
    --output outputs/evidence/test_run \
    --device cuda:1 --batch 16 --queries-per-block 20 \
    --document-mode baseline --max-docs 200 --max-chunks 120 --threshold -6
```

Ngưỡng `-6` ở đây chỉ minh họa, chưa có nhãn hiệu chỉnh. File nộp nằm ở
`outputs/evidence/test_run/selected/submission.zip`; các checkpoint cho phép tiếp
tục inference sau khi bị gián đoạn. `--prepare-only` chỉ đối chiếu corpus và lưu
pool chưa chấm, không cần Torch. Có thể dùng `select_evidence.py` để replay ngưỡng
khác từ `rankings.jsonl.gz` mà không chạy lại model. Bản thử nghiệm này chỉ xét
những evidence đã nằm trong ZIP baseline, không mở rộng candidate pool.

## Ghi chú vận hành

- **`resp.content.read(n)` của aiohttp KHÔNG đọc đủ n byte** — nó chỉ trả về phần đã
  nằm sẵn trong buffer. Lỗi này cắt cụt mọi trang ở chunk mạng đầu tiên: median
  **20 KB thay vì 217 KB**, khiến ta index phần header/CSS thay vì nội dung bài viết.
  Tỉ lệ bóc được nội dung dùng được: **44% → 99,5%** sau khi chuyển sang
  `iter_chunked()`. Triệu chứng dễ chẩn đoán nhầm: trang trông như "render bằng JS,
  không có body" — thực ra body chưa kịp về. Luôn so kích thước với một lần fetch mới.
- **Crawler phải interleave theo domain.** Thứ tự ứng viên gom nhiều URL cùng host
  cạnh nhau, làm slot concurrency toàn cục bị kẹt sau giới hạn per-domain:
  **4 req/s → 63 req/s** sau khi sửa.
- **Backoff phải nằm NGOÀI semaphore của domain**, nếu không vài host bị throttle sẽ
  giữ slot trong lúc ngủ và làm đói toàn bộ crawl.
- **Circuit breaker là bắt buộc**: 11.614 URL của host bị Cloudflare chặn sẽ ngốn
  ~13 giờ retry vô ích. Sau 30 lần lỗi liên tiếp, host bị bỏ qua.
- **Header browser-realistic là bắt buộc**: nhiều host 403 mọi UA lạ (97,5% thành công
  sau khi đổi). robots.txt vẫn được tôn trọng, rate limit vẫn giữ.
- **selectolax parse hỏng âm thầm** trên một số trang (suckhoedoisong.vn): trả về cây
  rỗng không báo lỗi. Đã chuyển fallback sang lxml (yield 47% → 85%).
- **`zysjonline.com` (243k URL) đứng sau Cloudflare JS challenge** (`cf-mitigated: challenge`),
  không phải lọc UA đơn thuần. Hiện bỏ qua — nên hỏi BTC vì chính họ cung cấp URL này.
  `bingli.iiyi.com` trả 521 (origin down).
- Mọi bước đều **resume được**; chạy lại `crawl.py` sẽ bỏ qua URL đã có trong manifest.

## Module

| Module | Vai trò |
|---|---|
| `urltext.py` | Bóc text từ URL: giải %-encoding, bỏ dấu tiếng Việt, giữ CJK |
| `candidates.py` | BM25 trên slug của 1,34M URL → shortlist không cần crawl |
| `crawl.py` | Crawler async: giới hạn per-domain, robots.txt, resume, manifest parquet |
| `extract.py` | trafilatura + fallback lxml; đóng gói chunk theo `--target-words` |
| `rank.py` | BGE-M3 dense + bge-reranker-v2-m3 → JSON + ZIP đúng format |
| `evidence_selector.py` | Lọc bằng score thật, giới hạn số lượng và cho phép trả rỗng |
| `evidence_io.py` | Cache ranking có fingerprint và IO submission cho CPU replay |
| `select_evidence.py` | Replay Top-K/selector từ cache, xuất ZIP và diagnostics |
| `calibrate_evidence.py` | Xuất mẫu gán nhãn, hiệu chỉnh trên dev và kiểm tra holdout |
| `test_evidence_submission.py` | Chấm lại evidence baseline bằng model thật, checkpoint, chọn và validate ZIP |
| `validate.py` | Kiểm tra submission theo đúng spec trước khi upload |
| `calibrate.py` | Suy `\|G\|` và `K` tối ưu cho F2 từ chỉ số leaderboard |
| `taxonomy.py` | Gán taxonomy/metadata có thể kiểm toán cho query, chunk và tài liệu |
| `probe_domains.py` | Thử khả năng crawl từng domain trước khi chạy lớn |

## Taxonomy và metadata (thử nghiệm)

Taxonomy v1 là bộ alias đa ngôn ngữ có chọn lọc, gồm bệnh, triệu chứng, giải phẫu,
xét nghiệm, thuốc, ý định truy vấn và chuyên khoa. Đây là tín hiệu từ vựng để thử
truy hồi; `confidence` chưa được hiệu chỉnh và `source_type` chỉ mô tả domain, không
đánh giá độ tin cậy của nội dung.

Chạy pilot trên 1.000 tài liệu đầu tiên để tạo hai bảng Parquet và summary:

```bash
python -u r2ai/taxonomy.py --max-docs 1000 \
    --out-dir outputs/taxonomy_pilot_v1
```

Để tạo metadata toàn bộ corpus, bỏ `--max-docs` và chọn thư mục output riêng:

```bash
python -u r2ai/taxonomy.py --out-dir outputs/taxonomy_full_v1
```

BM25 có thể mở rộng truy vấn bằng alias đã khai báo (cần index BM25 hiện có):

```bash
python -u r2ai/bm25index.py search --out bm25idx \
    --taxonomy-config configs/vibiomir/taxonomy_v1.json
```

Late fusion trong bước rank nhận metadata tài liệu/chunk. Bắt đầu thử weight nhỏ;
đo trên tập validation có nhãn hoặc chỉ số leaderboard trước khi chọn cấu hình:

```bash
python -u r2ai/rank.py --taxonomy-docs outputs/taxonomy_full_v1/docs_taxonomy_v1.parquet \
    --taxonomy-chunks outputs/taxonomy_full_v1/chunks_taxonomy_v1.parquet \
    --taxonomy-weight 0.1
```

Giữ baseline không taxonomy và so sánh cùng dữ liệu, candidates, cutoff và seed.
Pilot đầu tiên có thể lệch theo thứ tự corpus, nên không dùng nó để kết luận điểm
retrieval toàn tập.

# Pipeline và workflow dự án R2AI Medical Information Retrieval

Tài liệu này dùng làm nội dung thuyết trình và kịch bản demo cho dự án. Nội dung
được đối chiếu với code hiện tại, vì vậy các phần **đã triển khai**, **đang thử
nghiệm** và **chưa được thể lệ xác nhận** được tách riêng.

## 1. Cách giới thiệu dự án trong 30 giây

> Dự án xây dựng một hệ thống truy xuất thông tin y tế đa ngôn ngữ. Đầu vào là
> câu hỏi tiếng Việt; kho tài liệu có thể chứa tiếng Việt, tiếng Anh và tiếng
> Trung. Hệ thống không sinh câu trả lời y khoa mà xếp hạng, chọn và trả về ID
> của các tài liệu cùng các đoạn văn liên quan. Giải pháp kết hợp BM25 để bắt
> từ khóa, dense retrieval để bắt ngữ nghĩa xuyên ngôn ngữ, RRF để hợp nhất hai
> danh sách và có thể dùng cross-encoder để rerank. Kết quả được đánh giá bằng
> Precision, Recall và Macro F2 ở cả cấp document và chunk.

## 2. Bài toán và hợp đồng vào/ra

### Đầu vào

Mỗi truy vấn có dạng:

```json
{"id": "Q001", "query": "Triệu chứng thường gặp của tăng huyết áp là gì?"}
```

Kho dữ liệu được chuẩn hóa thành các chunk:

```json
{
  "chunk_id": "C00012",
  "doc_id": "DOC_005",
  "chunk_index": 2,
  "language": "vi",
  "text": "...",
  "title": "...",
  "context": "..."
}
```

### Đầu ra

Hệ thống trả về danh sách ID, không tạo câu trả lời mới:

```json
{
  "id": "Q001",
  "relevant_docs": ["DOC_005"],
  "relevant_chunks": ["C00012", "C00013"]
}
```

Ràng buộc quan trọng là mọi `chunk_id` được chọn phải có `doc_id` cha xuất hiện
trong `relevant_docs`. Pipeline tự bổ sung document cha nếu còn thiếu.

## 3. Kiến trúc tổng thể

Hệ thống có hai pha: **offline** để chuẩn bị dữ liệu/index và **online hoặc batch**
để xử lý truy vấn.

```text
PHA OFFLINE

Dữ liệu thô
    │
    ▼
Chuẩn hóa thành queries.jsonl / chunks.jsonl / ground_truth.jsonl
    │
    ├──────────────► BM25 index
    │
    └──────────────► Passage embeddings ─► Exact/FAISS dense index
                                           │
                                           ▼
                                    Manifest + corpus hash


PHA TRUY VẤN

Vietnamese query
    │
    ▼
Validate + normalize Unicode, chữ thường, khoảng trắng, dấu câu
    │
    ├────────► BM25 retrieval ────────┐
    │                                 │
    └────────► Dense retrieval ───────┤
                                      ▼
                            RRF fusion (nếu hybrid)
                                      │
                                      ▼
                          Cross-encoder reranking
                              (bật/tắt bằng config)
                                      │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
              Chọn chunk                     Gom điểm theo document
                     │                                 │
                     └────────────────┬────────────────┘
                                      ▼
                         Enforce parent consistency
                                      │
                                      ▼
                   Prediction JSONL / API response / evaluation
```

## 4. Pha offline: chuẩn bị dữ liệu và xây index

### Bước 1 — Chuẩn hóa dữ liệu

Ba tập dữ liệu lõi là:

- `queries.jsonl`: ID và câu hỏi.
- `chunks.jsonl`: nội dung, ngôn ngữ, thứ tự chunk và document cha.
- `ground_truth.jsonl`: document/chunk đúng cho từng query, dùng ở dev/test nội bộ.

Các schema Pydantic dùng chế độ strict: từ chối field lạ, ID rỗng và ID trùng.
Corpus còn được kiểm tra quan hệ document–chunk trước khi pipeline khởi động.

### Bước 2 — Xây sparse index BM25

BM25 biểu diễn văn bản bằng token và phù hợp với:

- tên thuốc, mã bệnh và thuật ngữ chuyên ngành;
- trường hợp query và tài liệu có nhiều từ trùng nhau;
- baseline nhẹ, nhanh và dễ giải thích.

Code hiện hỗ trợ tokenizer theo từ hoặc character n-gram. Index được lưu thành
JSON cùng tham số `k1`, `b`, tokenizer và SHA-256 của corpus. Khi corpus/config
không khớp index, hệ thống dừng thay vì âm thầm chạy sai.

### Bước 3 — Xây dense index

Mỗi chunk được đưa qua embedding model đa ngôn ngữ để tạo vector passage. Cấu
hình baseline dùng `BAAI/bge-m3`; smoke test nhẹ hơn có thể dùng
`intfloat/multilingual-e5-small`.

Quy trình:

```text
chunk text ─► passage prefix ─► tokenizer ─► encoder
           ─► pooling/normalization ─► vector ─► index
```

Index hiện hỗ trợ exact search; cấu hình cũng dành chỗ cho FAISS khi corpus lớn.
Manifest ghi model/tokenizer revision, dimension, dtype, danh sách chunk ID và
hash corpus để tái lập thí nghiệm.

### Vì sao phải tách index khỏi truy vấn?

Embedding toàn bộ corpus tốn thời gian nhưng corpus thay đổi ít. Làm trước một
lần giúp lúc chạy query chỉ cần encode một câu hỏi và tìm lân cận gần nhất.

## 5. Pha truy vấn: pipeline chi tiết

### Bước 1 — Load và validate cấu hình

Mọi experiment đi qua YAML strict. Cấu hình quyết định:

- backend: `sparse`, `dense` hoặc `hybrid`;
- đường dẫn corpus/index;
- model và revision;
- độ sâu candidate `top_k`;
- có fusion/reranker hay không;
- cách gom điểm document;
- threshold và giới hạn số kết quả.

Ví dụ, backend `hybrid` bắt buộc bật cả sparse, dense và RRF. Tên field cấu hình
sai sẽ bị từ chối ngay.

### Bước 2 — Validate và chuẩn hóa query

`QueryRecord` kiểm tra ID và nội dung không rỗng. Adapter sau đó:

1. chuẩn hóa Unicode NFC để giữ đúng dấu tiếng Việt;
2. chuyển về chữ thường;
3. rút gọn dấu câu lặp;
4. rút gọn khoảng trắng;
5. giữ nguyên tên thuốc, đơn vị và thực thể y khoa, không stemming phá nghĩa.

Query expansion đa ngôn ngữ đã có module nền, nhưng chưa nằm trong critical path
mặc định vì cần ablation và nguồn từ điển được kiểm duyệt.

### Bước 3 — Candidate retrieval

Pipeline hỗ trợ ba lựa chọn.

#### A. Sparse/BM25

BM25 tính độ liên quan từ tần suất từ trong chunk, độ hiếm của từ trong corpus và
độ dài chunk. Kết quả là danh sách `(chunk_id, score)`.

Ưu điểm: mạnh với exact keyword. Hạn chế: khó bắt paraphrase và khác ngôn ngữ.

#### B. Dense retrieval

Query được encode bằng cùng họ model với passage. Hệ thống tìm các vector chunk
có độ tương đồng cao nhất.

Ưu điểm: bắt ngữ nghĩa và cross-lingual. Hạn chế: nặng hơn, phụ thuộc model và có
thể bỏ qua một số exact medical term.

#### C. Hybrid retrieval

BM25 và dense chạy trên cùng query. Hai ranking được hợp nhất bằng Reciprocal
Rank Fusion:

```text
RRF_score(c) = Σ 1 / (k + rank_i(c))
```

với `rank_i(c)` là thứ hạng của chunk `c` trong retriever thứ `i`. Baseline dùng
`k = 60`. RRF dùng thứ hạng thay vì cộng trực tiếp hai score khác thang đo.

Pipeline giữ provenance cho từng candidate: đến từ BM25, dense hay cả hai, cùng
rank ở từng nguồn. Thông tin này phục vụ error analysis.

### Bước 4 — Reranking tùy chọn

Nếu bật reranker, chỉ top-N candidate sau retrieval/fusion được chấm lại bằng
cross-encoder `BAAI/bge-reranker-v2-m3`:

```text
(query, title + context + chunk text) ─► cross-encoder ─► relevance score
```

Khác embedding retrieval, cross-encoder nhìn query và passage cùng lúc nên chính
xác hơn nhưng đắt hơn. Chỉ candidate đã rerank mới đủ điều kiện đi tiếp vào bước
selection; `reranker.top_k` phải lớn hơn hoặc bằng `selection.chunk.max_k`.

### Bước 5 — Chọn relevant chunks

Không dùng một `top_k` cố định cho mọi câu hỏi. Hàm selection kết hợp hai điều
kiện:

```text
score(c) >= absolute_threshold
score(c) >= top_score - relative_delta
```

Sau đó áp dụng `min_k` và `max_k`.

- Threshold tuyệt đối loại candidate có điểm thấp.
- Relative delta loại candidate cách quá xa kết quả tốt nhất của chính query đó.
- `min_k` tránh danh sách rỗng khi cấu hình yêu cầu ít nhất một kết quả.
- `max_k` kiểm soát precision và kích thước output.

### Bước 6 — Từ chunk score sang document score

Vì retriever tìm ở cấp chunk nhưng bài toán cần cả document, pipeline gom điểm
các chunk theo document cha. Có ba chiến lược:

- `max`: điểm document bằng điểm chunk tốt nhất;
- `mean`: trung bình điểm các chunk candidate của document;
- `hybrid_mean`: `0.8 × max + 0.2 × mean(top-N)`.

Baseline mặc định dùng `max`: một đoạn rất liên quan đủ làm document đó liên
quan. Danh sách document sau đó đi qua selection riêng với threshold/min/max
riêng.

### Bước 7 — Parent consistency và ghi kết quả

Pipeline tạo `PredictionRecord`, sau đó kiểm tra từng relevant chunk. Nếu document
cha chưa có trong `relevant_docs`, document đó được thêm vào mà vẫn giữ thứ tự và
không tạo duplicate. Kết quả được ghi JSONL hoặc trả trực tiếp qua API.

## 6. Đánh giá

Với từng query và ở từng cấp document/chunk:

```text
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
F2        = 5 × Precision × Recall / (4 × Precision + Recall)
```

F2 đặt trọng số recall cao hơn precision. Điều này phù hợp với retrieval y tế:
bỏ sót tài liệu liên quan thường bị phạt mạnh hơn trả thêm một vài candidate.

Evaluator tính metric từng query rồi macro-average. Code hiện còn tính chỉ số
nội bộ tổng hợp:

```text
composite_macro_f2 = (document_macro_f2 + chunk_macro_f2) / 2
```

Lưu ý khi thuyết trình: công thức composite cuối cùng vẫn phải đối chiếu đề chính
thức của ban tổ chức; đây hiện là evaluator nội bộ.

Trước khi chấm điểm, validator kiểm tra:

- đủ và đúng tập query ID;
- không có query, document hoặc chunk ID trùng;
- không tham chiếu ID ngoài corpus;
- mọi chunk có document cha trong output;
- prediction và ground truth có cùng tập query.

## 7. Workflow phát triển và thực nghiệm

```text
Đọc/khóa data contract
        │
        ▼
Validate dữ liệu + tạo split train/dev/test
        │
        ▼
Chạy no-model/demo smoke test
        │
        ▼
BM25 baseline ─► Dense baseline ─► Hybrid RRF ─► Reranker
        │                 mỗi bước chỉ đổi một biến
        ▼
Calibrate threshold/min/max trên dev
        │
        ▼
Error analysis theo query/language/failure type
        │
        ▼
Khóa config + model revision + corpus hash
        │
        ▼
Chạy test cuối ─► validate submission ─► đóng gói
```

Nguyên tắc thực nghiệm:

1. Luôn có baseline đơn giản trước khi thêm model phức tạp.
2. Mỗi experiment chỉ đổi một biến chính để biết cải thiện đến từ đâu.
3. Chọn model, threshold và `K` trên dev, không dùng test để tuning.
4. Pin model/tokenizer revision và lưu config/hash để tái lập.
5. So sánh không chỉ Macro F2 mà cả latency, RAM/VRAM và độ ổn định.
6. Lưu error cases: keyword miss, semantic miss, cross-lingual miss, negation,
   entity mismatch và lỗi document aggregation.

## 8. Bốn baseline để trình bày

| Baseline | Thành phần | Mục đích |
|---|---|---|
| BM25 | Sparse lexical retrieval | Mốc nhẹ, giải thích được, mạnh với từ khóa |
| Dense | Multilingual embedding | Đo khả năng semantic/cross-lingual |
| Hybrid | BM25 + Dense + RRF | Kết hợp lexical và semantic recall |
| Hybrid + reranker | Hybrid + cross-encoder | Tăng precision ở nhóm candidate cuối |

Giả thuyết kỳ vọng là BM25 bắt tốt tên thuốc/thuật ngữ chính xác; dense bổ sung
paraphrase và tài liệu khác ngôn ngữ; hybrid tăng candidate recall; reranker giảm
false positive trước selection.

## 9. Giao diện sử dụng

### CLI/batch

Luồng batch đọc toàn bộ query, chạy pipeline và ghi prediction JSONL:

```bash
python -m scripts.retrieve \
  --config configs/baseline_bm25.yaml \
  --queries data/medquad/dev/queries.jsonl \
  --output outputs/presentation_demo/predictions.jsonl
```

Đánh giá:

```bash
python -m scripts.evaluate \
  --prediction outputs/presentation_demo/predictions.jsonl \
  --ground-truth data/medquad/dev/ground_truth.jsonl \
  --corpus-chunks data/medquad/dev/chunks.jsonl \
  --experiment-name presentation_demo \
  --output-json outputs/presentation_demo/metrics.json
```

Validate output:

```bash
python -m src.cli validate outputs/presentation_demo/predictions.jsonl \
  --corpus-chunks data/medquad/dev/chunks.jsonl \
  --test-queries data/medquad/dev/queries.jsonl
```

Kịch bản trên đã được chạy lại ngày 25/09/2026 với 2.333 query MedQuAD dev. Kết
quả BM25 là document F2 `0.8984`, chunk F2 `0.3693`, composite nội bộ `0.6338`;
validator xác nhận đủ 2.333 query, không có ID lạ, duplicate hoặc parent
violation. Đây là **kết quả kiểm tra kỹ thuật trên benchmark nội bộ**, không phải
điểm cuộc thi và không nên dùng để so sánh trực tiếp với leaderboard.

### HTTP API

FastAPI load pipeline một lần khi khởi động. Hai endpoint hiện có:

- `GET /health`: trạng thái service và backend;
- `POST /search`: nhận `QueryRecord` và trả `PredictionRecord`.

```bash
R2AI_CONFIG=configs/baseline_bm25.yaml \
python -m uvicorn src.api:create_app --factory --host 127.0.0.1 --port 8000
```

## 10. ViMedAQA được dùng như thế nào?

ViMedAQA đã được tải vào `data/raw/vimedaqa`. Cấu hình `all` có 44.313 mẫu với
các trường như `question`, `answer`, `context`, `title`, `topic` và URL nguồn.

Đây là dataset question-answering, chưa phải dataset retrieval đúng schema của
dự án. Để dùng hợp lệ cho test retrieval cần một bước chuyển đổi có kiểm soát:

1. coi mỗi bài/context nguồn là một document;
2. nhóm hoặc khử trùng lặp theo URL/context;
3. chia document thành chunk và tạo `doc_id`, `chunk_id`, `chunk_index`;
4. dùng `question` làm query;
5. map context chứa answer thành relevant chunk/document;
6. tránh để cùng article/context lọt sang nhiều split gây data leakage;
7. validate nguồn, license và chất lượng nhãn trước khi dùng để báo cáo kết quả.

Không nên dùng trực tiếp trường `answer` làm passage duy nhất rồi báo cáo đó là
kết quả competition; cách này làm bài toán quá dễ và không phản ánh retrieval
trên corpus thực.

## 11. Trạng thái triển khai thực tế

| Hạng mục | Trạng thái hiện tại |
|---|---|
| Schema, loader, normalization, data validation | Đã triển khai |
| Demo retriever và end-to-end JSONL | Đã triển khai |
| BM25 persistent index/retrieval | Đã triển khai và có artifact MedQuAD |
| Dense exact index/retrieval | Đã triển khai; cần đúng model cache/index cho từng config |
| Hybrid RRF và provenance | Đã triển khai |
| BGE cross-encoder reranker | Đã triển khai, phụ thuộc model/hardware |
| Chunk/document selection | Đã triển khai |
| Parent consistency, submission validator | Đã triển khai |
| Macro P/R/F2 evaluator và experiment log | Đã triển khai |
| FastAPI `/health`, `/search` | Đã triển khai |
| Threshold calibration grid search | Mới là scaffold, chưa chạy grid search thật |
| Query expansion dictionary | Có module nền, chưa bật trong pipeline mặc định |
| Tantivy backend | Chưa triển khai |
| FAISS production-scale index | Có contract/config; cần hoàn thiện và benchmark theo corpus thật |
| ViMedAQA → retrieval benchmark | Dữ liệu đã tải, chưa chuyển đổi sang schema dự án |
| External LLM/corpus/API policy | Chưa được thể lệ chính thức xác nhận |

## 12. Kịch bản thuyết trình 10–12 phút

### Slide 1 — Bài toán (1 phút)

Nói: “Đầu vào là câu hỏi y tế tiếng Việt, nhưng tài liệu có ba ngôn ngữ. Hệ thống
phải trả đúng document và chunk, không sinh lời khuyên y khoa.”

### Slide 2 — Thách thức (1 phút)

Nêu ba điểm: cross-lingual; thuật ngữ/tên thuốc cần exact match; phải cân bằng
recall và precision ở hai cấp document/chunk.

### Slide 3 — Kiến trúc hai pha (1 phút)

Trình bày sơ đồ offline/online. Nhấn mạnh index được xây trước, còn query được xử
lý nhanh ở runtime.

### Slide 4 — Retrieval ba tầng (2 phút)

Giải thích BM25 bắt từ khóa, dense bắt ngữ nghĩa, RRF hợp nhất ranking. Nêu lý do
không cộng trực tiếp score: hai retriever có thang điểm khác nhau.

### Slide 5 — Rerank và selection (1,5 phút)

Cross-encoder chấm kỹ top candidate; dual threshold thích nghi theo từng query;
min/max K kiểm soát output.

### Slide 6 — Document aggregation và consistency (1 phút)

Retriever làm việc ở cấp chunk. Điểm document được suy ra từ chunk; sau cùng hệ
thống bảo đảm mọi chunk có document cha.

### Slide 7 — Đánh giá và workflow thực nghiệm (1,5 phút)

Giải thích F2 ưu tiên recall. Nêu quy tắc chỉ tuning trên dev và chỉ thay một biến
mỗi experiment.

### Slide 8 — Demo (2 phút)

Chạy một query hoặc batch nhỏ, mở prediction JSONL, sau đó hiện bảng metric và
validation report.

### Slide 9 — Trạng thái và bước tiếp theo (1 phút)

Nói rõ phần đã triển khai. Bước tiếp theo là chuyển ViMedAQA thành benchmark
retrieval không leakage, chạy ablation bốn baseline và chờ xác nhận chính sách
external model/LLM từ ban tổ chức.

## 13. Ba câu hỏi phản biện thường gặp

### “Tại sao không chỉ dùng LLM để trả lời?”

Vì nhiệm vụ hiện được mô tả là retrieval-only và output là ID. Retrieval còn cho
phép truy vết nguồn, đo recall/precision rõ ràng và tránh sinh nội dung y khoa
không có căn cứ. Việc dùng external LLM cũng chưa được thể lệ xác nhận.

### “Tại sao cần cả BM25 và dense?”

Hai phương pháp bù trừ nhau. BM25 mạnh với exact term nhưng yếu với paraphrase;
dense mạnh với semantic/cross-lingual nhưng đôi khi bỏ sót token hiếm. Hybrid
nhắm tới candidate recall cao hơn trước khi rerank.

### “Làm sao biết điểm tăng là thật?”

Giữ split cố định, pin version, chỉ đổi một biến, log mọi config, đánh giá trên
dev bằng cùng evaluator và chỉ dùng test cho quyết định cuối. Ngoài điểm trung
bình còn phân tích lỗi theo ngôn ngữ và loại truy vấn.

## 14. Một câu kết bài

> Giá trị chính của hệ thống không chỉ là chọn một model mạnh, mà là xây dựng một
> pipeline retrieval có hợp đồng dữ liệu rõ ràng, kết quả có thể kiểm chứng, thí
> nghiệm có thể tái lập và đủ linh hoạt để thay model khi thể lệ chính thức được
> khóa.

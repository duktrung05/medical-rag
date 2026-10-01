# Đánh giá lại dự án `medical-rag`

**Ngày đánh giá:** 2026-09-28  
**Phạm vi:** code, cấu hình, dữ liệu cục bộ, artifact benchmark, test, lint, API/UI và Docker trong workspace hiện tại.  
**Nguyên tắc:** đây là đánh giá kỹ thuật; không sửa pipeline, không tune model, không chạy lại benchmark GPU và không xử lý các hạng mục khó cần nhãn, hardware hoặc quyết định từ chủ dự án.

## 1. Kết luận tổng quan

Dự án đã vượt qua giai đoạn scaffold. Đây là một hệ thống retrieval hoàn chỉnh ở mức prototype nghiên cứu:

- có BM25, dense retrieval, hybrid RRF và cross-encoder reranker;
- có pipeline chung cho batch, API và UI;
- có document/chunk hierarchy, parent consistency và submission validator;
- có manifest, corpus hash, config hash và model revision;
- có benchmark ViMed validation đủ bốn stage với 2.210 query;
- có giao diện replay benchmark và gán nhãn thủ công.

Điểm tiến bộ lớn so với bản đánh giá cũ là **candidate recall không còn là ẩn số** và **selection sau reranker không còn trộn score retrieval với score cross-encoder**. Hai vấn đề này đã có code và test bảo vệ.

Nút thắt hiện tại chuyển sang chất lượng bằng chứng:

1. Benchmark ViMed chỉ có tiếng Việt, trong khi bài toán mục tiêu là query Việt và corpus Việt/Anh/Trung.
2. Mỗi query validation chỉ có đúng một positive là context nguồn suy ra từ QA; chưa có relevance judgment đầy đủ.
3. Cấu hình được chọn luôn trả đúng 5 chunk và 5 document, nên dynamic selection có code nhưng chưa thực sự được sử dụng.
4. Chưa có hard-negative benchmark, nDCG, oracle F2 hoặc query-adaptive fusion/cardinality.
5. Luồng live đầy đủ phụ thuộc NVIDIA CUDA, dense index và model cache; máy mới clone repo chưa thể chạy ngay.
6. Test API/UI treo trên môi trường Linux Python 3.14 hiện tại, dù các nhóm test lõi chạy đạt.

Vì vậy, trạng thái hợp lý nhất của dự án là:

> **Prototype retrieval có kiến trúc tốt và benchmark nội bộ hữu ích, nhưng chưa đủ bằng chứng để kết luận năng lực cross-lingual, relevance y khoa hoặc mức sẵn sàng production/clinical.**

## 2. Bảng điểm hiện tại

Điểm dưới đây là đánh giá kỹ thuật tương đối, không phải điểm cuộc thi.

| Hạng mục | Điểm | Nhận định |
|---|---:|---|
| Kiến trúc và phân tách module | 8/10 | Rõ data, retrieval, reranking, scoring, selection, evaluation và submission |
| Sparse/dense/hybrid implementation | 8/10 | Có BM25, exact/FAISS adapter, RRF và provenance; chưa có benchmark char-BM25 đồng nhất |
| Reranker và quản lý score | 8/10 | Đã tách raw retrieval score và selection score; reranker giữ nguyên candidate set |
| Evaluation engineering | 7/10 | Có Recall@K, MRR, topic breakdown, Macro F2 và integrity gate; thiếu nDCG/oracle/uncertainty |
| Dữ liệu và ground truth | 4/10 | Corpus thực tế hữu ích nhưng qrels một-positive, chưa có hard negatives hoặc review lâm sàng |
| Cross-lingual medical retrieval | 3/10 | Model đa ngôn ngữ nhưng benchmark chính hiện 100% tiếng Việt |
| Document/chunk optimization | 6/10 | Có max/mean/hybrid mean và parent consistency; chưa tune joint objective |
| Selection và F2 optimization | 4/10 | Có threshold engine nhưng config thắng khóa top 5; calibration chưa đồng nhất runtime |
| Tái lập thí nghiệm | 7/10 | Có pin revision, hash và manifest; artifact lớn/cache chưa đóng gói, worktree chưa sạch |
| Test và chất lượng code | 5/10 | Test coverage theo chức năng tốt; test web treo trên Python 3.14, lint còn 317 findings |
| API/UI và observability | 6/10 | UI phân tích tốt cho local review; serving tuần tự, diagnostics còn mutable state |
| Docker và đa nền tảng | 3/10 | Compose hợp lệ nhưng khóa NVIDIA, CUDA 12.8 và `linux/amd64` |
| Production readiness | 3/10 | Chưa có CI, auth, rate limit, load test, CPU profile hoặc artifact bootstrap |
| Clinical readiness | 2/10 | Chưa có clinical reviewer, source review đầy đủ hoặc validation cho sử dụng lâm sàng |

## 3. Những điểm trong bản đánh giá cũ đã thay đổi

| Nhận định cũ | Trạng thái mới | Bằng chứng |
|---|---|---|
| Chưa biết candidate recall dense/hybrid/reranker trên medical | **Đã xử lý một phần** | ViMed validation có Recall@K, MRR và union candidate recall cho 2.210 query |
| Pipeline có nguy cơ trộn score RRF/dense/reranker khi selection | **Đã xử lý** | `src/pipeline.py` dùng reranker score cho selection và document aggregation khi reranker bật; raw score chỉ giữ cho diagnostics |
| MedQuAD là benchmark medical chính nhưng chỉ có tiếng Anh | **Không còn là benchmark chính** | Luồng chính đã chuyển sang ViMed tiếng Việt; MedQuAD còn dùng regression/baseline |
| Chưa có document aggregation ngoài `max` | **Đã có thêm lựa chọn** | `mean` và `hybrid_mean = 0.8 × max + 0.2 × mean(top-N)` đã được triển khai |
| Chưa có UI để phân tích retrieval | **Đã xử lý** | Inspector hỗ trợ benchmark replay, live query, source context, miss filter và human labels |
| Chưa có model/index identity | **Đã xử lý khá tốt** | Config và manifest lưu corpus/config hash, model/tokenizer revision, precision và encoder profile |
| Chưa có query-adaptive selection | **Vẫn còn** | Hàm threshold tồn tại nhưng ViMed config đặt `min_k = max_k = 5` |
| Chưa có hard-negative dataset | **Vẫn còn** | Không có hard-negative labels hoặc training corpus trong critical path |
| Chưa có oracle analysis | **Vẫn còn** | Không có metric hay script tính oracle F2/nDCG |
| Chưa đánh giá riêng theo ngôn ngữ | **Vẫn còn** | ViMed corpus hiện chỉ có `language = vi`; MedQuAD riêng chỉ có `en` |

## 4. Dữ liệu và benchmark

### 4.1 ViMed hiện tại

Corpus đã chuẩn hóa có:

- 17.955 chunks;
- 1.935 documents;
- 2.210 validation queries;
- 2.213 test queries nội bộ;
- 100% chunk được gắn ngôn ngữ `vi`;
- mỗi validation query có đúng 1 relevant chunk và 1 relevant document.

Độ dài context theo số ký tự:

- median: 406;
- p95: 1.204;
- lớn nhất: 5.712.

ViMed tốt hơn MedQuAD cho việc đo retrieval tiếng Việt và gần mục tiêu dự án hơn. Tuy nhiên, nó vẫn là QA dataset được chuyển thành closed-corpus retrieval. Ground truth là context nguồn của câu hỏi, không phải toàn bộ các context có thể trả lời đúng.

### 4.2 Ý nghĩa của qrels một-positive

Với mỗi query chỉ có một positive:

- Recall@K đo khả năng tìm lại context nguồn;
- candidate không phải context nguồn bị xem là unjudged, không chắc là irrelevant;
- Precision và F2 sẽ phạt mọi kết quả khác dù một số kết quả có thể đúng về y khoa;
- reranker có thể đưa một passage tương đương lên cao nhưng metric vẫn coi là sai;
- không thể kết luận hệ thống có precision thấp theo nghĩa lâm sàng chỉ từ qrels này.

Cấu hình reranker luôn trả 5 chunk. Khi tính trực tiếp trên source-context labels hiện tại:

| Cấp | Macro Precision | Macro Recall | Macro F2 |
|---|---:|---:|---:|
| Chunk | 17,73% | 88,64% | 49,25% |
| Document | 19,35% | 96,74% | 53,75% |
| Composite |  |  | 51,50% |

Các số này phản ánh chính sách top 5 đối với một nhãn positive duy nhất. Không nên dùng chúng để tuyên bố precision y khoa thực tế.

### 4.3 Cross-lingual gap

Project brief đặt mục tiêu query tiếng Việt trên corpus `vi/en/zh`, nhưng dữ liệu thực tế tách rời:

- ViMed: 17.955 chunks tiếng Việt;
- MedQuAD dev: 2.333 chunks tiếng Anh;
- chưa có benchmark chung chứa các positive Việt, Anh và Trung trong cùng candidate space;
- chưa có breakdown Recall/F2 theo ngôn ngữ;
- chưa có test cho entity, dosage, negation và population qua ba ngôn ngữ.

Model BGE-M3 hỗ trợ đa ngôn ngữ không tự động chứng minh pipeline giải quyết được cross-lingual medical retrieval.

### 4.4 Hard negatives

Repository chưa có tập hard negatives được gán nhãn. Random/unjudged candidates trong ranking không thể thay thế các nhóm gần đúng như:

- cùng thuốc nhưng sai chỉ định;
- cùng bệnh nhưng sai intent;
- đúng intent nhưng sai population hoặc thai kỳ;
- cùng document nhưng sai chunk;
- cùng thuật ngữ nhưng khác polarity/negation;
- khuyến cáo cũ hoặc sai bối cảnh.

Đây vẫn là một trong các hạng mục khó nhất và được giữ nguyên trong lần đánh giá này.

## 5. Retrieval và reranking

### 5.1 Kết quả validation hiện có

| Stage | Recall@1 | Recall@5 | Recall@10 | Recall@100 | MRR@100 | Miss@100 |
|---|---:|---:|---:|---:|---:|---:|
| BM25 | 49,05% | 69,19% | 75,25% | 88,37% | 0,5839 | 257 |
| Dense BGE-M3 | 55,66% | 76,61% | 81,63% | 92,22% | 0,6504 | 172 |
| Hybrid RRF | 54,80% | 77,19% | 82,26% | 92,49% | 0,6498 | 166 |
| Hybrid + reranker | 74,93% | 88,64% | 90,63% | 92,49% | 0,8091 | 166 |

Kết luận có thể rút ra:

1. Dense mạnh hơn BM25 trên ViMed ở mọi Recall@K chính.
2. Hybrid tăng candidate coverage nhưng cải thiện top 5 ít so với dense.
3. Reranker tạo cải thiện lớn ở top 1/top 5/top 10.
4. Reranker không thể sửa 166 query đã mất positive khỏi RRF top 100.
5. Trong 251 miss top 5 của reranker, 166 là retrieval/fusion miss và 85 là ranking/selection miss.

### 5.2 Oracle của candidate generation chưa được khai thác hết

Union của BM25 top 100 và dense top 100 đạt 94,52%, tương đương:

- 2.089 query có positive trong union;
- 121 query không có positive trong union;
- RRF top 100 chỉ giữ 2.044 query;
- 45 positive có trong union nhưng bị mất khi cắt RRF xuống top 100.

Đây là bằng chứng cho thấy một phần lỗi nằm ở fusion depth hoặc RRF ordering, không chỉ ở encoder. Trước khi đổi model, nên tách:

```text
miss ngoài sparse ∪ dense
miss có trong union nhưng mất sau fusion
miss có trong fused top 100 nhưng ngoài reranked top 5
hit top 5 nhưng không được final selection chọn
```

### 5.3 Score handling

Rủi ro trộn thang điểm trong bản đánh giá cũ đã được xử lý trong critical path:

- không bật reranker: selection dùng score của retriever tương ứng;
- bật reranker: chunk selection và document aggregation chỉ dùng raw logits của reranker;
- retrieval score, rerank score, selection score và eligibility được lưu riêng;
- candidate ngoài reranker depth không được selection dùng;
- test kiểm tra raw retrieval score không thay thế reranker score.

Phần còn thiếu là calibration theo từng score family. Raw BM25, cosine/RRF và cross-encoder logits vẫn cần policy riêng; không nên dùng một threshold tuyệt đối chung giữa các backend.

### 5.4 Dense index và khả năng mở rộng

Code hỗ trợ:

- exact matrix search;
- FAISS `IndexFlatIP`;
- profile E5/BGE/generic;
- CLS/mean pooling;
- float32/float16;
- CPU/CUDA/auto;
- index manifest và corpus hash.

`IndexFlatIP` vẫn là exact search trong FAISS, chưa phải ANN. Với 17.955 chunks đây là lựa chọn hợp lý. Khi corpus lớn hơn nhiều, cần benchmark ANN riêng thay vì giả định FAISS tự động giải quyết scalability.

## 6. Query understanding

Normalization hiện chỉ thực hiện:

- Unicode NFC;
- lowercase;
- rút gọn punctuation lặp;
- chuẩn hóa whitespace.

Chưa có trong critical path:

- sửa lỗi chính tả hoặc query không dấu;
- chuẩn hóa viết tắt y khoa;
- map biệt dược và hoạt chất;
- nhận diện disease/drug/symptom/procedure;
- intent classification;
- negation, temporality, dosage, population và pregnancy;
- query variants có provenance;
- terminology mapping Việt–Anh–Trung.

Đây là khoảng trống cạnh tranh lớn. Một kiến trúc phù hợp vẫn là:

```text
query gốc
  ├── normalized literal branch
  ├── abbreviation/drug/disease branch
  ├── intent + negation + population branch
  ├── English terminology branch
  └── Chinese terminology branch
          ↓
retrieval độc lập theo nhánh
          ↓
fusion có provenance
```

Không nên nối mọi expansion thành một query dài vì sẽ khó ablation và khó biết nhánh nào mang positive về.

## 7. Document–chunk scoring

Pipeline đã có ba cách aggregation:

1. `max`;
2. `mean`;
3. `hybrid_mean = 0.8 × max + 0.2 × mean(top-N)`.

Đây là nền tảng tốt hơn chỉ dùng `max`, nhưng ViMed config hiện không chứng minh `hybrid_mean` tốt hơn. Chưa có:

- số lượng supporting chunks;
- title/entity coverage;
- learned document scorer;
- loss hoặc calibration chung cho document và chunk;
- oracle phân biệt lỗi chunk ranking và document aggregation.

Document Recall@5 của reranker là 96,74%, cao hơn chunk Recall@5 88,64%. Một document có nhiều chunk khiến document hit dễ hơn, nhưng không có nghĩa chunk được chọn đã trả lời đúng intent.

Parent consistency được enforce sau selection. Điều này bảo đảm output hợp lệ nhưng có thể làm số document cuối khác số document mà selection đã chọn. Calibration hiện không gọi cùng bước này, nên metric calibration có thể khác runtime trong trường hợp document cha được bổ sung.

## 8. Selection và tối ưu F2

Hàm selection hỗ trợ threshold tuyệt đối, relative delta, `min_k` và `max_k`. Tuy nhiên, tất cả ViMed config hiện đặt:

```yaml
min_k: 5
max_k: 5
```

Threshold vì thế không thay đổi cardinality cuối: hệ thống luôn trả 5 chunk và 5 document nếu đủ candidate.

Đây là khoảng cách lớn giữa capability trong code và policy thực tế. Chưa có:

- prediction cardinality theo query;
- calibration theo score gap/entropy;
- BM25–dense agreement feature;
- entity/intent feature;
- separate policy cho query hẹp và query rộng;
- holdout đủ nhãn để chọn policy bằng Macro F2.

Hướng cạnh tranh vẫn là dự đoán cardinality từ:

```text
query features
+ top score và score gaps
+ score distribution/entropy
+ sparse–dense overlap
+ entity/intent/negation/population
→ số chunk/document nên trả
```

Không nên triển khai hoặc tune phần này trên source-context labels một-positive rồi coi là tối ưu relevance thật.

## 9. Evaluation và thiết kế thí nghiệm

### 9.1 Đã có

- Macro Precision, Recall và F2 ở chunk/document level;
- Recall@1/3/5/10/20/50/100;
- document Recall@K;
- MRR@100;
- breakdown theo topic;
- zero-hit count;
- candidate union recall cho hybrid;
- latency mean/p50/p95 trong artifact;
- corpus/config/samples/rankings checksum;
- model revision và encoder profile.

### 9.2 Còn thiếu

- nDCG với graded relevance;
- oracle Recall/F2/cardinality;
- paired confidence interval hoặc bootstrap;
- significance test giữa hai config;
- evaluation theo language;
- evaluation theo intent/entity/negation/population;
- judged/unjudged policy;
- inter-annotator agreement;
- held-out relevance set độc lập với source context;
- latency end-to-end live dưới concurrency thực.

### 9.3 Ba tầng cần đo cho mỗi experiment

1. **Candidate coverage**
   - positive có trong sparse/dense union không;
   - positive có bị fusion cắt không;
   - Recall@100/200/300 theo đúng depth.
2. **Ranking quality**
   - Recall@5/10, MRR và nDCG;
   - reranker có cải thiện trên cùng một candidate set không.
3. **Final selection**
   - Precision/Recall/F2 với qrels được review;
   - số kết quả theo query;
   - parent consistency và document/chunk joint error.

Thêm oracle:

> Nếu chọn được tập con tốt nhất từ top 100, Macro F2 tối đa là bao nhiêu?

Cách đọc:

- candidate oracle thấp: lỗi retrieval/query understanding;
- candidate oracle cao nhưng top 10 thấp: lỗi fusion/reranking;
- top 10 tốt nhưng final F2 thấp: lỗi selection/cardinality;
- chunk tốt nhưng document kém: lỗi aggregation;
- EN/ZH thấp hơn VI: lỗi cross-lingual.

## 10. Test và chất lượng code

### 10.1 Kết quả kiểm tra hiện tại

Môi trường đang dùng:

- Linux;
- Python 3.14.6;
- pytest 9.1.1;
- FastAPI 0.141.1;
- Starlette 0.52.1;
- HTTPX 0.28.1.

Kết quả:

- 39 test schema/config/metrics/scoring-selection: **pass trong 2,21 giây**;
- 3 test workflow integrity: **pass trong 8,84 giây**;
- `tests/integration/runtime/test_api.py`: treo ở test đầu, bị timeout sau 20–30 giây;
- `tests/integration/runtime/test_inspector.py`: treo ở test đầu, bị timeout sau 20–30 giây;
- full suite không hoàn tất trong lần đánh giá này;
- kết quả lịch sử “121 passed” không được dùng như kết quả hiện tại.

API và inspector thật vẫn đã chạy được bằng Uvicorn trên localhost. Vấn đề quan sát được nằm ở `TestClient`/lifespan trong tổ hợp dependency và Python 3.14 hiện tại. Cần xác nhận trên Python được project hỗ trợ chính thức, nên khóa phiên bản CI thay vì kết luận code web hỏng chỉ từ hiện tượng này.

### 10.2 Lint

Ruff báo 317 findings:

- 207 findings có thể auto-fix;
- phần lớn là import order, typing cũ và style;
- 5 lỗi F821 liên quan annotation `torch` trong `dense_encoding.py`;
- 2 import F401 không dùng trong test;
- chưa có policy Ruff được cấu hình trong `pyproject.toml`;
- `make lint` hiện không phải quality gate xanh.

Không nên auto-fix toàn bộ trong một commit lớn. Ưu tiên correctness/import, sau đó khóa rule set và xử lý dần.

### 10.3 CI và project governance

Chưa thấy:

- workflow CI;
- `.env.example`;
- `LICENSE` ở root;
- contribution guide;
- changelog/release process;
- supported Python matrix.

Đây không chặn nghiên cứu nội bộ, nhưng chặn việc gọi repository là dự án lớn sẵn sàng cộng tác hoặc phát hành.

## 11. API, UI và concurrency

### 11.1 Điểm mạnh

- API load pipeline một lần;
- input/output được validate bằng Pydantic;
- inspector đối chiếu corpus, sample, config và ranking checksums;
- UI phân biệt benchmark query nguyên bản và query đã sửa;
- human labels được lưu append-only;
- một lock bảo vệ pipeline có mutable diagnostics.

### 11.2 Giới hạn

- API serialize toàn bộ request bằng `threading.Lock`;
- `last_candidate_scores` nằm trên pipeline dùng chung;
- chưa có throughput/concurrency test ổn định;
- không có auth, user isolation hoặc rate limit;
- labels file không có transaction hoặc multi-process coordination;
- health endpoint không chứng minh dense model/index sẵn sàng cho live query;
- latency hybrid/rerank trong artifact là tổng stage cache, không phải API latency end-to-end.

UI phù hợp cho local analysis. Không nên expose trực tiếp như public service.

## 12. Docker, artifact và đa nền tảng

### 12.1 Những gì đã tốt

- base image PyTorch/CUDA được pin digest;
- Torch 2.7.1 và CUDA 12.8 được kiểm tra khi build;
- Docker dependency được pin và có constraints;
- Compose chỉ publish ra `127.0.0.1` mặc định;
- volume data/artifacts là read-only;
- có healthcheck;
- `docker compose config` hợp lệ.

### 12.2 Những gì chưa portable

- service bắt buộc reserve NVIDIA GPU;
- image khóa `linux/amd64`;
- ViMed configs dùng `device: cuda`, `precision: float16`;
- Apple Silicon/ARM và máy không NVIDIA không có supported full-UI profile;
- `HF_HUB_OFFLINE=1` mặc định yêu cầu model cache có sẵn;
- bind mounts đặt `create_host_path: false`, nên thư mục host phải tồn tại;
- fresh clone không có `data/vimed`, dense index và BGE model cache;
- chưa có `.env.example`;
- chưa có bootstrap command/container để tạo data/index;
- image khởi động inspector chứ không có demo mode tự chứa.

Trên workspace Linux hiện tại:

- BM25 index ViMed đã được tạo;
- BGE-M3 dense index không có trong `artifacts/`;
- Hugging Face cache chỉ có `intfloat/multilingual-e5-small`;
- benchmark replay chạy được;
- live dense/hybrid/reranker không sẵn sàng.

Do đó cần phân biệt ba trạng thái:

| Chế độ | Trạng thái |
|---|---|
| Demo JSON API với `configs/demo.yaml` | Chạy CPU, không cần model |
| Inspector benchmark replay | Chạy nếu data và benchmark cache hợp lệ |
| Inspector live BGE-M3 + reranker | Cần dense index, hai model revision và NVIDIA CUDA |

CPU/cross-platform path cho full UI là hạng mục chưa xử lý và được giữ nguyên.

## 13. Reproducibility và artifact integrity

Điểm mạnh:

- corpus hash;
- samples hash;
- config hash;
- ranking hash;
- model/tokenizer revision;
- encoder profile, pooling, precision, device;
- output riêng theo stage;
- validation cache bị từ chối nếu identity không khớp.

Giới hạn:

- dense/model artifacts không nằm trong repo và chưa có artifact registry;
- dependency local dùng range rộng, Docker dùng pin riêng nên có drift;
- benchmark manifest hiện có local modifications để đồng bộ checksum/config hash;
- worktree còn README edit và `docs/docker_vi.md` bị xóa;
- resume đọc toàn bộ JSONL và không phục hồi dòng cuối ghi dở;
- metrics/manifest chưa được ghi atomic;
- chưa ghi code commit hash trực tiếp trong mọi run identity.

Trước experiment mới nên đóng băng một baseline sạch thay vì dùng worktree có thay đổi không rõ ownership.

## 14. Medical, legal và safety

Project brief đã xác định đúng rằng đây là retrieval-only và không dành cho chẩn đoán hoặc điều trị. Đây là quyết định tốt.

Các rủi ro vẫn mở:

- chưa có clinical reviewer được chỉ định;
- source-context không phải clinical relevance judgment;
- source có thể cũ, thiếu version hoặc không còn đúng;
- chưa có policy xử lý passage mâu thuẫn;
- chưa có adjudication cho dose, contraindication, pregnancy và population;
- một số nguồn authoritative được ghi `REFERENCE-ONLY` hoặc `PERMISSION-REQUIRED`;
- repository chưa có license/attribution package hoàn chỉnh cho mọi artifact phân phối.

Không dùng điểm benchmark hiện tại để tuyên bố hệ thống an toàn cho sử dụng lâm sàng.

## 15. Các vấn đề khó được giữ nguyên

Theo yêu cầu, các mục sau chỉ được ghi nhận, không triển khai trong lần đánh giá này:

1. Xây relevance benchmark được human review.
2. Chỉ định clinical reviewer và annotation/adjudication policy.
3. Tạo corpus benchmark chung Việt–Anh–Trung.
4. Medical terminology normalization đa ngôn ngữ.
5. Hard-negative mining và fine-tuning.
6. Query-adaptive fusion.
7. Dynamic F2/cardinality selection.
8. Joint document–chunk learned scoring.
9. Oracle F2/nDCG với graded relevance.
10. CPU/ARM full-UI profile.
11. Concurrent model serving và load testing.
12. Legal clearance cho nguồn dữ liệu/guideline.
13. Xác nhận exact competition schema, limits và hardware rules.

## 16. Kiến trúc mục tiêu có khả năng cạnh tranh

```text
Vietnamese medical query
    │
    ├── literal normalization
    ├── abbreviation + drug/ingredient normalization
    ├── disease/symptom/procedure entities
    ├── intent + negation + population + temporality
    ├── Vietnamese query variants
    ├── English terminology variants
    └── Chinese terminology variants
            │
            ▼
Candidate generation
    ├── BM25 word
    ├── BM25 character n-gram
    ├── multilingual dense
    ├── medical dense
    └── terminology/entity retrieval
            │
            ▼
Query-adaptive fusion with branch provenance
            │
            ▼
Hard-negative-trained cross-encoder
            │
            ▼
Joint document/chunk scoring
            │
            ▼
Dynamic F2 cardinality selection
            │
            ▼
relevant_docs + relevant_chunks
```

Điểm khác biệt nên đến từ error coverage có chủ đích, không phải thêm nhiều model nhưng không biết model nào sửa loại lỗi nào.

## 17. Cách dùng AI phù hợp

AI hữu ích cho:

- sinh query paraphrases có provenance;
- gợi ý thuật ngữ, viết tắt, biệt dược và hoạt chất;
- đề xuất hard negatives từ candidate pool;
- gom nhóm lỗi retrieval;
- hỗ trợ dịch Việt–Anh–Trung;
- sinh synthetic query để training;
- viết test, chạy ablation và tổng hợp report.

AI không nên là nguồn quyết định cuối cho:

- ground truth clinical relevance;
- tương đương về liều lượng/chống chỉ định;
- passage cũ và passage hiện hành;
- correctness của bản dịch y khoa;
- threshold/cardinality khi chỉ có ít ví dụ;
- benchmark chính được tạo hoàn toàn synthetic;
- single-model LLM judge.

Quy trình phù hợp:

```text
AI đề xuất
→ retrieval pooling
→ rule validation
→ human review ca khó
→ adjudication
→ provenance
→ error analysis
→ experiment trên tune set
→ xác nhận trên holdout
```

## 18. Ưu tiên tiếp theo

### P0 — Khóa baseline đáng tin

1. Chọn và ghi rõ Python được hỗ trợ, ưu tiên 3.11/3.12 trước khi xác nhận 3.14.
2. Làm test API/UI kết thúc ổn định trong môi trường CI đã khóa.
3. Đưa regression gate về xanh với rule lint tối thiểu tập trung correctness.
4. Đóng băng code/config/corpus/model/artifact identity trong một baseline sạch.
5. Xác nhận exact submission contract và giới hạn output.
6. Phân biệt rõ readiness của replay UI và live model trong health/status.

### P1 — Làm evaluation đáng tin

1. Review toàn bộ 166 fused-pool misses.
2. Phân tích 45 positive bị mất từ union khi RRF cắt top 100.
3. Lấy mẫu 85 query có positive trong pool nhưng ngoài reranked top 5.
4. Gán nhãn 200–300 query với relevance 0/1/2 và hard negatives.
5. Có subset hai người chấm và đo agreement.
6. Thêm oracle coverage, oracle F2, nDCG và paired bootstrap.
7. Khóa tune set và holdout trước khi tuning.

### P2 — Tăng retrieval quality

1. Sweep sparse/dense depth và RRF k trên cùng protocol.
2. So sánh BM25 word và character n-gram trên ViMed.
3. Thử query normalization theo nhóm lỗi đã gán nhãn.
4. Tạo benchmark nhỏ query Việt với positive Anh/Trung.
5. Chỉ fine-tune retriever/reranker sau khi hard negatives và holdout đủ tốt.

### P3 — Tối ưu selection và serving

1. Đồng nhất calibration với parent consistency runtime.
2. So sánh fixed K và dynamic cardinality trên relevance labels đầy đủ hơn.
3. Tách diagnostics khỏi mutable pipeline state.
4. Đo cold/warm latency, p50/p95, throughput, VRAM ở concurrency 1/2/4.
5. Thêm Docker CPU profile hoặc document rõ platform được hỗ trợ.
6. Thêm artifact bootstrap và `.env.example`.

## 19. Phân bổ công sức đề xuất

| Nhóm | Tỷ lệ |
|---|---:|
| Ground truth, relevance review và error analysis | 35% |
| Cross-lingual query understanding và terminology | 25% |
| Hard negatives, reranker và retrieval experiments | 20% |
| F2 calibration và document/chunk selection | 15% |
| UI, demo và pitching | 5% |

Nếu chỉ chọn một hướng tạo lợi thế:

> **Medical-aware query decomposition + hard-negative reranker + dynamic F2 selection trên một relevance benchmark được review.**

BM25 + BGE-M3 + RRF + reranker hiện đã tạo baseline tốt. Lợi thế tiếp theo không đến từ thay model ngẫu nhiên mà từ dữ liệu đánh giá, phân loại lỗi và selection policy tốt hơn.

## 20. Quyết định cuối

### Có thể dùng ngay cho

- nghiên cứu retrieval tiếng Việt;
- benchmark replay và error analysis;
- so sánh sparse/dense/hybrid/reranker trên source-context protocol;
- batch prediction, validation và demo local;
- làm nền để xây relevance dataset tốt hơn.

### Chưa nên tuyên bố

- đã giải quyết cross-lingual Việt–Anh–Trung;
- đã tối ưu Macro F2 theo relevance thật;
- có precision lâm sàng theo con số hiện tại;
- chạy được trên mọi máy bằng Docker;
- production ready;
- medically validated hoặc safe for clinical use.

Kết luận ngắn:

> Dự án có nền tảng kỹ thuật tốt và đã chứng minh reranker mang lại giá trị rõ ràng. Rủi ro lớn nhất không còn là thiếu module retrieval, mà là thiếu relevance data đủ sâu để biết nên tối ưu module nào và để chứng minh cải thiện có ý nghĩa.

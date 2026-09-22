# Playbook vận hành dự án Multilingual Medical Retrieval từ trang trắng

## 1. Mục đích của playbook

Tài liệu này mô tả cách vận hành một dự án từ lúc repository gần như trống, chỉ có
một chủ đề tổng quát:

> Nhận query tiếng Việt và truy xuất đúng document/chunk y khoa trong corpus tiếng
> Việt, tiếng Anh và tiếng Trung.

Mục tiêu không phải là viết thật nhiều code càng sớm càng tốt. Mục tiêu là tạo ra
một chuỗi bằng chứng đủ chắc để trả lời lần lượt:

1. Ta đang giải bài toán nào?
2. Dữ liệu và nhãn có đáng tin không?
3. Metric có đo đúng thứ ta quan tâm không?
4. Baseline đơn giản nhất có chạy xuyên suốt không?
5. Lỗi hiện tại nằm ở dữ liệu, chunking, retrieval, ranking hay selection?
6. Thay đổi tiếp theo có thực sự cải thiện kết quả không?
7. Kết quả cuối có tái lập, kiểm tra và bàn giao được không?

Chuỗi vận hành tổng quát:

```text
Problem contract
  -> research và source policy
  -> benchmark contract
  -> schema và validator
  -> smoke data
  -> evaluator known-answer tests
  -> end-to-end smoke test
  -> chọn backbone tối thiểu
  -> khóa dev protocol
  -> cải tiến từng giả thuyết
  -> chạy held-out test một lần
  -> reproducibility và bàn giao
```

## 2. Bốn mục tiêu phải được tách riêng

Ngay từ đầu phải ghi rõ bốn loại mục tiêu. Không dùng một con số metric để thay thế
cho cả bốn.

### 2.1 Product aim

Với một query tiếng Việt, hệ thống trả về danh sách document và chunk liên quan,
có thể nằm trong corpus `vi`, `en` hoặc `zh`.

### 2.2 Scientific aim

Mọi quyết định về chunking, retriever, K, reranker hoặc threshold phải xuất phát từ
một giả thuyết đã ghi trước và được kiểm tra trên dev, không giải thích hồi tố sau
khi đã nhìn test.

### 2.3 Engineering aim

Cùng input, config và code revision phải tạo lại được artifact, ranking và metric
tương đương; lỗi dữ liệu phải được phát hiện trước khi chạy model.

### 2.4 Safety aim

Nội dung y khoa phải có provenance, quyền sử dụng và trạng thái review. Hệ thống
retrieval không được mô tả như công cụ chẩn đoán hay tư vấn điều trị.

Nếu chưa có người có chuyên môn y khoa review, artifact chỉ được gọi là
`software/synthetic smoke benchmark`, không gọi là medically validated benchmark.

## 3. Cách tôi làm việc ở mỗi bước

Mỗi bước luôn dùng cùng một vòng lặp:

```text
Inspect -> Question -> Hypothesis -> Small artifact -> Test -> Evidence -> Decision
```

Chi tiết:

1. **Inspect:** đọc trạng thái repository, dữ liệu, config, test và output hiện có.
2. **Question:** ghi câu hỏi cụ thể đang cần trả lời.
3. **Hypothesis:** ghi kết quả dự kiến và dấu hiệu có thể bác bỏ nó.
4. **Small artifact:** tạo artifact nhỏ nhất có thể kiểm tra được.
5. **Test:** chạy kiểm tra tự động hoặc review thủ công đã định trước.
6. **Evidence:** lưu output, hash, metric, lỗi và quan sát.
7. **Decision:** chốt, giữ mở hoặc loại bỏ option; ghi lý do và điều kiện xem lại.

Một task chỉ được đánh dấu `DONE` khi có đủ:

- Artifact cụ thể.
- Validator hoặc test tương ứng.
- Kết quả test đã lưu.
- Quyết định hoặc kết luận được ghi lại.
- Không còn câu hỏi mở có thể làm thay đổi artifact đó.

`Code đã viết xong` không đồng nghĩa với `DONE`.

## 4. File đầu tiên và thứ tự tạo file

Không bắt đầu bằng `app.py`, vector database hoặc model. Bắt đầu bằng hợp đồng bài
toán. Thứ tự dưới đây là thứ tự mặc định; chỉ chuyển bước khi điều kiện `DONE` của
bước trước đã đạt.

## 5. Bước 1 — Tạo `docs/00_project_brief.md`

Đây là file đầu tiên.

Nội dung bắt buộc:

```markdown
# Project brief

## Problem
## Intended user
## Input
## Output
## Languages
## Unit of retrieval
## Success criteria
## Non-goals
## Medical and legal risks
## Known facts
## Assumptions
## Open questions
```

### Phân tích cần làm

- Query là câu hỏi, keyword hay clinical intent?
- Cần retrieve document, chunk hay cả hai?
- Corpus cố định hay cập nhật liên tục?
- Ba ngôn ngữ là corpus song song hay độc lập?
- Recall quan trọng hơn precision đến mức nào?
- Output phục vụ competition, nghiên cứu hay người dùng thật?
- Có latency, memory hoặc hardware constraint không?
- Ai có quyền quyết định nội dung y khoa là đúng?

### Khi nào bước 1 DONE

- Input/output được mô tả bằng ví dụ JSON cụ thể.
- Unit đánh giá ở chunk và document được xác định.
- Non-goals được ghi rõ, ví dụ chưa làm generation hoặc medical advice.
- Open question nào có thể đổi kiến trúc phải có owner và deadline.
- Người phụ trách dự án đồng ý với project brief.

Không viết pipeline trước khi bước này đạt.

## 6. Bước 2 — Tạo `docs/01_research_log.md`

File này ghi lại ta đã tìm gì, ở đâu, vào ngày nào và kết luận gì. Không để research
chỉ tồn tại trong lịch sử trình duyệt hoặc trí nhớ.

Mẫu entry:

```markdown
## R-001: Nguồn cho xử trí sốc phản vệ

- Question:
- Search terms:
- Primary sources checked:
- Version/date accessed:
- License/reuse status:
- Findings:
- Remaining uncertainty:
- Decision affected:
```

### Thứ tự tìm kiếm

1. Đề bài chính thức hoặc specification của hệ thống.
2. Tài liệu từ cơ quan/tổ chức phát hành nguồn y khoa.
3. Model card và documentation chính thức của model/tokenizer.
4. Paper gốc nếu một quyết định phụ thuộc vào kết quả nghiên cứu.
5. Issue tracker chỉ dùng để hiểu lỗi triển khai, không dùng thay nguồn chính.

### Những thứ cần tìm

- Schema và metric chính thức của bài toán.
- Quyền tái sử dụng dữ liệu và guideline.
- Phiên bản và ngày cập nhật nguồn y khoa.
- Model hỗ trợ ngôn ngữ nào và yêu cầu prefix/pooling ra sao.
- Giới hạn token, behavior của tokenizer với Việt/Anh/Trung.
- Dependency, hardware và license của model.

### Chú ý

- Không sao chép guideline dài vào repository nếu quyền sử dụng không rõ.
- URL không đủ làm provenance; cần phiên bản, ngày truy cập và checksum nếu có.
- Tách `nội dung đúng` khỏi `được phép tái sử dụng`.
- Không coi blog tổng hợp là nguồn y khoa chính.

### Khi nào bước 2 DONE

- Mỗi quyết định quan trọng có ít nhất một nguồn chính hoặc được đánh dấu là giả
  định cần kiểm chứng.
- Source/license risk có phương án xử lý.
- Không còn unknown có thể làm đổi hoàn toàn input/output contract.

## 7. Bước 3 — Tạo `docs/02_decision_log.md`

Mọi lựa chọn chưa chốt phải đi vào decision log, không chọn ngầm trong code.

Mẫu quyết định:

```markdown
## D-001: Retriever đầu tiên

- Status: proposed | experimenting | accepted | rejected | superseded
- Context:
- Options:
- Fixed constraints:
- Evidence needed:
- Experiment:
- Decision rule:
- Result:
- Decision:
- Consequences:
- Revisit trigger:
```

### Cách xử lý option chưa rõ

Phân loại quyết định:

1. **Dễ đảo ngược, chi phí thấp:** chọn default đơn giản nhất, ghi lại và tiếp tục.
2. **Có thể đo được:** tạo thí nghiệm nhỏ trên dev với decision rule định trước.
3. **Khó đảo ngược:** dừng triển khai, research thêm hoặc xin quyết định từ owner.
4. **Yêu cầu chuyên môn y khoa/pháp lý:** không tự suy đoán; chuyển cho reviewer phù
   hợp.

Không chạy năm option chỉ vì chúng tồn tại. Mỗi thí nghiệm phải trả lời một câu hỏi.

Ví dụ:

```text
Question: exact dense hay FAISS cho smoke corpus?
Constraint: corpus chỉ khoảng vài chục đến vài trăm chunk.
Rule: chọn phương án ít dependency hơn nếu ranking giống nhau.
Expected decision: exact dense.
Revisit trigger: corpus lớn đến mức latency/memory vượt budget.
```

### Khi nào bước 3 DONE

- Các quyết định kiến trúc đang mở đều có ID.
- Mỗi quyết định có evidence cần thu thập và decision rule.
- Không có option quan trọng chỉ tồn tại dưới dạng TODO trong code.

## 8. Bước 4 — Tạo `docs/03_benchmark_contract.md`

Benchmark phải được thiết kế trước retriever, vì model chỉ có ý nghĩa khi ground
truth và metric có ý nghĩa.

File này phải chốt:

- Chủ đề y khoa.
- Source group và document group.
- Ngôn ngữ query/corpus.
- Clinical intent và query group.
- Định nghĩa relevance `2`, `1`, `0`, `-1`.
- Quy tắc positive, hard negative và unjudged candidate.
- Quy tắc chia dev/test chống leakage.
- Metric chunk-level và document-level.
- Quy tắc dùng dev và thời điểm được mở test.
- Vai trò của reviewer và adjudicator.

### Quy mô smoke benchmark ban đầu

Một cấu hình hợp lý cho dự án này:

- 5 topic.
- Mỗi topic có document `vi`, `en`, `zh`.
- Mỗi document có 3–5 section theo intent.
- Khoảng 20 intent group.
- Hai paraphrase tiếng Việt cho mỗi intent, khoảng 40 query.
- Split theo intent group, không split từng paraphrase.

### Khi nào bước 4 DONE

- Một reviewer khác có thể đọc contract và tự gán cùng một loại nhãn.
- Không dùng quy tắc `khác group = negative`.
- Test protocol được khóa trước khi chạy model.
- Có rule rõ cho candidate chưa được đánh giá.

## 9. Bước 5 — Tạo config trước code model

File tiếp theo:

```text
configs/smoke_medical_dense_e5.yaml
```

Config chứa mọi tham số có thể làm thay đổi artifact hoặc metric:

- Dataset và input paths.
- Seed.
- Chunk strategy, target/max/overlap token.
- Tokenizer revision.
- Model revision.
- Prefix và pooling.
- Normalize embeddings.
- Similarity.
- Ranking depth.
- K candidates.
- Document aggregation.

Sau đó mới tạo schema config và test:

```text
src/config.py
tests/test_medical_config.py
```

### Test bắt buộc

- Config hợp lệ load được.
- Field lạ bị từ chối.
- Thiếu model/tokenizer revision bị từ chối.
- Giá trị K âm, max token không hợp lệ hoặc language không hỗ trợ bị từ chối.
- Cùng config tạo cùng canonical hash.

### Khi nào bước 5 DONE

- Không còn tham số ảnh hưởng kết quả bị hard-code trong runner.
- Config có hash và có thể snapshot cạnh output.
- Unit test config không cần cài Torch/model.

Đây là Gate 0.

## 10. Bước 6 — Tạo schema dữ liệu và fixture cực nhỏ

File theo thứ tự:

```text
src/data/schema.py
tests/fixtures/medical_minimal/
tests/test_medical_schema.py
```

Schema tối thiểu:

- `SourceRecord`.
- `DocumentRecord`.
- `QueryRecord` có intent group.
- `ChunkRecord`.
- `QrelRecord` có relevance grade.
- `PredictionRecord`.

Fixture ban đầu chỉ cần đủ để phá vỡ các giả định sai:

- 1 topic.
- 3 document `vi/en/zh`.
- Mỗi document ít nhất 2 section.
- 2 query.
- Một positive cùng ngôn ngữ.
- Một positive chéo ngôn ngữ.
- Một hard negative cùng bệnh nhưng sai intent.
- Một candidate chưa được đánh giá.

### Test bắt buộc

- ID rỗng/trùng bị từ chối.
- Language ngoài contract bị từ chối.
- Grade ngoài `-1, 0, 1, 2` bị từ chối.
- Query group và split không hợp lệ bị từ chối.
- JSONL đọc và ghi lại không làm mất field.

### Khi nào bước 6 DONE

- Mọi artifact sau này đều có schema máy kiểm tra được.
- Test fixture đủ chứa positive, hard negative, cross-lingual và unjudged.

## 11. Bước 7 — Tạo source registry, raw documents và query plan

Thứ tự file:

```text
data/medical_smoke/sources.jsonl
data/medical_smoke/raw/documents.jsonl
data/medical_smoke/intent_groups.jsonl
data/medical_smoke/queries_dev.jsonl
data/medical_smoke/queries_test.jsonl
data/medical_smoke/annotation_guidelines.md
data/medical_smoke/split_manifest.json
data/medical_smoke/README.md
```

Không viết qrels chunk-level cuối cùng ở đây vì chunk ID chưa được khóa. Thay vào đó,
mỗi query ghi `positive_span` hoặc `positive_section_id` dự kiến và hard-negative
intent. Đây là Gate 1A.

### Review bắt buộc

- 100% document có nguồn và license/reuse status.
- Nội dung y khoa được reviewer có thẩm quyền duyệt.
- Query không sao chép nguyên positive.
- Intent group không xuất hiện ở cả dev và test.
- Mỗi query có positive section và hard-negative section dự kiến.

### Khi nào bước 7 DONE

- Validator nguồn/document/query đều pass.
- Reviewer sign-off hoặc benchmark được ghi rõ là synthetic-only.
- Split manifest và input hashes đã được tạo.

## 12. Bước 8 — Viết chunker trước retriever

File theo thứ tự:

```text
src/data/chunking.py
scripts/build_medical_corpus.py
tests/test_medical_chunking.py
```

Chiến lược đầu tiên nên là sentence/paragraph-aware token window, không dùng semantic
chunking ở vòng đầu. Lý do: deterministic, dễ giải thích và dễ tìm lỗi.

Default khởi đầu:

```text
target: 180–256 token
default target: 220
hard max: 320
overlap: 32
```

### Những boundary phải kiểm tra

- Không tách tên thuốc khỏi liều và đơn vị.
- Không tách câu phủ định khỏi chỉ định/chống chỉ định.
- Giữ heading cùng nội dung liên quan khi có thể.
- Sentence splitting hoạt động với tiếng Việt, Anh và Trung.
- Chunk ID ổn định khi input/config không đổi.

### Test bắt buộc

- Document một section ngắn.
- Document nhiều section.
- Một câu dài hơn hard max.
- Câu chứa số, liều và đơn vị.
- Text tiếng Trung không có khoảng trắng kiểu tiếng Anh.
- Chạy hai lần tạo cùng chunks và IDs.

### Khi nào bước 8 DONE

- Không chunk nào vượt hard max mà không có error rõ ràng.
- Parent mapping và chunk index liên tục.
- Golden fixture tạo cùng checksum ở hai lần chạy.

## 13. Bước 9 — Tạo data auditor và khóa chunk IDs

File theo thứ tự:

```text
src/data/audit.py
scripts/audit_medical_data.py
tests/test_medical_audit.py
data/medical_smoke/processed/chunks.jsonl
data/medical_smoke/processed/doc_map.json
data/medical_smoke/processed/build_manifest.json
outputs/medical_smoke/data_audit.json
```

Audit phải fail closed khi có:

- Duplicate ID.
- Text rỗng.
- Chunk index trùng hoặc không liên tục.
- Parent không tồn tại.
- Language không khớp.
- Qrel/positive span trỏ tới ID không tồn tại.
- Exact hoặc normalized duplicate ngoài dự kiến.
- Chunk vượt giới hạn token.
- Dev/test leakage.

Audit phải báo thống kê count và token length theo topic/language, cùng input/config/
output hashes.

### Khi nào bước 9 DONE

- `data_audit.json` có `status: pass`.
- Build hai lần cho cùng hashes.
- Chunk IDs được khóa; thay chunking từ đây là một benchmark version mới.

Đây là Gate 2.

## 14. Bước 10 — Hoàn thành annotation ở cấp chunk

Sau khi chunk ID ổn định mới tạo:

```text
data/medical_smoke/qrels_dev.jsonl
data/medical_smoke/private/qrels_test.jsonl
outputs/medical_smoke/annotation_report.json
```

Quy trình:

1. Map positive section/span sang chunk IDs.
2. Lấy candidates cùng topic để tìm hard negatives.
3. Có thể dùng BM25 và dense đã pin để hỗ trợ pooling, nhưng chưa xem metric và chưa
   tune model.
4. Gán grade `2/1/0`; phần chưa review giữ `-1`.
5. Double-annotate một phần pool.
6. Adjudicate mọi bất đồng.
7. Khóa qrels và hashes.

### Khi nào bước 10 DONE

- 100% query có ít nhất một positive được review.
- Mỗi intent chính có hard negative.
- Có positive cross-lingual.
- Unjudged không bị tự động tính là negative.
- Test qrels chưa được dùng để ra quyết định.

Đây là Gate 1B và là lúc Gate 0–2 được khóa chung.

## 15. Bước 11 — Chứng minh evaluator đúng trước model

File theo thứ tự:

```text
src/evaluation/evaluator.py
tests/fixtures/evaluator_known_answers/
tests/test_medical_evaluator.py
```

Known-answer tests:

1. Perfect prediction: `P=R=F2=1`.
2. Empty prediction: `P=R=F2=0`.
3. Một đúng, một sai: kết quả khớp tính tay.
4. Đúng document nhưng sai chunk.
5. Candidate `-1` không tự động trở thành false positive trong pooled evaluation.
6. Thiếu/thừa query phải báo lỗi.
7. Unknown hoặc duplicate ID phải báo lỗi.
8. Parent consistency violation phải báo lỗi.

### Khi nào bước 11 DONE

- Mọi expected metric được tính tay và test pass.
- Evaluator strict với query universe và ID universe.
- Không thay evaluator sau khi nhìn test; nếu bắt buộc thay thì phải version benchmark.

Đây là Gate 3.

## 16. Bước 12 — Chạy smoke test không model

Trước dense model, chạy end-to-end bằng retriever deterministic rất đơn giản, ví dụ
keyword hoặc score fixture.

Smoke test cần chứng minh:

```text
documents -> chunks -> validation -> retrieval -> predictions
          -> parent aggregation -> submission validation -> evaluator
```

Mục tiêu không phải điểm cao. Mục tiêu là phát hiện lỗi wiring, schema, ID, path,
parent mapping và output format.

### Khi nào DONE

- Một command chạy xuyên suốt trên fixture.
- Output hợp lệ và metric đúng dự kiến.
- Failure injection tạo đúng lỗi ở đúng bước.

## 17. Bước 13 — Chọn backbone bằng smoke experiments

Không chọn backbone vì model phổ biến. Tạo ba baseline có vai trò khác nhau:

### Option A — BM25

Dùng để đo lexical matching, debug query/corpus và tạo hard-negative candidates.
BM25 là comparator bắt buộc nhưng khó giải quyết truy xuất chéo ngôn ngữ nếu query
và passage không dùng chung từ vựng.

### Option B — Multilingual dense E5

Dùng để kiểm tra semantic và cross-lingual retrieval. Với smoke corpus nhỏ, dùng
exact cosine search để tránh đưa FAISS/vector database vào quá sớm.

### Option C — Hybrid BM25 + dense

Chỉ thử sau khi A và B đều chạy đúng. Hybrid có ý nghĩa khi error analysis cho thấy
hai retriever sửa được các loại lỗi khác nhau.

### Smoke experiments cần chạy

1. **S1 — Overfit fixture:** positive hiển nhiên phải xuất hiện ở top đầu.
2. **S2 — Cross-lingual:** query Việt tìm được positive Anh/Trung.
3. **S3 — Hard negative:** đúng bệnh nhưng sai intent không được đứng trên positive.
4. **S4 — Full ranking:** lưu toàn bộ corpus để biết exact rank của positive.
5. **S5 — Determinism:** chạy lại cùng config cho cùng ranking/metric trong tolerance.
6. **S6 — Parent aggregation:** nhiều chunk cùng document không tạo document trùng.
7. **S7 — Failure injection:** model revision, field, ID hoặc input hash sai phải dừng.

### Bằng chứng dùng để chọn

- Positive coverage ở ranking depth lớn.
- Cross-lingual coverage.
- Khả năng phân biệt hard negative.
- Mức đơn giản của dependency và vận hành.
- Tính tái lập.
- Latency/memory trong constraint.
- Lỗi có giải thích được hay không.

### Hướng backbone ban đầu cho dự án này

Nếu smoke test xác nhận khả năng cross-lingual, chọn:

```text
multilingual E5
  + exact cosine ranking
  + full ranking trên smoke corpus
  + max chunk score cho document aggregation
  + fixed K được chọn trên dev
```

Giữ BM25 làm comparator, diagnostic tool và candidate-pooling tool. Chưa thêm hybrid,
reranker, query expansion hoặc vector database.

Lý do chọn hướng này:

- Dense multilingual phù hợp mục tiêu query Việt/corpus đa ngôn ngữ.
- Exact search đủ cho corpus nhỏ và dễ tái lập.
- Ít thành phần giúp xác định nguyên nhân lỗi.
- BM25 vẫn cung cấp tín hiệu lexical độc lập để so sánh.

Quyết định chỉ được chuyển từ `proposed` sang `accepted` sau khi smoke evidence được
lưu trong decision log.

## 18. Bước 14 — Sau khi chốt backbone

Thứ tự tiếp theo:

1. Tách build index khỏi query retrieval.
2. Lưu embeddings, chunk order, manifest và checksums.
3. Load lại index và chứng minh ranking không đổi.
4. Sinh full ranking trên dev.
5. Tạo chunk predictions tại các K đã định trước.
6. Aggregate document bằng rule đã khóa.
7. Đánh giá dev cho từng K.
8. Chọn K theo decision rule đã đăng ký trước.
9. Khóa config thắng cuộc.
10. Chưa mở test.

File/artifact:

```text
artifacts/medical_smoke_dense/
  embeddings.npy
  chunk_ids.json
  index_manifest.json

outputs/medical_smoke/dev/
  rankings.jsonl
  predictions_k1.jsonl
  predictions_k3.jsonl
  predictions_k5.jsonl
  predictions_k10.jsonl
  metrics.json
  manifest.json
```

Rule chọn K ví dụ:

1. Macro F2 dev cao nhất.
2. Chênh lệch dưới `0.001` coi là hòa.
3. Khi hòa chọn K nhỏ hơn.
4. Không thay rule sau khi xem kết quả test.

## 19. Bước 15 — Error analysis trước mọi cải tiến

Không cải tiến bằng cách ngẫu nhiên thay model. Với mỗi query dev sai, lưu:

- Exact rank của positive đầu tiên.
- Positive có nằm trong candidate set không.
- Top false positive và error tag.
- Ngôn ngữ của query/positive.
- Lỗi ở chunk-level hay document-level.
- Query intent, topic và độ khó.

Decision tree:

```text
Positive không có trong top sâu
  -> kiểm tra corpus, source, chunking, query representation hoặc retriever

Positive có trong top sâu nhưng không vào top K
  -> ranking/reranking problem

Đúng document nhưng sai chunk
  -> chunking hoặc chunk scoring/aggregation problem

Nhiều candidate liên quan một phần
  -> qrels/relevance definition hoặc selection problem

Chỉ lỗi cross-lingual
  -> multilingual representation/alignment problem

Sai do thông tin nguồn
  -> data problem; không sửa bằng model
```

## 20. Bước 16 — Thứ tự cải tiến

Mỗi vòng chỉ thay một nhóm biến và tạo output directory mới.

### Ưu tiên 1 — Dữ liệu và annotation

- Sửa source/document lỗi.
- Bổ sung hard negatives thiếu.
- Giải quyết disagreement.
- Kiểm tra leakage và coverage.

### Ưu tiên 2 — Chunking

- Điều chỉnh boundary, size hoặc overlap nếu lỗi đúng document/sai chunk phổ biến.
- Version lại processed corpus và qrels nếu chunk IDs thay đổi.

### Ưu tiên 3 — Retriever

- So BM25 với dense.
- Chỉ thử hybrid khi hai hệ thống có lỗi bổ sung cho nhau.
- Chỉ đổi embedding model khi có giả thuyết cụ thể.

### Ưu tiên 4 — Reranker

Chỉ thêm reranker nếu positive đã xuất hiện trong candidate depth nhưng thứ hạng thấp.
Nếu positive hoàn toàn không được retrieve, reranker không giải quyết được vấn đề.

### Ưu tiên 5 — Query expansion

Chỉ thử khi error analysis cho thấy mismatch thuật ngữ, viết tắt hoặc tên thuốc là
nguyên nhân chính. Expansion phải có ablation vì nó cũng có thể làm giảm precision.

### Ưu tiên 6 — Selection/threshold

Chỉ calibrate sau khi ranking đã ổn định. Không dùng threshold để che lỗi retriever.

Mỗi experiment phải có:

```text
hypothesis
changed variables
fixed variables
expected effect
decision rule
dev result
error analysis
accept/reject decision
```

## 21. Bước 17 — Khi nào được chạy test

Chỉ mở held-out test khi:

- Data, qrels và evaluator đã khóa.
- Backbone và config đã khóa.
- K/threshold được chọn hoàn toàn trên dev.
- Không còn blocker trong gate report.
- Command tái tạo dev đã chạy lại thành công.
- Test protocol và reporting template đã viết sẵn.

Test chỉ chạy một lần cho quyết định cuối. Nếu sau đó thay model/config dựa trên test,
test đó đã trở thành dev và phải có một held-out set mới.

## 22. Bước 18 — Reproducibility và bàn giao

Artifact cuối:

```text
README.md
docs/00_project_brief.md
docs/01_research_log.md
docs/02_decision_log.md
docs/03_benchmark_contract.md
docs/experiment_log.md
configs/locked_medical_backbone.yaml
outputs/final/gate_report.json
outputs/final/manifest.json
outputs/final/metrics.json
outputs/final/error_analysis.jsonl
```

Một command phải có khả năng:

```text
validate inputs
  -> build/load corpus
  -> build/load index
  -> retrieve
  -> aggregate/select
  -> validate predictions
  -> evaluate
  -> write manifest và report
```

Manifest cuối phải chứa code commit, config hash, dependency versions, model/tokenizer
revision, input hashes, artifact hashes, seed, hardware/device và elapsed time.

## 23. Definition of Done cho mục tiêu cuối

Dự án chỉ đạt aim khi đồng thời thỏa mãn:

### Data

- Nguồn, version, license và review status đầy đủ.
- Không có integrity error hoặc dev/test leakage.
- Query có positive, hard negative và coverage cross-lingual.

### Evaluation

- Evaluator vượt known-answer tests.
- Chunk/document metric và parent consistency đúng.
- Test không được dùng để tune.

### Model

- Backbone được chọn bằng evidence trên dev.
- Mọi cải tiến có ablation hoặc comparator.
- Error analysis giải thích được các failure mode chính.

### Engineering

- Một command tái tạo pipeline.
- Config và artifact có hash/manifest.
- Hai lần chạy độc lập cho kết quả nằm trong tolerance đã định.
- Test suite và gate report đều pass.

### Safety và bàn giao

- Có disclaimer đúng phạm vi.
- Nội dung y khoa có sign-off phù hợp, hoặc được ghi rõ là synthetic-only.
- README cho biết cách chạy, giới hạn và cách tái tạo kết quả.
- Người khác có thể clone repository và làm lại kết quả mà không cần thông tin ngầm.

## 24. Nguyên tắc dừng

Phải dừng và sửa lớp trước đó nếu:

- Source hoặc license không rõ.
- Ground truth mâu thuẫn.
- Evaluator chưa được chứng minh đúng.
- Config còn tham số ẩn.
- Dev/test leakage chưa được giải quyết.
- Experiment thay nhiều biến nhưng không có ablation.
- Kết quả không tái lập.

Không thêm model mạnh hơn để vượt qua một gate dữ liệu hoặc đánh giá đang hỏng.

## 25. Checklist vận hành ngắn gọn

```text
[ ] Project brief được duyệt
[ ] Research/source/license log hoàn chỉnh
[ ] Decision log có mọi option quan trọng
[ ] Benchmark contract được khóa
[ ] Config schema strict và có hash
[ ] Data schemas + minimal fixture pass
[ ] Source/doc/query Gate 1A pass
[ ] Chunker + deterministic audit Gate 2 pass
[ ] Chunk-level qrels Gate 1B pass
[ ] Evaluator known-answer Gate 3 pass
[ ] End-to-end no-model smoke pass
[ ] BM25 comparator pass
[ ] Dense cross-lingual smoke pass
[ ] Backbone decision được ghi bằng evidence
[ ] Full dev ranking và error analysis hoàn chỉnh
[ ] Mỗi cải tiến chỉ kiểm tra một giả thuyết
[ ] Final config được khóa trước test
[ ] Held-out test chạy đúng protocol
[ ] Reproducibility + handoff pass
```

Playbook này biến việc phát triển từ chuỗi hành động “thử model rồi xem điểm” thành
chuỗi quyết định có hợp đồng, evidence, gate và điều kiện dừng rõ ràng. Đích cuối
không chỉ là metric cao, mà là một hệ thống retrieval y khoa đa ngôn ngữ đúng dữ
liệu, đúng phép đo, giải thích được, tái lập được và có thể bàn giao.

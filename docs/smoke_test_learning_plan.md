# Kế hoạch học workflow Retrieval qua một smoke test

## 1. Mục tiêu

Smoke test này không nhằm tạo hệ thống có điểm thi cao. Mục tiêu là đi hết một
vòng đời retrieval có thể kiểm tra được:

```text
tài liệu -> chunk -> embedding/index -> truy hồi chunk -> suy ra document
         -> chọn kết quả -> JSONL -> Precision/Recall/F2 -> phân tích lỗi
```

Sau khi hoàn thành, người thực hiện phải giải thích được:

1. Dữ liệu nào được xử lý offline và dữ liệu nào được xử lý khi có query.
2. Vì sao một query tiếng Việt có thể tìm chunk tiếng Anh hoặc tiếng Trung.
3. `top_k`, threshold và score ảnh hưởng Precision/Recall/F2 như thế nào.
4. Từ kết quả chunk suy ra document và bảo toàn quan hệ cha-con ra sao.
5. Một lỗi retrieval thuộc data, candidate generation, ranking hay selection.
6. Khi BTC phát hành dữ liệu, phần nào được giữ lại và phần nào phải thay.

## 2. Phạm vi và nguyên tắc

### Trong smoke test

- Dữ liệu nhỏ, tự kiểm soát và được gán nhãn thủ công.
- Query luôn bằng tiếng Việt; corpus có `vi`, `en`, `zh`.
- Retrieval-only, không sinh câu trả lời bằng LLM.
- Không training hoặc fine-tuning.
- Dense retrieval là đường chạy đầu tiên.
- Exact vector search được ưu tiên để chưa đưa sai số ANN vào thí nghiệm.
- Mỗi thí nghiệm chỉ thay đổi một biến chính.
- Mọi kết quả phải tái lập từ config và command đã lưu.

### Ngoài phạm vi smoke test đầu tiên

- Tối ưu cho hàng chục triệu chunk.
- Reranker, BM25, RRF và query translation.
- Fine-tuning và hard-negative training.
- API, dashboard và giao diện demo.
- Kết luận model nào tốt nhất cho bài thi.

Các mục ngoài phạm vi được đưa vào phần mở rộng sau khi dense smoke test chạy
đúng và đã có báo cáo lỗi.

## 3. Lựa chọn kỹ thuật cho đường chạy đầu tiên

| Thành phần | Lựa chọn smoke test | Lý do |
|---|---|---|
| Embedding | `intfloat/multilingual-e5-small` | Nhẹ hơn BGE-M3, phù hợp học workflow trên CPU |
| Query format | tiền tố `query: ` | Theo cách sử dụng của E5 |
| Chunk format | tiền tố `passage: ` | Đồng nhất với cách E5 được huấn luyện |
| Vector scoring | cosine similarity | Dễ hiểu và kiểm tra |
| Vector search | exact search | Mini corpus không cần ANN |
| Retrieval unit | chunk-first | Khớp trực tiếp output của cuộc thi |
| Document score | max chunk score | Baseline rõ ràng, dễ giải thích |
| Selection | fixed top K | Tách bài toán retrieval khỏi threshold calibration |
| Metric | P/R/F2 cho chunk và document | Bám sát thể lệ hiện có |

`multilingual-e5-small` chỉ là model học workflow. Khi đường chạy ổn định, thay
model bằng BGE-M3 chỉ nên là một thí nghiệm cấu hình, không làm đổi data contract
hoặc evaluator. Model card E5:
<https://huggingface.co/intfloat/multilingual-e5-small>.

## 4. Kịch bản dữ liệu

### 4.1 Quy mô

- 5 chủ đề y khoa.
- 3 document cho mỗi chủ đề: một `vi`, một `en`, một `zh`.
- 15 document, khoảng 45-60 chunk.
- 10 query tiếng Việt.
- 6 query dùng để quan sát/tinh chỉnh (`dev`).
- 4 query giữ nguyên để kiểm tra cuối (`test`).

Chia `dev/test` ở đây chỉ để học kỷ luật thực nghiệm; số lượng này không đủ để
kết luận thống kê về chất lượng model.

### 4.2 Chủ đề đề xuất

1. Xử trí sốc phản vệ.
2. Tắc nghẽn đường tiết niệu do sỏi thận.
3. Điều trị ban đầu đái tháo đường type 2.
4. Kiểm soát cơn hen cấp.
5. Tăng huyết áp ở phụ nữ mang thai.

Mỗi chủ đề cần có:

- Một đoạn liên quan trực tiếp.
- Một đoạn cùng bệnh nhưng sai ý định, ví dụ chẩn đoán thay vì điều trị.
- Một đoạn gần nghĩa nhưng sai đối tượng hoặc mức độ.
- Một đoạn không liên quan.
- Ít nhất một positive khác ngôn ngữ với query.

### 4.3 Schema học tập

Document thô:

```json
{"doc_id":"DOC_ANAPHYLAXIS_EN","language":"en","title":"...","text":"..."}
```

Chunk:

```json
{"chunk_id":"DOC_ANAPHYLAXIS_EN_000","doc_id":"DOC_ANAPHYLAXIS_EN","chunk_index":0,"language":"en","text":"..."}
```

Query:

```json
{"id":"Q001","query":"Xử trí ban đầu sốc phản vệ độ 3 như thế nào?"}
```

Ground truth:

```json
{"id":"Q001","relevant_docs":["DOC_ANAPHYLAXIS_EN"],"relevant_chunks":["DOC_ANAPHYLAXIS_EN_001"]}
```

### 4.4 Quy tắc chunk thử nghiệm

- Cắt theo ranh giới câu/đoạn trước.
- Mục tiêu 180-256 token.
- Tối đa khoảng 320 token.
- Overlap một câu hoặc khoảng 32 token.
- Không tách số khỏi đơn vị, tên thuốc hoặc câu hướng dẫn.
- `chunk_id` ổn định khi chạy lại với cùng đầu vào và config.

Đây là chunker phục vụ học tập. Corpus chính thức được BTC nói là đã chunk sẵn;
khi đó pipeline bỏ qua bước chunking và giữ nguyên ID chính thức.

## 5. Definition of Done cho toàn bộ smoke test

Smoke test chỉ được coi là hoàn thành khi:

- Một command tạo được chunks từ documents thô.
- Validator không phát hiện ID trùng, text rỗng hoặc quan hệ cha-con sai.
- Perfect prediction cho F2 bằng 1 và empty prediction cho F2 bằng 0.
- Corpus được encode và index được lưu kèm metadata/config.
- 10 query tạo được ranking chunk có score.
- Ít nhất một query Việt tìm thấy positive Anh/Trung trong top 5.
- Prediction JSONL vượt qua submission validator nội bộ.
- Báo cáo có P/R/F2 ở chunk-level và document-level.
- Có bảng error analysis cho từng query sai.
- Chạy lại cùng seed/config tạo cùng ranking và metric.
- Có một lệnh hoặc checklist tái tạo toàn bộ kết quả từ đầu.

Không đặt mục tiêu F2 cao cho smoke test vì corpus nhỏ và tự tạo. Chất lượng của
smoke test nằm ở khả năng quan sát, giải thích và tái lập.

## 6. Kế hoạch thực hiện theo từng gate

Mỗi gate gồm: yêu cầu, cách suy nghĩ, lựa chọn, cách agent tự vận hành, đầu ra và
điều kiện được đi tiếp. Nếu gate chưa đạt, không thêm kỹ thuật mới.

---

### Gate 0 - Đóng băng hợp đồng thí nghiệm

#### Yêu cầu

Tạo một config duy nhất chứa tối thiểu:

```yaml
experiment_name: smoke_dense_e5
seed: 42
model_name: intfloat/multilingual-e5-small
query_prefix: "query: "
passage_prefix: "passage: "
normalize_embeddings: true
top_k: 5
document_aggregation: max
```

#### Cách suy nghĩ

Nếu model, prefix, K hoặc cách normalize không được ghi lại, hai lần chạy khác
nhau không còn là cùng một thí nghiệm. Config là danh tính của lần chạy.

#### Hướng khác có thể dùng

- BGE-M3 thay E5-small.
- Dot product thay cosine.
- Top K khác 5.

Các lựa chọn này được giữ làm thí nghiệm sau; chưa thay ở Gate 0.

#### Nếu agent tự vận hành

1. Tạo config.
2. Kiểm tra schema config và từ chối khóa lạ.
3. In toàn bộ config vào đầu log chạy.
4. Lưu một bản config cạnh kết quả thí nghiệm.

#### Đầu ra

- `configs/smoke_dense_e5.yaml`
- Một bảng giả định: điều đã biết, điều đang giả định, điều chờ BTC xác nhận.

#### Điều kiện đi tiếp

Config đọc được, có giá trị mặc định rõ ràng và không chứa thông số ẩn trong code.

---

### Gate 1 - Xây mini benchmark và gán nhãn

#### Yêu cầu

Tạo documents, queries và ground truth theo schema ở mục 4. Nội dung phải có
positive, hard negative và dữ liệu ba ngôn ngữ.

#### Cách suy nghĩ

Một benchmark chỉ có positive rất dễ và không cho biết model đang hiểu nghĩa hay
chỉ nhận ra từ khóa. Hard negative phải cùng chủ đề nhưng sai ở một điều kiện có
ý nghĩa lâm sàng.

Ví dụ:

```text
Query: điều trị ban đầu sốc phản vệ
Positive: adrenaline là điều trị đầu tay
Hard negative: tiêu chuẩn nhận biết sốc phản vệ
Hard negative đối lập: không cần sử dụng adrenaline
```

#### Hướng khác có thể dùng

- Tải guideline công khai và trích đoạn có dẫn nguồn.
- Viết toy text ngắn chỉ để kiểm tra logic phần mềm.

Smoke test đầu dùng toy text có kiểm soát. Trước khi dùng nội dung làm benchmark
chất lượng, cần nguồn y khoa và giấy phép rõ ràng.

#### Nếu agent tự vận hành

1. Tạo danh sách chủ đề và ma trận `query x language x intent`.
2. Viết hoặc thu thập documents với nguồn/giấy phép được ghi lại.
3. Tạo query mà không sao chép nguyên câu trong positive.
4. Gán nhãn thủ công theo định nghĩa relevance của thể lệ.
5. Kiểm tra mỗi query có positive và hard negative.
6. Chia dev/test trước khi nhìn kết quả model.

#### Đầu ra

- `data/smoke/raw/documents.jsonl`
- `data/smoke/queries_dev.jsonl`
- `data/smoke/queries_test.jsonl`
- `data/smoke/ground_truth_dev.jsonl`
- `data/smoke/ground_truth_test.jsonl`
- `data/smoke/README.md` ghi nguồn và quy tắc nhãn.

#### Điều kiện đi tiếp

100% query có nhãn, có positive chéo ngôn ngữ và có ít nhất một hard negative.

---

### Gate 2 - Chunking và kiểm tra dữ liệu

#### Yêu cầu

Biến documents thô thành chunks, giữ mapping hai chiều:

```text
chunk_id -> doc_id
doc_id -> [chunk_id]
```

#### Cách suy nghĩ

Retrieval đúng nội dung nhưng sai ID vẫn tạo submission sai. Vì vậy data integrity
được kiểm tra trước model.

#### Hướng khác có thể dùng

- Fixed token windows.
- Sentence-aware chunking.
- Semantic chunking.

Smoke test dùng sentence-aware token window. Semantic chunking làm khó xác định
nguyên nhân lỗi và chưa cần thiết ở quy mô nhỏ.

#### Nếu agent tự vận hành

1. Chạy chunker bằng config cố định.
2. Kiểm tra ID duy nhất, index không trùng và text không rỗng.
3. Kiểm tra mọi nhãn chunk đều tồn tại.
4. Kiểm tra mọi chunk nhãn có document cha nằm trong nhãn document.
5. Xuất thống kê số document, chunk, ngôn ngữ và độ dài.

#### Đầu ra

- `data/smoke/processed/chunks.jsonl`
- `data/smoke/processed/doc_map.json`
- `outputs/smoke/data_audit.json`

#### Điều kiện đi tiếp

Không có lỗi integrity và dữ liệu chạy lại tạo cùng ID.

---

### Gate 3 - Xác minh evaluator trước khi chạy model

#### Yêu cầu

Chứng minh evaluator hoạt động bằng các prediction có kết quả biết trước.

#### Cách suy nghĩ

Nếu metric sai, mọi quyết định model sau đó đều vô nghĩa. Evaluator phải được
kiểm thử trước retriever.

#### Các ca bắt buộc

1. Prediction giống ground truth: P=R=F2=1.
2. Prediction rỗng: P=R=F2=0.
3. Prediction có một đúng, một sai: kiểm tra bằng tính tay.
4. Query bị thiếu hoặc thừa: evaluator báo lỗi rõ ràng.
5. Chunk có parent không được dự đoán: validator chặn.

#### Nếu agent tự vận hành

1. Tạo fixture nhỏ cho từng ca.
2. Tính tay expected metric.
3. Chạy evaluator và so sánh với expected.
4. Chỉ sửa evaluator nếu sai; chưa chạm vào retriever.

#### Đầu ra

- Test cho metric và submission contract.
- Một ví dụ tính P/R/F2 bằng tay trong báo cáo.

#### Điều kiện đi tiếp

Tất cả ca metric và validator đều đạt.

---

### Gate 4 - Encode corpus và tạo exact vector index

#### Yêu cầu

Tạo một vector cho mỗi chunk bằng cùng model và cùng preprocessing.

#### Cách suy nghĩ

Đây là đường offline:

```text
chunks -> passage prefix -> tokenizer -> model -> normalized vectors -> index
```

Corpus embedding không được tính lại cho từng query.

#### Hướng khác có thể dùng

- E5-small: ưu tiên tốc độ học workflow.
- BGE-M3: ứng viên chất lượng cao hơn nhưng lớn hơn đáng kể.
- API embedding: nhanh triển khai nhưng phụ thuộc mạng và điều khoản dữ liệu.

Smoke test chọn E5-small chạy local. Sau này đổi model qua config.

#### Nếu agent tự vận hành

1. Kiểm tra model cache; tải model nếu chưa có.
2. Encode theo batch phù hợp CPU/GPU.
3. Normalize vector.
4. Kiểm tra số vector bằng số chunk và không có NaN/Inf.
5. Lưu vectors, thứ tự `chunk_id`, model name và hash/config.
6. Load lại index và kiểm tra một vector mẫu giống trước khi lưu.

#### Đầu ra

- `artifacts/smoke_dense/embeddings.npy`
- `artifacts/smoke_dense/chunk_ids.json`
- `artifacts/smoke_dense/manifest.json`

#### Điều kiện đi tiếp

Index load lại được, số vector khớp metadata và không có vector lỗi.

---

### Gate 5 - Truy hồi query và quan sát ranking

#### Yêu cầu

Với mỗi query:

```text
query -> query prefix -> embedding -> cosine với corpus -> top K chunks
```

Lưu cả `chunk_id`, raw score, rank, language, doc_id và một đoạn text preview.

#### Cách suy nghĩ

Prediction cuối chỉ chứa ID, nhưng debug cần biết model đã xếp nội dung gì và vì
sao một positive bị đẩy xuống.

#### Hướng khác có thể dùng

- Fixed K.
- Absolute threshold.
- Relative threshold.

Smoke test đầu chạy K trong `[1, 3, 5, 10]`, nhưng đây là các evaluation run độc
lập. Không thay model hoặc dữ liệu cùng lúc.

#### Nếu agent tự vận hành

1. Encode toàn bộ query theo batch.
2. Tính similarity chính xác với toàn corpus nhỏ.
3. Sort ổn định: score giảm dần, `chunk_id` làm tie-break.
4. Lưu full ranking đủ sâu để đánh giá nhiều K mà không encode lại.
5. Kiểm tra thủ công ít nhất một query có kết quả VI, EN và ZH.

#### Đầu ra

- `outputs/smoke/rankings.jsonl`
- `outputs/smoke/predictions_k{K}.jsonl`

#### Điều kiện đi tiếp

Mọi query có ranking hợp lệ; có thể lần theo từ query đến text của từng kết quả.

---

### Gate 6 - Tổng hợp document và tạo submission

#### Yêu cầu

Dùng ranking chunk để tạo hai danh sách:

```text
relevant_chunks
relevant_docs
```

#### Cách suy nghĩ

Một document có thể có nhiều chunk trong top K. Document chỉ xuất hiện một lần và
thứ hạng của nó được xác định bởi chunk có điểm cao nhất trong baseline.

#### Nếu agent tự vận hành

1. Chọn top K chunk.
2. Map từng chunk về document cha.
3. Gán `doc_score = max(chunk_score)`.
4. Sort documents theo score và loại trùng.
5. Enforce parent consistency.
6. Validate JSONL trước khi đánh giá.

#### Đầu ra

- Prediction JSONL đúng schema cuộc thi đang giả định.
- Báo cáo validator với 0 unknown ID và 0 parent violation.

#### Điều kiện đi tiếp

Tất cả prediction vượt validator.

---

### Gate 7 - Đánh giá và chọn K trên dev

#### Yêu cầu

Đánh giá riêng document và chunk cho K trong `[1, 3, 5, 10]`.

#### Cách suy nghĩ

- K nhỏ thường tăng Precision nhưng giảm Recall.
- K lớn thường tăng Recall nhưng thêm false positive.
- F2 ưu tiên Recall, nhưng vẫn phạt việc trả quá nhiều kết quả sai.

Không chọn K bằng điểm test. Dùng dev để chọn, sau đó chạy test đúng một cấu hình
đã chốt.

#### Nếu agent tự vận hành

1. Tạo bảng metric theo K.
2. Chọn K có dev Macro F2 tốt nhất; nếu hòa, chọn cấu hình đơn giản/nhanh hơn.
3. Khóa config.
4. Chạy một lần trên test.
5. Ghi rõ kết quả test là kiểm tra smoke, không phải ước lượng điểm thi.

#### Đầu ra

| K | Chunk P | Chunk R | Chunk F2 | Doc P | Doc R | Doc F2 | Macro F2 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | ... | ... | ... | ... | ... | ... | ... |
| 3 | ... | ... | ... | ... | ... | ... | ... |
| 5 | ... | ... | ... | ... | ... | ... | ... |
| 10 | ... | ... | ... | ... | ... | ... | ... |

#### Điều kiện đi tiếp

Có K được chọn bằng quy tắc định trước và test chưa bị dùng để tinh chỉnh.

---

### Gate 8 - Error analysis

#### Yêu cầu

Mỗi query sai phải được gán ít nhất một nhóm lỗi.

#### Nhóm lỗi

| Mã | Nhóm lỗi | Dấu hiệu | Hướng xử lý sau này |
|---|---|---|---|
| E-DATA | Dữ liệu/nhãn | Nhãn thiếu hoặc không nhất quán | Sửa benchmark |
| E-CHUNK | Chunking | Ý bị cắt hoặc thiếu ngữ cảnh | Đổi chunk/context window |
| E-LANG | Cross-lingual | Chỉ tìm đúng cùng ngôn ngữ | Đổi model, dịch query |
| E-TERM | Thuật ngữ | Mất tên thuốc, số, viết tắt | BM25/query expansion |
| E-INTENT | Sai ý định | Đúng bệnh, sai điều trị/chẩn đoán | Reranker/hard negative |
| E-CONTRA | Đối lập | Candidate phủ định lõi query | Reranker/entity rule |
| E-SELECT | Chọn K | Positive có trong ranking nhưng bị cắt | Threshold/K calibration |
| E-DOC | Tổng hợp doc | Chunk đúng nhưng doc rank sai | Đổi aggregation |

#### Nếu agent tự vận hành

1. Với mỗi false negative, tìm rank thật của positive.
2. Với mỗi false positive, so sánh candidate với lõi query.
3. Gán mã lỗi và ghi bằng chứng từ text/score.
4. Đếm tần suất từng nhóm.
5. Đề xuất đúng một thay đổi cho lỗi phổ biến nhất.
6. Không triển khai thay đổi cho đến khi báo cáo được duyệt.

#### Đầu ra

- `experiments/smoke_dense_e5/error_analysis.csv`
- `experiments/smoke_dense_e5/report.md`

Các cột tối thiểu:

```text
query_id, query, expected_id, predicted_id, expected_rank,
language, error_type, evidence, proposed_next_experiment
```

#### Điều kiện đi tiếp

Có thể giải thích từng lỗi bằng dữ liệu cụ thể, không dùng mô tả chung như
"model chưa tốt".

---

### Gate 9 - Kiểm tra tái lập và bàn giao

#### Yêu cầu

Một người khác có thể làm theo README và tạo lại kết quả.

#### Nếu agent tự vận hành

1. Xóa cache đầu ra trong một thư mục thử nghiệm tạm, không xóa dữ liệu người dùng.
2. Chạy pipeline từ documents đến report.
3. So sánh config, số record, ranking và metric.
4. Chạy unit/integration tests.
5. Ghi thời gian từng bước và môi trường thực thi.

#### Đầu ra

- Một command hoặc Make target chạy toàn smoke test.
- README hướng dẫn.
- Manifest gồm Python version, packages, model và config.

#### Điều kiện hoàn thành

Lần chạy lại cho cùng kết quả và toàn bộ Definition of Done ở mục 5 được đánh dấu.

## 7. Cách agent xử lý sự cố trong lúc tự vận hành

| Sự cố | Kiểm tra đầu tiên | Cách xử lý |
|---|---|---|
| Model không tải được | mạng, dung lượng, model ID | báo rõ blocker; không tự đổi model âm thầm |
| Hết RAM/VRAM | batch size, sequence length | giảm batch; giữ nguyên model/config logic |
| Vector có NaN | input rỗng, dtype, normalize | dừng indexing và sửa dữ liệu/inference |
| Không tìm thấy cross-lingual positive | prefix, language data, model | kiểm tra từng tầng trước khi đổi model |
| F2 bất thường | evaluator fixtures | xác minh metric trước retrieval |
| Parent violation | doc map và selection | sửa mapping/consistency, không sửa nhãn |
| K lớn luôn thắng | benchmark thiếu negatives | xem lại dữ liệu trước khi kết luận |
| Kết quả đổi giữa các lần chạy | seed, sort tie-break, model mode | bật deterministic behavior và ghi manifest |

Nguyên tắc xử lý: xác định tầng gây lỗi, tạo kiểm tra nhỏ tái hiện lỗi, sửa đúng
tầng đó, chạy lại gate liên quan rồi mới tiếp tục.

## 8. Các thí nghiệm sau khi smoke test đạt

### E02 - BM25-only

Giữ nguyên data, query, metric và K grid; chỉ đổi retriever. Mục tiêu là xác định
BM25 thắng dense ở thuật ngữ, số, liều hoặc viết tắt nào.

### E03 - Hybrid bằng RRF

Lấy ranking BM25 và dense đã lưu, hợp nhất bằng Reciprocal Rank Fusion. Chưa thêm
reranker. Đo mức tăng Recall và số false positive mới.

### E04 - Multilingual reranker

Rerank top candidates của hybrid. Đo khả năng sửa `E-INTENT` và `E-CONTRA`. Đồng
thời ghi latency vì reranker chỉ có ý nghĩa khi candidate pool đủ nhỏ.

### E05 - Query expansion/translation

Tạo route query VI/EN/ZH cho lexical retrieval. Đánh giá riêng từng route rồi mới
fusion, tránh không biết bản dịch nào gây nhiễu.

### E06 - Dynamic selection

So sánh fixed K, absolute threshold, relative threshold và score gap. Chọn trên
dev theo Macro F2.

### E07 - Model upgrade

Thay E5-small bằng BGE-M3 trong cùng pipeline. Không đổi K, data hoặc evaluator ở
lần so sánh đầu. Sau đó mới hiệu chỉnh lại selection nếu model tốt hơn.

### E08 - Fine-tuning

Chỉ mở khi có đủ nhãn và error analysis chỉ ra lỗi biểu diễn. Tạo positives và
hard negatives từ chính các lỗi retrieval/reranking; giữ một test set chưa dùng.

## 9. Khi dữ liệu chính thức được phát hành

Thứ tự thích nghi pipeline:

1. Đọc đặc tả, không chạy model ngay.
2. Viết adapter sang schema nội bộ; giữ nguyên ID BTC.
3. Chạy data audit và validator.
4. Xác minh evaluator chính thức khác giả định hiện tại ở đâu.
5. Chạy baseline trên một sample nhỏ.
6. Ước lượng thời gian encode, RAM và dung lượng index trên toàn corpus.
7. Chọn ANN/index theo quy mô và tài nguyên.
8. Chạy dense, BM25, hybrid và reranker như các thí nghiệm tách biệt.
9. Calibrate selection trên dev.
10. Tạo submission, validate và tái lập từ manifest.

Những thành phần được giữ lại gồm schema nội bộ, validator, evaluator, experiment
config, logging, pipeline interface và error taxonomy. Toy documents, chunker học
tập, model mặc định và mọi threshold đều có thể phải thay.

## 10. Nhịp làm việc đề xuất cho người mới

### Buổi 1 - Dữ liệu và metric

- Hoàn thành Gate 0-3.
- Tự tính một ví dụ Precision/Recall/F2.
- Tự giải thích parent consistency.

### Buổi 2 - Embedding và retrieval

- Hoàn thành Gate 4-5.
- Mở ranking của từng query và đọc top 5 bằng mắt.
- Giải thích khác biệt giữa offline indexing và online query.

### Buổi 3 - Submission và đánh giá

- Hoàn thành Gate 6-7.
- So sánh K=1/3/5/10.
- Chọn K bằng dev, không nhìn test để điều chỉnh.

### Buổi 4 - Phân tích lỗi và tái lập

- Hoàn thành Gate 8-9.
- Chọn một thí nghiệm tiếp theo dựa trên lỗi phổ biến nhất.
- Chạy lại toàn bộ từ config.

Khi kết thúc mỗi buổi, người học nên viết lại bằng lời của mình: đầu vào, biến đổi,
đầu ra, điều đã kiểm chứng và điều vẫn chỉ là giả định. Đây là cách biến smoke test
thành hiểu biết có thể chuyển sang dữ liệu cuộc thi.

# Kế hoạch chuyển sang Medical-first Benchmark

## 1. Quyết định kiến trúc

Hướng chính của dự án là **multilingual medical retrieval**. Từ 2026-09-25,
XQuAD đã được loại khỏi repository theo quyết định của project owner. MedQuAD là
benchmark public đầu tiên để kiểm tra pipeline medical end-to-end; benchmark
đa ngôn ngữ có clinical review vẫn là đích tiếp theo.

Benchmark chính mới là `medical_smoke_v1`:

```text
nguồn y khoa có kiểm chứng
    -> document đa ngôn ngữ có metadata nguồn
    -> chunking với nhiều chunk trên mỗi document
    -> query tiếng Việt theo ý định lâm sàng
    -> candidate pooling từ BM25 và dense retrieval
    -> đánh giá relevance thủ công
    -> chọn cấu hình trên dev
    -> khóa config
    -> chạy test đúng một lần
```

Nguyên tắc quan trọng:

```text
MedQuAD v0:     qrel cấu trúc từ cặp question-answer
Medical review: candidate chưa được đánh giá -> chưa biết, không mặc định negative
```

Nhờ đó, một đoạn khác ID nhưng vẫn trả lời đúng query sẽ không tự động bị tính là
false positive.

## 2. Phạm vi benchmark

### 2.1 Chủ đề ban đầu

1. Xử trí sốc phản vệ.
2. Đánh giá và xử trí sỏi thận/sỏi niệu quản.
3. Điều trị ban đầu đái tháo đường type 2.
4. Đánh giá và xử trí cơn hen cấp.
5. Tăng huyết áp thai kỳ, tiền sản giật và sản giật.

### 2.2 Nguồn tham chiếu y khoa

Nguồn tham chiếu ban đầu:

- Resuscitation Council UK, *Emergency treatment of anaphylaxis*:
  https://www.resus.org.uk/library/additional-guidance/guidance-anaphylaxis/emergency-treatment-anaphylactic-reactions>
- NICE NG118, *Renal and ureteric stones: assessment and management*:
  <https://www.nice.org.uk/guidance/ng118/chapter/recommendations>
- NICE NG28, *Type 2 diabetes in adults: management*:
  <https://www.nice.org.uk/guidance/NG28/chapter/initial-medicines>
- Global Initiative for Asthma, strategy reports:
  <https://ginasthma.org/reports/>
- WHO, *Recommendations for prevention and treatment of pre-eclampsia and
  eclampsia*:
  <https://www.who.int/publications/i/item/9789241548335>

Nguồn dùng để xác minh nội dung không đồng nghĩa với quyền sao chép nguyên văn.
Trước khi đưa text vào repository phải kiểm tra license. Nếu không có quyền tái
sử dụng, benchmark chỉ dùng bản tóm tắt do dự án tự viết, có dẫn nguồn, không sao
chép đoạn dài và phải được người có chuyên môn kiểm tra.

Mọi nội dung trong benchmark chỉ phục vụ đánh giá retrieval, không phải hướng dẫn
điều trị hay tư vấn y khoa.

## 3. Mô hình dữ liệu

### 3.1 Source group

Mỗi chủ đề là một source group:

```text
MED_ANAPHYLAXIS
MED_RENAL_STONE
MED_T2D
MED_ACUTE_ASTHMA
MED_PREECLAMPSIA
```

Mỗi group có document `vi`, `en`, `zh` được xây dựng từ cùng một nội dung y khoa
đã kiểm chứng:

```text
MED_ANAPHYLAXIS
|-- MED_ANAPHYLAXIS_vi
|-- MED_ANAPHYLAXIS_en
`-- MED_ANAPHYLAXIS_zh
```

Không mặc định toàn bộ chunk trong ba document là relevant. Relevance được gán ở
cấp chunk theo query và ý định lâm sàng.

### 3.2 Document nhiều chunk

Một document phải có nhiều chunk để kiểm tra chunking và document aggregation:

```text
MED_ANAPHYLAXIS_vi
|-- MED_ANAPHYLAXIS_vi_000  # nhận biết/chẩn đoán
|-- MED_ANAPHYLAXIS_vi_001  # xử trí ban đầu
|-- MED_ANAPHYLAXIS_vi_002  # theo dõi sau cấp cứu
`-- MED_ANAPHYLAXIS_vi_003  # dự phòng/auto-injector
```

Ví dụ query:

```text
Thuốc ưu tiên trong xử trí ban đầu sốc phản vệ là gì?
```

Các chunk `_001` bằng `vi`, `en`, `zh` có thể là positive. Các chunk chẩn đoán,
theo dõi và dự phòng là hard negatives cùng bệnh nhưng sai ý định.

### 3.3 Cấu trúc thư mục dự kiến

```text
data/medical_smoke/
|-- raw/
|   `-- documents.jsonl
|-- processed/
|   |-- chunks.jsonl
|   `-- doc_map.json
|-- sources.jsonl
|-- queries_dev.jsonl
|-- queries_test.jsonl
|-- qrels_dev.jsonl
|-- qrels_test.jsonl
|-- annotation_guidelines.md
`-- README.md
```

## 4. Quy tắc annotation

### 4.1 Relevance grades

Candidate được đánh giá theo bốn mức:

```text
 2 = trả lời trực tiếp và đầy đủ ý định query
 1 = liên quan và hỗ trợ một phần
 0 = không trả lời query hoặc sai ý định
-1 = chưa được đánh giá
```

Khi tính metric nhị phân ban đầu:

```text
relevant = label 1 hoặc 2
negative = label 0
unjudged = không tự động xem là negative
```

### 4.2 Candidate pooling

Không tạo negative chỉ bằng group ID. Dùng pooling:

1. BM25 lấy top candidates.
2. Dense E5 lấy top candidates.
3. Hợp nhất và loại trùng.
4. Reviewer đọc query và candidate.
5. Gán relevance grade và error tag nếu cần.

Pool cần chứa:

- Positive trực tiếp.
- Positive khác ngôn ngữ khi phù hợp.
- Hard negative cùng bệnh nhưng sai ý định.
- Candidate chứa thuật ngữ gần giống.
- Candidate có phủ định hoặc chống chỉ định.
- Candidate đúng chủ đề nhưng sai đối tượng hay mức độ nặng.

### 4.3 Chống leakage

Các query là paraphrase hoặc cùng clinical intent phải nằm trong cùng query
group. Toàn bộ query group chỉ được gán vào dev hoặc test.

```text
ANAPHYLAXIS_INITIAL_TREATMENT
|-- paraphrase 1
|-- paraphrase 2
`-- paraphrase 3
```

Corpus là một kho cố định dùng chung. Query và qrels test được khóa kín. Không
chia query ngẫu nhiên nếu chúng dùng cùng một ý định hoặc cùng một template.

## 5. Kế hoạch cải tiến Gate 0-7

### Gate 0 - Khóa hợp đồng medical experiment

Tạo `configs/smoke_medical_dense_e5.yaml` với các nhóm cấu hình rõ ràng:

```yaml
experiment:
  name: smoke_medical_dense_e5
  seed: 42

data:
  dataset: medical_smoke_v1
  query_language: vi
  corpus_languages: [vi, en, zh]
  chunks: data/medical_smoke/processed/chunks.jsonl

embedding:
  model_name: intfloat/multilingual-e5-small
  model_revision: "<pinned-revision>"
  pooling: mean
  query_prefix: "query: "
  passage_prefix: "passage: "
  normalize: true
  max_length: 512
  batch_size: 32

retrieval:
  type: exact
  similarity: cosine
  ranking_depth: all

selection:
  type: fixed_k
  top_k_values: [1, 3, 5, 10]

document:
  aggregation: max
```

Điều kiện đạt:

- Không còn tham số ảnh hưởng kết quả bị ẩn trong code.
- Pin model revision và tokenizer revision.
- Config schema từ chối field lạ.
- Manifest lưu config hash.
- Output lưu snapshot config đã dùng.

### Gate 1 - Medical benchmark và ground truth

- Tạo documents đa ngôn ngữ có `source_id`, URL, ngày truy cập và license.
- Tạo query tiếng Việt theo clinical intent, không sao chép nguyên câu positive.
- Tạo hard negatives có chủ đích.
- Dùng pooling và review để tạo qrels.
- Chia dev/test trước khi chạy model.
- Có annotation guideline và bước adjudication.

Điều kiện đạt:

- 100% query có ít nhất một positive được review.
- Có positive cross-lingual.
- Có hard negative cho từng nhóm intent chính.
- Không dùng `khác group = negative`.
- Nguồn và quyền tái sử dụng được ghi lại.

### Gate 2 - Chunking và data audit

Chunker dùng ranh giới câu/đoạn, mục tiêu 180-256 token, giới hạn khoảng 320
token và overlap hợp lý. Không tách tên thuốc khỏi liều hoặc câu hướng dẫn.

Artifact:

```text
data/medical_smoke/processed/chunks.jsonl
data/medical_smoke/processed/doc_map.json
outputs/medical_smoke/data_audit.json
```

Audit kiểm tra:

- Duplicate ID.
- Text rỗng.
- `chunk_index` trùng hoặc không liên tục.
- Mapping chunk-document.
- Ground-truth ID không tồn tại.
- Parent consistency.
- Phân bố ngôn ngữ và độ dài.
- Có ít nhất một document chứa nhiều chunk.
- Chạy lại cùng config tạo cùng ID.

### Gate 3 - Evaluator nghiêm ngặt

Evaluator mặc định phải báo lỗi khi:

- Thiếu hoặc thừa query.
- Query ID trùng.
- Unknown chunk/document.
- Ground truth rỗng.
- Parent violation.
- Prediction có ID trùng.

Bổ sung test cho trường hợp document đúng nhưng chunk sai, document nhiều chunk,
doc/chunk metric khác nhau và unjudged candidate.

### Gate 4 - Index corpus độc lập

Tách build index khỏi query retrieval:

```text
build_dense_index -> embeddings/chunk IDs/manifest
run_retrieval     -> query rankings
```

Manifest index cần có model/tokenizer revision, pooling, dtype, dimension, chunk
order hash và embedding checksum. Index phải load lại được và cho cùng ranking.

### Gate 5 - Full medical ranking

Với corpus smoke nhỏ, lưu ranking toàn bộ thay vì chỉ top 10. `ranking_depth` và
`selection K` là hai khái niệm riêng:

```text
ranking_depth = all          # phục vụ phân tích
top_k_values = [1,3,5,10]    # phục vụ prediction/evaluation
```

Ranking lưu `rank`, `chunk_id`, `doc_id`, `language`, `score` và text preview.
Nhờ đó có thể tìm exact rank của mọi positive trong Gate 8.

### Gate 6 - Document aggregation thực tế

Baseline dùng:

```text
doc_score = max(score của các chunk thuộc document)
```

Phải có integration test với nhiều chunk cùng document, loại document trùng,
giữ đúng thứ tự theo score tốt nhất và enforce parent consistency.

### Gate 7 - Chọn K trên medical dev

Quy tắc phải được khai báo trước:

1. Chọn dev Macro F2 cao nhất.
2. Nếu chênh lệch nhỏ hơn `0.001`, xem như hòa.
3. Khi hòa, chọn K nhỏ hơn.
4. Không dùng test để chọn model, K hoặc threshold.
5. Chỉ chạy test sau khi qrels, evaluator và config đã khóa.

Không dùng kết quả benchmark đã bị loại để chọn cấu hình medical.

## 6. Thứ tự triển khai

### P0 - Baseline medical public

1. Chuyển MedQuAD thành corpus/query/qrels tách biệt.
2. Chia theo source document để chống leakage.
3. Chạy BM25 trước, sau đó dense, hybrid và reranker trên cùng dev split.

### P1 - Medical Gate 0-3

1. Tạo config và schema medical experiment.
2. Tạo source registry và kiểm tra license.
3. Viết annotation guideline.
4. Tạo raw documents đa ngôn ngữ.
5. Chạy chunker và tạo data audit.
6. Pool candidates và review qrels dev.
7. Siết evaluator/validator và bổ sung test.

### P2 - Medical Gate 4-7

1. Build và load lại exact dense index.
2. Sinh full ranking trên dev.
3. Kiểm tra cross-lingual retrieval.
4. Tạo document/chunk predictions.
5. Đánh giá K theo quy tắc đã khóa.
6. Khóa config tốt nhất.
7. Chạy test đúng một lần.

### P3 - Gate 8-9

1. Error analysis cho từng query dev sai.
2. Chọn đúng một thí nghiệm tiếp theo dựa trên lỗi phổ biến nhất.
3. Cập nhật experiment log.
4. Chạy reproducibility trong output directory mới.
5. Bàn giao README, manifest và một command chạy toàn pipeline.

## 7. Definition of Done

Medical benchmark chỉ được coi là hoàn thành khi:

- Config chứa mọi quyết định ảnh hưởng kết quả.
- Corpus có nguồn và quyền tái sử dụng rõ ràng.
- Nội dung y khoa đã được review.
- Một document có nhiều chunk và ID ổn định.
- Query có positive, hard negative và nhãn relevance đã duyệt.
- Unjudged candidate không tự động bị coi là negative.
- Evaluator strict với query và ID universe.
- Index được lưu/load độc lập.
- Full ranking cho phép tìm exact rank của positive.
- Document aggregation được kiểm thử trên dữ liệu nhiều chunk.
- K được chọn trước test bằng quy tắc định trước.
- Test chỉ được chạy sau khi khóa benchmark và config.
- Có error analysis và một lệnh tái tạo toàn bộ kết quả.

## 8. Điểm dừng an toàn tiếp theo

Mốc tiếp theo là hoàn thành Gate 0-2 của `medical_smoke_v1`:

1. Config contract.
2. Source/license registry.
3. Annotation guideline.
4. Raw medical documents.
5. Chunker, doc map và data audit.

Chưa chạy model hoặc tạo test metric trước khi các thành phần trên được khóa.

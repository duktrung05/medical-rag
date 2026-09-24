# Research log

**Status:** In progress  
**Started:** 2026-09-23  
**Project:** Road to AI 2026 — Multilingual Medical Retrieval

Tài liệu này ghi lại nguồn đã kiểm tra, phiên bản, quyền tái sử dụng, kết luận và
uncertainty còn lại. Một URL xuất hiện ở đây không tự động có nghĩa nội dung tại
URL đó được phép sao chép vào corpus.

## Evidence policy

Thứ tự ưu tiên:

1. Đề/specification chính thức của BTC.
2. Trang và tài liệu của tổ chức phát hành guideline y khoa.
3. Model card, repository và documentation chính thức của model/tokenizer.
4. Paper gốc khi quyết định phụ thuộc kết quả nghiên cứu.
5. Issue tracker chỉ dùng để điều tra lỗi triển khai.

Trạng thái reuse dùng trong log:

- `CLEARED`: điều khoản đã được đọc và phù hợp phạm vi sử dụng dự kiến.
- `CONDITIONAL`: chỉ được dùng khi tuân thủ điều kiện cụ thể.
- `REFERENCE-ONLY`: dùng để kiểm tra fact; chưa đưa nguyên văn vào corpus.
- `PERMISSION-REQUIRED`: cần văn bản cho phép trước khi tái sản xuất/phân phối.
- `UNKNOWN`: chưa đủ bằng chứng để kết luận.

---

## R-001: Hợp đồng bài toán công khai của Road to AI 2026

- **Question:** Query dùng ngôn ngữ nào, corpus gồm ngôn ngữ nào và hệ thống phải
  trả document, chunk hay câu trả lời?
- **Search terms:** `Road to AI 2026 medical retrieval query Vietnamese VI EN ZH`,
  `R2AI relevant_docs relevant_chunks Macro F2`.
- **Primary sources checked:**
  - [Road to AI 2026 — trang chính thức](https://r2ai.aiguru.com.vn/)
- **Version/date accessed:** Trang web động, truy cập 2026-09-23; chưa thấy số
  phiên bản specification hoặc file kỹ thuật có checksum.
- **License/reuse status:** `REFERENCE-ONLY`. Có thể liên kết và ghi lại fact;
  chưa xác minh điều khoản cho việc sao chép toàn bộ nội dung trang.
- **Findings:**
  - Query được mô tả là query y khoa tiếng Việt.
  - Corpus được mô tả gồm tiếng Việt, tiếng Anh và tiếng Trung.
  - Task yêu cầu cả document retrieval và chunk retrieval.
  - Trang mô tả cross-lingual matching và retrieval-only, không sinh văn bản.
  - Output minh họa dùng `RELEVANT_DOCS` và `RELEVANT_CHUNKS`.
  - Precision, Recall và Macro F2 được nêu là metric; F2 ưu tiên Recall.
- **Remaining uncertainty:**
  - Chưa có exact JSONL schema, casing field, ordering, giới hạn số ID hoặc rule
    cho prediction rỗng.
  - Chưa có công thức kết hợp document/chunk metric và cách macro-average chính
    thức.
  - Chưa biết corpus cố định hay thay đổi giữa các vòng.
- **Decision affected:**
  - Xác nhận product contract `query vi -> corpus vi/en/zh -> doc/chunk IDs`.
  - Không xác nhận các chi tiết submission/evaluator đang là quy ước nội bộ.

## R-002: Specification kỹ thuật và evaluator chính thức

- **Question:** Có tài liệu chính thức nào chốt schema, metric, data contract và
  runtime constraint không?
- **Search terms:** `R2AI 2026 technical specification submission JSONL`,
  `Road to AI medical retrieval evaluator Macro F2`, `relevant_docs
  relevant_chunks`.
- **Primary sources checked:**
  - [Road to AI 2026 — trang chính thức](https://r2ai.aiguru.com.vn/)
  - Repository nội bộ: `README.md`, `docs/00_project_brief.md`, schema và
    submission validator hiện có.
- **Version/date accessed:** 2026-09-23.
- **License/reuse status:** Không áp dụng cho contract nội bộ; official spec vẫn
  `UNKNOWN` vì chưa được tìm thấy trong nguồn công khai đã kiểm tra.
- **Findings:**
  - Trang công khai đủ để xác nhận dạng bài toán nhưng chưa đủ để implement
    submission cuối cùng một cách chắc chắn.
  - Schema `id`, `relevant_docs`, `relevant_chunks`, parent consistency và công
    thức composite Macro F2 hiện là contract nội bộ.
- **Remaining uncertainty:** Toàn bộ câu hỏi contract được liệt kê tại
  `docs/00_project_brief.md#open-questions` vẫn cần đề kỹ thuật hoặc fixture chính
  thức của BTC.
- **Decision affected:**
  - Không đánh dấu project brief là `accepted`.
  - Giữ adapter và evaluator tách rời để thay được khi BTC phát hành spec.
  - Không khóa submission pipeline dựa riêng trên trang marketing.

## R-003: `intfloat/multilingual-e5-small` và preprocessing bắt buộc

- **Question:** Model hỗ trợ workflow nào, cần prefix/pooling/normalization gì,
  giới hạn token và license ra sao?
- **Search terms:** `multilingual-e5-small model card query passage prefix mean
  pooling normalize max length license`.
- **Primary sources checked:**
  - [Model card `intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small)
  - [Pinned model card revision](https://huggingface.co/intfloat/multilingual-e5-small/blob/ada7b62be30f82b0bc5da131b0477721c8fc14e9/README.md)
  - [Tokenizer config](https://huggingface.co/intfloat/multilingual-e5-small/blob/main/tokenizer_config.json)
  - Local Hugging Face snapshot
    `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- **Version/date accessed:** 2026-09-23. Snapshot local hiện dùng revision
  `614241f622f53c4eeff9890bdc4f31cfecc418b3`.
- **License/reuse status:** `CLEARED` cho model code/weights theo model card:
  MIT. Dataset hoặc nội dung đưa vào model không kế thừa license này.
- **Findings:**
  - Model card yêu cầu prefix `query: ` và `passage: ` kể cả với text không phải
    tiếng Anh; thiếu prefix có thể làm giảm chất lượng.
  - Ví dụ chính thức dùng attention-mask-aware average/mean pooling rồi L2
    normalization.
  - Tokenizer có `model_max_length = 512`; text dài hơn bị truncate.
  - Config công bố hidden size/embedding dimension 384.
  - Với embedding đã normalize, inner product dùng trong exact search tương
    đương cosine similarity.
  - Model card cảnh báo version PyTorch/Transformers khác nhau có thể tạo chênh
    lệch số nhỏ.
- **Local artifact checksums:**
  - `config.json`:
    `69137736cab8b8903a07fe8afaafdda25aac55415a12a55d1bffa9f581abf959`
  - `tokenizer_config.json`:
    `a1d6bc8734a6f635dc158508bef000f8e2e5a759c7d92f984b2c86e5ff53425b`
  - `tokenizer.json`:
    `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`
  - `model.safetensors`:
    `1a55775f53449dac10a2bcbc312469fac40b96d53198c407081a831f81c98477`
- **Remaining uncertainty:**
  - Chưa benchmark riêng tokenizer behavior và truncation rate cho medical text
    vi/en/zh.
  - Model card nói hỗ trợ đa ngôn ngữ nhưng không thay thế đánh giá domain y khoa.
- **Decision affected:**
  - Gate 0 phải pin revision, prefix, pooling, normalization và max length.
  - Gate 2 phải audit token length theo ngôn ngữ trước khi encode.
  - E5-small chỉ là smoke baseline, không phải model cuối được mặc định chọn.

## R-004: Nguồn cho xử trí sốc phản vệ

- **Question:** Nguồn chính nào phù hợp để xác minh nội dung sốc phản vệ và có thể
  đưa nguyên văn vào benchmark không?
- **Search terms:** `Resuscitation Council UK emergency treatment anaphylaxis
  guideline latest`, `RCUK permission reproduce materials`.
- **Primary sources checked:**
  - [RCUK — Emergency treatment of anaphylactic reactions](https://www.resus.org.uk/library/additional-guidance/guidance-anaphylaxis/emergency-treatment-anaphylactic-reactions)
  - [RCUK — Application for permission to reproduce materials](https://www.resus.org.uk/application-permission-reproduce-rcuk-materials)
  - [Guideline PDF](https://www.resus.org.uk/media/337/download)
- **Version/date accessed:** Guideline cập nhật May 2021, ghi review date May
  2026; truy cập 2026-09-23.
- **License/reuse status:** `PERMISSION-REQUIRED` cho việc tái sản xuất guideline.
  PDF ghi không được sao chép nếu chưa có written permission; RCUK cung cấp quy
  trình xin phép riêng.
- **Findings:**
  - Đây là nguồn chuyên môn chính thức cho healthcare providers và có lịch sử
    phát triển/public consultation rõ.
  - Nội dung phù hợp làm nguồn kiểm tra fact, nhưng không được copy vào corpus chỉ
    vì PDF tải công khai.
- **Remaining uncertainty:**
  - Review date May 2026 đã qua; cần xác minh có guideline anaphylaxis mới trong
    bộ RCUK 2025/2026 hay không trước khi viết benchmark.
  - Chưa có written permission cho repository/AI benchmark.
- **Decision affected:**
  - Đánh dấu RCUK 2021 là `reference-only`.
  - Không ingest nguyên văn, algorithm, bảng hoặc hình.
  - Chỉ tạo paraphrase sau khi medical review và legal/source policy cho phép.

## R-005: Nguồn NICE cho sỏi thận và đái tháo đường type 2

- **Question:** NICE có guideline phù hợp không và việc dùng nội dung tại Việt Nam
  hoặc cho AI benchmark có được phép không?
- **Search terms:** `NICE NG118 renal ureteric stones recommendations`, `NICE
  NG28 type 2 diabetes initial medicines`, `NICE international AI content reuse
  licence`.
- **Primary sources checked:**
  - [NICE NG118 — Renal and ureteric stones](https://www.nice.org.uk/guidance/ng118/chapter/recommendations)
  - [NICE NG28 — Type 2 diabetes initial medicines](https://www.nice.org.uk/guidance/NG28/chapter/initial-medicines)
  - [NICE terms and conditions](https://www.nice.org.uk/terms-and-conditions)
  - [NICE UK Open Content Licence](https://www.nice.org.uk/reusing-our-content/nice-uk-open-content-licence)
  - [NICE content available for reuse](https://www.nice.org.uk/reusing-our-content/what-content-is-available-for-reuse-)
- **Version/date accessed:**
  - NG118 published 2019; page accessed 2026-09-23.
  - NG28 page accessed 2026-09-23; public page reports update in 2026.
  - Terms/licensing pages accessed 2026-09-23.
- **License/reuse status:** `PERMISSION-REQUIRED` cho intended use hiện tại.
  NICE nêu UK Open Content Licence áp dụng tại UK; use outside UK và AI use cần
  approval/licensing, có thể kèm fee. Third-party content không tự động được NICE
  cấp quyền.
- **Findings:**
  - NG118 và NG28 là nguồn có provenance tốt để xác minh medical facts.
  - Việc trang web truy cập tự do không đồng nghĩa được phép ingest text vào một
    AI corpus quốc tế.
  - Published recommendations còn có hạn chế về adaptation/wording.
- **Remaining uncertainty:**
  - Chưa gửi request cho NICE và chưa có approval cho AI/international reuse.
  - Chưa xác minh từng component có third-party copyright hay không.
- **Decision affected:**
  - Dùng NICE làm `reference-only`.
  - Không copy/translate recommendations vào repository.
  - Ưu tiên tìm nguồn open-license phù hợp hoặc dùng factual paraphrase được
    reviewer duyệt và legal owner chấp thuận.

## R-006: Nguồn GINA cho hen

- **Question:** GINA có phải nguồn hiện hành cho hen và có thể dùng document của
  họ làm corpus không?
- **Search terms:** `GINA 2026 strategy report asthma`, `GINA copyright requests
  distribute adapt documents`.
- **Primary sources checked:**
  - [GINA Reports](https://ginasthma.org/reports/)
- **Version/date accessed:** Trang liệt kê 2026 Strategy Report, Summary Guide và
  Severe Asthma Guide; truy cập 2026-09-23.
- **License/reuse status:** `PERMISSION-REQUIRED`. GINA ghi tài liệu được bảo hộ
  copyright, không được copy để phân phối hoặc đăng lại nếu chưa được cho phép;
  adaptation/reproduction có quy trình liên hệ riêng.
- **Findings:**
  - GINA 2026 là nguồn hiện hành phù hợp để kiểm tra nội dung và version.
  - Không được dùng sự kiện “PDF tải miễn phí” làm căn cứ cho phép tái phân phối.
- **Remaining uncertainty:**
  - Chưa có copyright permission.
  - Chưa xác định một nguồn open-license thay thế đủ authoritative cho corpus.
- **Decision affected:**
  - GINA là `reference-only`.
  - Không commit bản sao, bản dịch hoặc đoạn dài từ report vào benchmark.

## R-007: Nguồn WHO cho tiền sản giật/sản giật

- **Question:** Khuyến cáo WHO nào phù hợp và license có cho phép dùng làm corpus
  đa ngôn ngữ không?
- **Search terms:** `WHO recommendations prevention treatment pre-eclampsia
  eclampsia`, `WHO open access policy publications before 2017 reuse`.
- **Primary sources checked:**
  - [WHO — Recommendations for prevention and treatment of pre-eclampsia and eclampsia](https://www.who.int/publications/i/item/9789241548335)
  - [WHO Open Access policy](https://www.who.int/about/policies/publishing/open-access)
  - [WHO permissions](https://www.who.int/about/policies/publishing/permissions)
  - [WHO website terms of use](https://www.who.int/about/policies/terms-of-use)
- **Version/date accessed:** Publication 2011 với các related updates được liệt
  kê trên trang; truy cập 2026-09-23.
- **License/reuse status:** `PERMISSION-REQUIRED` hoặc tối thiểu
  `REFERENCE-ONLY` cho publication 2011. WHO cho biết publications từ
  12-11-2016 thường dùng CC BY-NC-SA 3.0 IGO, nhưng publication trước 2017 không
  tự động được tái phát hành theo licence đó.
- **Findings:**
  - Publication là nguồn chính thống nhưng cũ và có nhiều update liên quan.
  - Cần đánh giá recommendation hiện hành theo cả publication gốc và updates,
    không chỉ tóm tắt trang 2011.
  - Tái sử dụng phi thương mại, translation và substantial reproduction phụ
    thuộc license cụ thể hoặc permission.
- **Remaining uncertainty:**
  - License cụ thể của file cần dùng chưa được clearance.
  - Chưa lập version map giữa khuyến cáo 2011 và các update sau đó.
- **Decision affected:**
  - WHO 2011 là `reference-only` cho đến khi xác minh license/version.
  - Source registry phải ghi publication date, update chain và reuse status.

## R-008: XQuAD regression fixture

- **Question:** XQuAD hiện tại có thể giữ lại hợp pháp và nên giữ vai trò nào?
- **Search terms:** `google xquad dataset license`, `XQuAD Hugging Face license`.
- **Primary sources checked:**
  - [Hugging Face dataset `google/xquad`](https://huggingface.co/datasets/google/xquad)
  - Local converter, manifest và tests trong repository.
- **Version/date accessed:** Hugging Face dataset revision quan sát được
  `57b8ef9cad99ecf93e5398e5d27c50de2d5814b0`; truy cập 2026-09-23.
- **License/reuse status:** `CONDITIONAL`, CC BY-SA 4.0 theo dataset card. Cần
  attribution và tuân thủ ShareAlike khi phân phối adapted dataset.
- **Findings:**
  - XQuAD là extractive QA đa ngôn ngữ, không phải medical benchmark.
  - Parallel contexts phù hợp để regression-test cross-lingual pipeline.
  - Ground truth tạo từ alignment không giải quyết incomplete relevance judgments.
- **Remaining uncertainty:**
  - Repository chưa có file attribution/license notice riêng cho adapted XQuAD
    artifacts.
  - Cần legal owner xác nhận cách ShareAlike áp dụng cho converted JSONL và repo.
- **Decision affected:**
  - Đóng băng XQuAD làm engineering regression fixture.
  - Không dùng điểm XQuAD để tuyên bố chất lượng medical retrieval.
  - Thêm attribution/licence notice trước khi phân phối artifacts.

---

## Research decisions as of 2026-09-23

| ID | Decision | Status | Evidence |
|---|---|---|---|
| RD-001 | Query contract là tiếng Việt, corpus `vi/en/zh` | Accepted for current design | R-001 |
| RD-002 | Output là document/chunk IDs, không sinh answer | Accepted at product level | R-001 |
| RD-003 | Exact submission/evaluator contract chưa được khóa | Open | R-002 |
| RD-004 | E5-small dùng prefix, mean pooling, normalize và max 512 | Accepted for smoke baseline | R-003 |
| RD-005 | Pin E5 revision `614241f...` cho lần tái lập tiếp theo | Proposed; config chưa cập nhật | R-003 |
| RD-006 | RCUK/NICE/GINA/WHO hiện chỉ dùng reference | Accepted until clearance | R-004–R-007 |
| RD-007 | Không ingest guideline nguyên văn khi license chưa rõ | Accepted | R-004–R-007 |
| RD-008 | XQuAD chỉ là regression fixture | Accepted | R-008 |

## Remaining research backlog

| Priority | Question | Required evidence | Owner | Status |
|---:|---|---|---|---|
| P0 | Exact schema và evaluator chính thức là gì? | BTC technical spec + official fixture | Project owner | Open |
| P0 | Có constraint runtime/hardware/internet nào? | BTC technical spec | Project owner | Open |
| P0 | Ai duyệt correctness y khoa? | Named clinical reviewer/adjudicator | Project owner | Open |
| P0 | Nguồn nào được phép dùng làm distributed AI corpus? | Written licence/permission | Legal/source owner | Open |
| P1 | RCUK đã có anaphylaxis update thay bản 2021 chưa? | Current RCUK publication/version | Research owner | Open |
| P1 | WHO 2011 được update bởi tài liệu nào cho từng recommendation? | WHO update map | Medical reviewer | Open |
| P1 | Open-license source thay NICE/GINA là gì? | Primary source + explicit licence | Research owner | Open |
| P1 | E5 truncation/tokenization khác nhau thế nào trên vi/en/zh medical text? | Token audit fixture | Engineering owner | Open |
| P2 | XQuAD attribution/share-alike notice cần hình thức nào? | Licence review | Legal/source owner | Open |

## Step 2 completion status

**Not DONE.**

Đã có nguồn chính cho product contract, E5 preprocessing và năm nhóm nguồn y khoa,
nhưng hai blocker vẫn có thể làm đổi artifact:

1. Chưa có technical specification/fixture chính thức để khóa input-output và
   evaluator contract.
2. Chưa có medical source nào được clearance đầy đủ cho việc tạo và phân phối AI
   benchmark đa ngôn ngữ trong repository.

Không tạo corpus nguyên văn hoặc tuyên bố benchmark medically validated trước khi
hai blocker này được giải quyết hoặc có phương án thay thế được project owner,
clinical reviewer và legal/source owner duyệt.

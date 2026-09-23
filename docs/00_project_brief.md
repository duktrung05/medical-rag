# Project brief

**Status:** Draft — cần project owner duyệt và đối chiếu đề kỹ thuật chính thức  
**Last updated:** 2026-09-23  
**Primary public reference:** [Road to AI 2026](https://r2ai.aiguru.com.vn/)

## Problem

Xây dựng hệ thống **multilingual medical retrieval** nhận một truy vấn y khoa
bằng tiếng Việt và truy hồi các document cùng chunk liên quan nhất từ corpus đa
ngôn ngữ Việt (`vi`), Anh (`en`) và Trung (`zh`).

Đây là bài toán retrieval-only:

```text
Vietnamese medical query
    -> search the VI/EN/ZH corpus
    -> rank relevant chunks
    -> aggregate parent documents
    -> select document and chunk IDs
    -> validate submission
```

Hệ thống không cần sinh câu trả lời tự nhiên. Một kết quả có thể thuộc bất kỳ
ngôn ngữ nào trong corpus; relevance với query quan trọng hơn việc khớp ngôn ngữ.

Query được xem là một biểu đạt của **clinical information need**. Dạng phổ biến là
câu hỏi tự nhiên, nhưng implementation không nên phụ thuộc vào việc query luôn có
dấu hỏi hoặc luôn là câu hoàn chỉnh. Query ngắn dạng keyword chỉ được coi là input
hợp lệ nếu đề kỹ thuật không cấm; chất lượng trên dạng này phải được đánh giá
riêng.

## Intended user

### Primary user

- Đội phát triển tham gia Road to AI 2026.
- Hệ thống chấm/evaluator của cuộc thi.

### Secondary user

- Kỹ sư hoặc nhà nghiên cứu cần kiểm tra và tái lập pipeline retrieval y khoa đa
  ngôn ngữ.

### Not an intended user

- Bệnh nhân hoặc nhân viên y tế sử dụng output như chẩn đoán hay hướng dẫn điều
  trị trực tiếp.

Nếu sau này phát triển sản phẩm cho người dùng thật, cần một product contract,
safety review và clinical validation riêng; không suy rộng trực tiếp từ kết quả
cuộc thi.

## Input

### Query

Contract nội bộ hiện tại:

```json
{"id":"Q001","query":"Cần làm gì khi tắc nghẽn đường tiết niệu do sỏi thận?"}
```

- `id`: định danh duy nhất của query.
- `query`: chuỗi truy vấn y khoa bằng tiếng Việt.
- Query có thể mô tả triệu chứng, chẩn đoán, điều trị, chống chỉ định, theo dõi,
  dự phòng hoặc một clinical intent khác.

### Corpus

Contract chunk nội bộ hiện tại:

```json
{
  "chunk_id": "DOC_EN_0042_003",
  "doc_id": "DOC_EN_0042",
  "chunk_index": 3,
  "language": "en",
  "text": "..."
}
```

Corpus chứa document/chunk thuộc `vi`, `en` hoặc `zh`. ID chính thức của BTC phải
được giữ nguyên khi adapter chuyển dữ liệu sang schema nội bộ.

Đối với benchmark nội bộ, corpus được version hóa và giữ cố định trong một
experiment. Chưa có đủ bằng chứng công khai để kết luận corpus chính thức của BTC
là bất biến trong toàn bộ cuộc thi hay có thể được cập nhật giữa các vòng.

## Output

Output retrieval là các ID, không phải câu trả lời đã dịch hoặc văn bản được sinh:

```json
{
  "id": "Q001",
  "relevant_docs": ["DOC_EN_0042", "DOC_ZH_0008"],
  "relevant_chunks": ["DOC_EN_0042_003", "DOC_ZH_0008_001"]
}
```

- `relevant_docs`: danh sách document ID được dự đoán là relevant.
- `relevant_chunks`: danh sách chunk ID được dự đoán là relevant.
- ID có thể trỏ tới nội dung tiếng Việt, Anh, Trung hoặc kết hợp nhiều ngôn ngữ.
- Mỗi predicted chunk phải có parent document tương ứng trong
  `relevant_docs` theo contract nội bộ hiện tại.

Exact serialization, quy tắc ordering, giới hạn số ID và việc danh sách rỗng có
hợp lệ hay không phải được xác nhận bằng đề kỹ thuật chính thức.

## Languages

| Thành phần | Contract hiện tại | Trạng thái |
|---|---|---|
| Query | Tiếng Việt (`vi`) | Được trang cuộc thi công khai xác nhận |
| Corpus | Việt, Anh, Trung (`vi`, `en`, `zh`) | Được trang cuộc thi công khai xác nhận |
| Retrieval output | ID của item thuộc bất kỳ ngôn ngữ corpus nào | Suy ra từ bài toán cross-lingual retrieval |
| Generated answer | Không có | Trang cuộc thi mô tả retrieval-only |

Hệ thống tìm kiếm trên một không gian xếp hạng chung:

```text
query vi -> candidates vi + en + zh -> ranked document/chunk IDs
```

Không ép mỗi query phải trả đủ ba ngôn ngữ và không chọn một ngôn ngữ trước khi
retrieval, trừ khi một experiment đã chứng minh routing theo ngôn ngữ tốt hơn.

## Unit of retrieval

Hệ thống phải retrieve và đánh giá ở cả hai cấp:

1. **Chunk-level:** đơn vị candidate trực tiếp của retriever.
2. **Document-level:** document cha được tổng hợp từ chunk results.

Baseline nội bộ:

```text
doc_score(document) = max(score(chunk) for chunk in document)
```

Một document chỉ xuất hiện một lần trong output. Cách aggregation này là baseline
có thể thay đổi qua experiment, không được coi là quy tắc chính thức của BTC cho
đến khi đề kỹ thuật xác nhận.

## Success criteria

### Competition quality

- Prediction hợp lệ theo schema chính thức.
- Tối ưu Precision, Recall và Macro F2 ở document-level và chunk-level theo
  evaluator chính thức.
- Truy hồi được relevant English/Chinese content cho Vietnamese query khi ground
  truth yêu cầu.
- Không tăng recall bằng cách trả tràn lan đến mức làm precision và F2 giảm.

F2 đặt trọng số recall cao hơn precision, nhưng precision vẫn là một phần của
điểm. Mọi quy tắc chọn `K`, threshold hoặc reranking phải được chọn trên dev và
khóa trước khi chạy held-out test.

### Scientific quality

- Benchmark có positive, hard negative và quy tắc xử lý unjudged candidate.
- Mỗi experiment thay đổi một biến chính và có decision rule ghi trước.
- Không dùng test để chọn model, `K`, threshold hoặc prompt/query expansion.
- Kết luận phải dựa trên evidence, không chỉ một metric tổng hợp.

### Engineering quality

- Input/config/code revision giống nhau tạo ranking và metric tương đương.
- Validator phát hiện duplicate ID, unknown ID, query thiếu/thừa và parent
  violation trước khi đánh giá.
- Index, config, manifest, prediction và metric đều có provenance/hash cần thiết.
- Có một command hoặc checklist tái tạo pipeline từ dữ liệu đến report.

### Safety quality

- Nội dung y khoa có source, version/date, license/reuse status và review status.
- Artifact chưa được clinical reviewer duyệt phải được gọi là
  `software/synthetic smoke benchmark`, không gọi là medically validated.
- Retrieval output không được trình bày như chẩn đoán hoặc chỉ định điều trị.

## Non-goals

Trong phạm vi baseline/smoke project hiện tại, không nhằm:

- Sinh câu trả lời y khoa bằng LLM.
- Dịch output sang một ngôn ngữ hiển thị cố định.
- Chẩn đoán, kê đơn hoặc thay thế quyết định của nhân viên y tế.
- Chứng minh model an toàn để dùng trong lâm sàng.
- Fine-tune model trước khi có đủ nhãn và error analysis.
- Tối ưu ANN/vector database cho hàng chục triệu chunk trong smoke phase.
- Kết luận model tốt nhất chỉ từ benchmark nhỏ hoặc synthetic.
- Dùng điểm XQuAD làm bằng chứng chất lượng retrieval y khoa.
- Sao chép hoặc phát hành lại guideline nếu quyền sử dụng chưa rõ.

## Medical and legal risks

| Risk | Consequence | Required control |
|---|---|---|
| Nội dung y khoa sai hoặc lỗi thời | Retrieve thông tin không an toàn | Nguồn có phiên bản; clinical review; review status |
| Bản dịch làm đổi ý lâm sàng | Nhãn cross-lingual sai | Reviewer hiểu ngôn ngữ/domain; adjudication |
| Guideline bị sao chép trái phép | Rủi ro bản quyền/phân phối | Source registry và license check trước khi commit text |
| Ground truth thiếu relevant item | Hệ thống đúng ngữ nghĩa vẫn bị phạt | Candidate pooling; relevance review; giữ trạng thái unjudged |
| Hard negative sai nhãn | Metric và error analysis sai | Annotation guideline; reviewer thứ hai cho ca tranh chấp |
| Dữ liệu chứa thông tin nhận dạng | Rủi ro riêng tư | Chỉ dùng dữ liệu công khai/được phép; không ingest PHI/PII |
| Output bị hiểu như medical advice | Nguy cơ gây hại | Retrieval-only labeling; disclaimer; không đưa vào clinical use |
| Test leakage | Điểm dev/test không đáng tin | Split theo intent/source group; khóa qrels test |
| Thể lệ thay đổi giữa các vòng | Pipeline hoặc submission không hợp lệ | Version official spec; adapter; re-run contract audit |

Người có quyền quyết định một nội dung y khoa là đúng phải là clinical reviewer
được project owner chỉ định. Kỹ sư hoặc model không tự đóng vai trò medical
adjudicator. Hiện repository chưa ghi nhận danh tính/role đã được phê duyệt cho vị
trí này.

## Known facts

Những mục dưới đây đã được xác nhận từ trang công khai của cuộc thi, truy cập ngày
2026-09-23:

1. Bài toán là medical document và chunk retrieval đa ngôn ngữ.
2. Query đầu vào là tiếng Việt.
3. Corpus gồm tiếng Việt, tiếng Anh và tiếng Trung.
4. Hệ thống cần retrieve cả document và chunk.
5. Bài toán có cross-lingual matching.
6. Trang cuộc thi nêu Precision, Recall và Macro F2, với F2 ưu tiên recall.
7. Output minh họa dùng `RELEVANT_DOCS` và `RELEVANT_CHUNKS`.
8. Bài toán được mô tả là retrieval-only, không sinh văn bản.

Known facts từ repository hiện tại:

1. Schema nội bộ đã có `QueryRecord`, `ChunkRecord`, `PredictionRecord` và
   `GroundTruthRecord`.
2. Validator nội bộ enforce document-chunk parent consistency.
3. Dense smoke baseline hiện dùng `intfloat/multilingual-e5-small` với exact
   search.
4. XQuAD chỉ phù hợp làm engineering regression fixture, không phải benchmark
   chất lượng medical.
5. Hướng benchmark chính đã chuyển sang `medical_smoke_v1` với source provenance,
   nhiều chunk/document và relevance review.

## Assumptions

Các mục này là contract nội bộ tạm thời, chưa được coi là thể lệ chính thức:

1. File submission là JSONL, mỗi query một record.
2. Field output dùng tên `id`, `relevant_docs`, `relevant_chunks` viết thường.
3. Thứ tự trong danh sách thể hiện ranking.
4. Mọi predicted chunk phải có parent document trong `relevant_docs`.
5. Ground truth có thể chứa nhiều relevant document/chunk trên một query.
6. Document score baseline là max chunk score.
7. Corpus được chunk sẵn bởi BTC và ID phải được giữ nguyên.
8. Query chính thức là natural-language medical question; keyword query có thể
   xuất hiện nhưng chưa được xác nhận.
9. Corpus có thể chứa ba ngôn ngữ độc lập, không nhất thiết là bản dịch song song.
10. Không có latency SLA bắt buộc trong vòng đánh giá offline.
11. Smoke benchmark nội bộ chạy được trên CPU; pipeline chính có thể cần GPU khi
    corpus chính thức lớn.

Mỗi assumption có khả năng đổi schema, metric hoặc kiến trúc phải được xác nhận
trước khi khóa submission pipeline.

## Open questions

### Official task contract

1. Đề kỹ thuật/version chính thức nằm ở đâu và ngày hiệu lực là ngày nào?
2. Exact input schema, output schema và encoding là gì?
3. Output list có cần giữ thứ tự ranking hay được chấm như set?
4. Số lượng document/chunk tối đa hoặc tối thiểu trên mỗi query là bao nhiêu?
5. Empty prediction có hợp lệ không?
6. Unknown ID, duplicate ID, query thiếu/thừa được evaluator xử lý thế nào?
7. Parent consistency là quy tắc chính thức hay chỉ là contract nội bộ?

### Corpus and languages

8. Corpus cố định trong toàn cuộc thi hay được cập nhật giữa các vòng?
9. Document/chunk do BTC cung cấp đã chunk hoàn chỉnh chưa?
10. Ba ngôn ngữ là corpus độc lập, bản dịch song song hay hỗn hợp cả hai?
11. Mỗi record có language metadata chính thức không?
12. Có document trộn nhiều ngôn ngữ hoặc language ngoài `vi/en/zh` không?

### Labels and metrics

13. Relevance là binary hay graded?
14. Candidate không có nhãn bị tính negative hay bị bỏ qua?
15. Macro F2 được tính per-query rồi average hay từ macro Precision/Recall?
16. Document F2 và chunk F2 được kết hợp thành final score bằng công thức nào?
17. Beta, tie handling và query không có positive được xử lý ra sao?
18. Có public dev labels và hidden test labels không?

### Runtime and submission

19. Chấm offline bằng file hay chạy service/API?
20. Có latency, throughput, RAM, VRAM, image size hoặc internet constraint không?
21. Có được dùng external model, external corpus, translation API hoặc LLM không?
22. Có yêu cầu license/open-source khi nộp model và code không?

### Medical governance

23. Ai là project owner có quyền duyệt benchmark contract?
24. Ai là clinical reviewer và adjudicator?
25. Review status nào là bắt buộc trước khi một document/query/qrel được đưa vào
    dev hoặc test?
26. Quy trình xử lý khi nguồn y khoa được cập nhật hoặc hai guideline mâu thuẫn là
    gì?

## Approval

Project brief chỉ chuyển sang `accepted` khi:

- Project owner duyệt phạm vi và non-goals.
- Đề kỹ thuật chính thức giải quyết các open question có thể đổi kiến trúc.
- Clinical reviewer/adjudicator được chỉ định.
- Medical/legal source policy được duyệt.
- Input/output được kiểm tra bằng ít nhất một fixture chính thức của BTC.

# Kế hoạch cải thiện score ViBioMIR từ config tốt nhất hiện có

Ngày nghiên cứu và audit: 08/10/2026, múi giờ Asia/Ho_Chi_Minh.

## 1. Quyết định đề xuất

Ưu tiên **chọn đúng evidence trong pool hiện có → mở rộng passage pool có giữ đa dạng document → sửa extraction/chunking → cải thiện truy hồi song ngữ và fusion → thử reranker mới → fine-tune khi có nhãn**.

Dự án đã có BM25, dense, RRF và cross-encoder. Hướng cải thiện có cơ sở nhất hiện nay là sửa độ phủ và lựa chọn passage. Thêm một model lớn chưa giải quyết được tài liệu hoặc đoạn đúng đã bị loại trước reranker.

Đây là kế hoạch triển khai và thí nghiệm. Các mức tác động bên dưới là giả thuyết kỹ thuật có cách kiểm chứng, chưa phải mức tăng leaderboard đã đo.

## 2. Mốc đối chứng và bằng chứng local

Config xuất phát là `configs/vibiomir/evidence_selector_best_20261008.json`:

```json
{
  "mode": "threshold",
  "max_docs": 200,
  "max_chunks": 60,
  "chunks_per_doc": 1,
  "chunk_threshold": -2.5,
  "chunk_delta": null,
  "document_mode": "threshold",
  "doc_threshold": -2.5,
  "doc_delta": null,
  "chunk_order": "document"
}
```

Tên config thể hiện lựa chọn hiện tại của workspace. Chưa tìm thấy feedback leaderboard gắn với bản -2.5; cần nối score, ZIP SHA-256 và phase tương ứng trước khi gọi nó là bản có score cao nhất đã xác minh.

Mốc có đầy đủ bảy metric trong tài liệu là bản threshold -2, ghi tại `docs/vibiomir_chunk_precision_recall_plan_20261008_vi.md`. Các metric này là kết quả đã được ghi từ người dùng, chưa được tính lại bằng scorer local:

| Metric | Mốc threshold -2 |
|---|---:|
| FINAL_SCORE | 0.1681 |
| DOCS_F2MACRO | 0.2802 |
| CHUNKS_F2MACRO | 0.0560 |
| DOCS_PRECISION / DOCS_RECALL | 0.2665 / 0.3190 |
| CHUNKS_PRECISION / CHUNKS_RECALL | 0.0675 / 0.0618 |

Không dùng internal_macro_f2=0.81697 trên ViMedQA làm điểm ViBioMIR: khác dataset, gold và luật đánh giá.

Audit trực tiếp trong lượt lập kế hoạch:

| Phát hiện | Bằng chứng | Hàm ý cần kiểm chứng |
|---|---|---|
| Dense stage 1 chỉ biểu diễn phần đầu document | `r2ai/index.py`; script build dùng lead_chunks=2, seq=256 | Evidence ở giữa/cuối có thể không giúp document vào shortlist |
| Fusion giữ 400 document/query; chunk shortlist giữ 200 chunk/query | Script dense/fuse và metadata ranking cache | Có hai điểm cắt khác nhau; tăng pool chunk không phục hồi doc ngoài fusion |
| Reranker chỉ thấy trung bình 103.23 document/query | Đọc đủ 1.200 dòng cache; median 101, min 38, max 171 | Một số doc trong 400 ứng viên không còn passage được xét; đây là coverage, chưa phải gold recall |
| Cache có 240.000 cặp, toàn bộ có score thật | Pool=200, rerank_top=200; 0 score null | CPU replay tái dùng được khi chỉ đổi selector |
| Bản -2.5 chạm cap60 ở 876/1.200 query, tức 73% | Summary submission | Cap có ảnh hưởng tới tập đầu ra; chưa biết chunk ngoài cap đúng hay sai |
| Bản -2.5 xuất trung bình 80.18 docs, 55.24 chunks | Summary submission | max_docs=200 là giới hạn trên, không phải số doc thực tế |
| Bản -2.5 loại 86.161 candidates bởi cap1/doc và 30.928 bởi cap60 | Tổng reasons trong diagnostics | Có cơ hội thử nhiều passage/doc và cap tổng; số bị loại không phải số positives bị mất |
| Selector khử trùng text xuyên document | `seen_text` trong `r2ai/evidence_selector.py` | Cần kiểm scorer có phân biệt parent doc khi chấm evidence |
| Hai ZIP 2 chunks/doc đã tồn tại | Thư mục `vibiomir_neg2_chunks2_*_20261008` | Chấm/đối chiếu bản hiện có trước khi tạo lại |
| Chỉ đổi document-order thành score-order ở hai ZIP này đổi tập chunks của 969 query | So sánh trực tiếp ZIP; tập docs của cả 1.200 query giữ nguyên | Ordering có ảnh hưởng selection; chưa có bằng chứng hướng nào tăng score |

Cache đối chứng: `outputs/evidence/retrieval_crawl_rerank80_20261008/rankings.jsonl.gz`. Dù tên có “80”, metadata xác nhận pool200 và rerank_top200; không suy tham số từ tên thư mục.

## 3. Các năng lực cần có để nâng chất lượng retrieval

| Năng lực | Hiện trạng | Bổ sung đề xuất | Cải thiện trực tiếp |
|---|---|---|---|
| Evaluation đúng mục tiêu | Có validator, diagnostics và evaluator nội bộ; thiếu scorer/qrels chính thức local | Kiểm luật matching, gold/pool labels, dev/holdout và log kết quả theo SHA | Phân biệt tăng relevance thật với thay đổi số lượng; tránh chọn sai config |
| Extraction và passage đáng tin | Có trafilatura, fallback và chunk theo ước lượng token | Audit theo domain/ngôn ngữ; chunk bằng tokenizer thật và đoạn/câu | CHUNKS_PRECISION, CHUNKS_RECALL |
| Candidate recall qua từng tầng | Document hybrid rồi dense-only chunk shortlist | Chunk hybrid, nhiều passage/document, multi-window document | DOCS_RECALL, CHUNKS_RECALL |
| Query hiểu alias/ngôn ngữ | Query Việt; corpus gồm Việt và Trung | Alias y khoa có kiểm chứng, truy vấn Trung bổ sung, giữ truy vấn gốc | Recall tài liệu Trung và thuật ngữ khác cách viết |
| Fusion được hiệu chỉnh | RRF 1:1, k60 | Weighted RRF; convex score fusion khi có nhãn | Chất lượng pool và cân bằng lexical/dense |
| Reranking đúng ý hỏi | BGE-reranker-v2-m3, max_length512 | So model trên cùng pool; hard negatives và fine-tune nếu cần | Precision; recall sau gate khi positives được nâng score |
| Selection phù hợp F2 | Threshold/caps/cache đã có | Tách doc/chunk gates, thử cap/doc, score-order và dedup scope | F2 cuối cùng, số positives được giữ |
| Tái lập và quản lý chi phí | Cache và fingerprint đã có ở evidence workflow | Ghi revision stage1, pair cache, throughput/VRAM và thời gian mỗi tầng | Tốc độ thí nghiệm, độ tin cậy so sánh; không tự tăng relevance |

Các kỹ thuật mới phải được xem là thí nghiệm cho ViBioMIR. Nghiên cứu tiếng Việt năm 2026 đánh giá nhiều họ retrieval trên nhiều miền và chỉ ra kích thước model có giá trị dự báo hạn chế đối với hiệu quả retrieval. [Nguồn: nghiên cứu ViRE, EACL 2026](https://aclanthology.org/2026.findings-eacl.110/).

## 4. Kế hoạch theo thứ tự thực hiện

### P0 — Chốt baseline và đo lỗi: khoảng 0,5–1 ngày công, cộng thời gian gán nhãn

**Trạng thái 08/10/2026:** phần tự động đã hoàn tất tại
`outputs/evidence/task0_baseline_audit_20261008/`: baseline manifest, bảng metric,
error breakdown và 100 dev + 50 holdout không trùng nhau. Hai ZIP -2/-2.5 đều
qua validator; 158 test workflow ViBioMIR pass. 30.000 trường `relevant` vẫn
để `null` chờ người gán nhãn, nên chưa chạy calibration hoặc kết luận recall
trong pool. Trang công khai chỉ xác nhận document/chunk retrieval và Macro F2
ưu tiên recall; luật chunk matching chi tiết vẫn chưa xác minh được.

1. Gắn config -2.5 với ZIP/hash/feedback đúng. Giữ bản -2 score0.1681 làm đối chứng đã ghi nhận. Khi nhận score mới hơn, cập nhật mốc số liệu; giữ thiết kế so sánh.
2. Kiểm tài liệu/scorer chính thức: normalization, tokenizer, matching theo doc_id, ghép một-một/nhiều-một, macro, giới hạn text và submission. Truy cập web hiện chưa xác minh được luật LCS/40% được nhắc trong comment code.
3. Audit ban đầu 100–200 query phân tầng theo dạng câu hỏi, tiếng của evidence, domain, score và số chunks. Với các query được chọn, mở rộng pool từ nhiều retriever và xem nội dung các doc đúng, không chỉ gán nhãn top few.
4. Tạo dev/holdout tách query; kiểm nguồn trùng. Nhãn annotation trong pool chỉ đo recall trong pool. Đo recall toàn corpus cần gold có phạm vi đầy đủ.
5. Phân lỗi: doc đúng ngoài fusion; doc đúng trong fusion nhưng passage ngoài pool; passage đúng bị reranker/gate loại; extraction mất nội dung; passage thiếu ngữ cảnh; matching không đạt.

**Bàn giao:** baseline manifest, bảng metrics, tập annotation và error breakdown. Phân lỗi giúp chọn P2/P3/P4 đúng chỗ; không cần chờ annotation hoàn tất để so và chấm các ZIP CPU đã tồn tại.

### P1 — Tối ưu evidence selector: khoảng 1 ngày công, CPU

Lấy B0=-2.5 theo config xuất phát. Document gate ghi explicit=-2.5 để thử chunk độc lập. Trong cùng cache, mỗi bước chỉ đổi một yếu tố:

| Run | Thay đổi so với đối chứng chỉ định | Mục đích |
|---|---|---|
| B0 | Config hiện tại | Mốc cùng corpus/model/cache |
| S1 so B0 | chunks_per_doc: 1 → 2; cap60 giữ nguyên | Giữ thêm evidence ở document có nhiều ý |
| S2 so S1 | chunk_order: document → rerank_score | Chunk yếu của doc đầu không chiếm slot chunk mạnh của doc sau |
| S3 so bản thắng | max_chunks: 60 → 80 | Đo true positives bị giới hạn bởi cap |
| S4 so bản thắng | chunk_threshold lần lượt -3, -2, -1.5; doc_threshold giữ -2.5 | Tìm đánh đổi precision/recall của chunk |
| S5 so bản thắng | Chỉ đổi doc_threshold trên dev; chunk gate giữ nguyên | Tối ưu DOCS_F2 riêng |
| S6 sau kiểm scorer | Dedup exact text trong cùng doc thay vì xuyên doc | Tránh mất evidence khác parent ID |

Không cần chạy tất cả biến thể nếu nhãn hoặc các kết quả trước cho thấy không có lợi. `chunk_delta=6` là một probe độc lập, có thể đối chiếu ZIP -2.5+delta6 đã tồn tại với ZIP -2.5; documents của hai cấu hình này giữ cùng gate.

Hai ZIP -2/chunks2/document-order và -2/chunks2/score-order đang có dùng để thử tác động trên mốc -2, không gọi chúng là phép so một yếu tố với B0=-2.5.

**Có thể cải thiện:** chunk recall nhờ nhiều passages; chunk precision nhờ chọn thứ tự tốt hơn; doc precision nhờ gate riêng. Threshold chặt hơn có thể làm recall giảm. Chỉ tăng số outputs chưa chứng minh tăng F2.

**Bàn giao:** các ZIP thắng ứng viên, config, diagnostics và bảng bảy metric. Chỉ chấm biến thể có tập đầu ra khác, trừ khi scorer chính thức phụ thuộc thứ tự.

### P2 — Mở rộng passage pool và giữ đa dạng document: khoảng 1–2 ngày công, một GPU

1. Đối chứng pool200 → pool400, rerank_top400; giữ fused400 documents, chunks, encoder và selector của đối chứng. Tổng cặp full run tăng từ 240.000 lên tối đa 480.000.
2. Thử riêng chiến lược budget passage theo document: bảo đảm passage cho doc được ưu tiên bằng stage1, sau đó thêm passage thứ 2–4 theo dense và/hoặc BM25. So với global top400 ở **cùng budget cặp** để tách tác động đa dạng khỏi tăng compute.
3. So riêng chunk shortlist dense với union dense + BM25 trong cùng fused docs. Bộ lọc dense-only hiện có thể bỏ passage có từ khóa quan trọng; ưu tiên thực thể, xét nghiệm và con số trong audit.
4. Ghi số distinct docs, Recall@K trong phạm vi nhãn, gold passages được đưa thêm và positives bị mất. Phân biệt pool400 chunks với fusion400 documents.
5. Với cùng cache mới, replay CPU để hiệu chỉnh selector; giữ một lượt so selector cố định để thấy tác động riêng của pool.

**Có thể cải thiện:** CHUNKS_RECALL khi đúng passage chưa vào top200; DOCS_RECALL khi doc đã có trong fused400 nhưng trước đây không có passage được chấm. Không xử lý doc ngoài fused400.

**Chi phí:** pilot 50 query chỉ đo throughput và peak VRAM; sample score cần nhãn. Thời gian full run = số cặp cần chấm / pairs mỗi giây + I/O/load. Code hiện chưa có cơ chế nối cache điểm; mặc định chấm lại toàn bộ pool400. Reuse điểm cặp chỉ sau khi triển khai key bao gồm query/text/title/model/revision/truncation settings.

### P3 — Sửa extraction, chunking và ngữ cảnh: khoảng 2–3 ngày công cho pilot

1. Audit trang thành công nhưng extraction thiếu, các đoạn menu/form/hotline lẫn nội dung, đặc biệt trang Q&A Trung. Giữ câu hỏi và trả lời bác sĩ có liên quan; kiểm patch extractor theo từng domain trên mẫu nguồn.
2. Đo token thực tế của query + title + passage. Reranker hiện giới hạn512 cho cả cặp; chunk vượt giới hạn có thể bị cắt mất evidence.
3. Thử chunk đoạn/câu ở các budget 256/384/448 token với overlap khoảng10–20%; budget cuối phải trừ query/title/special tokens theo từng cặp. Đây là grid pilot, không phải độ dài tối ưu đã biết.
4. So passage nguyên bản với window nhỏ quanh passage trong cùng doc, giữ doc và trung tâm. Nếu dùng text mở rộng để quyết định relevance thì chấm lại text đó; score passage trung tâm không phải score của toàn window.
5. Nếu audit thấy mất ngữ cảnh là nguyên nhân chính, thử late chunking trên subset bằng model/pooling tương thích. Kỹ thuật này contextualize token trong văn bản dài trước khi chia và pool thành embeddings. [Nguồn: Late Chunking](https://arxiv.org/abs/2409.04701).

**Có thể cải thiện:** CHUNKS_PRECISION qua giảm boilerplate; CHUNKS_RECALL qua giữ câu trả lời và ngữ cảnh. Có thể hỗ trợ cả doc retrieval khi text biểu diễn tốt hơn. Window dài hơn có thể tăng noise/truncation, phải đo riêng.

**Bàn giao:** mẫu trước/sau extraction, thống kê truncation, chunk corpus pilot và kết quả scorer. Chỉ rebuild toàn corpus sau khi pilot có lợi; đổi text retrieval yêu cầu embedding/rerank mới.

### P4 — Truy hồi song ngữ, biểu diễn tài liệu và fusion: khoảng 2–3 ngày công cho pilot

1. Với lỗi doc đúng nằm ngoài fused400, thử biểu diễn nhiều cửa sổ đầu/giữa/cuối tài liệu, hoặc chunk-level retrieval rồi gom về document. Giữ tổng budget candidate để so fairness; tránh một document chiếm nhiều slots.
2. Query gốc vẫn là nhánh chính. Thêm alias y khoa Việt–Trung đã kiểm tra và một nhánh dịch truy vấn Trung, giữ tên bệnh/thuốc, phủ định, tuổi và con số. Nội dung evidence xuất ra vẫn là text nguồn. LLM expansion/Query2doc chỉ là nhánh thử khi alias chưa đủ; kiểm query drift. Query2doc có nghiên cứu cải thiện sparse/dense, nhưng mức tăng trên benchmark của paper không phải dự báo ViBioMIR. [Nguồn: Query2doc, EMNLP 2023](https://aclanthology.org/2023.emnlp-main.585/).
3. Thử weighted RRF với tỷ lệ dense:BM25 = 1:0.5, 1:1, 1:2; giữ k60. Chỉ sau đó thử k20/k100 trên dev. Khi có nhãn, so convex combination của score đã normalization; không cộng raw BM25 và cosine trực tiếp. Nghiên cứu fusion cho thấy RRF có độ nhạy tham số và convex fusion là lựa chọn đáng thử. [Nguồn: An Analysis of Fusion Functions for Hybrid Retrieval](https://arxiv.org/abs/2210.11934).
4. Dense search hiện bật lang_balance, nên phải có ablation bật/tắt trên cùng index và query set. Không ép tỷ lệ tiếng Trung chỉ vì corpus có nhiều trang Trung; chọn theo relevance.
5. Thử thêm BGE-M3 sparse head và ColBERT-style multi-vector **trên shortlist/pilot**. BGE-M3 có cả dense, sparse, multi-vector và hỗ trợ văn bản dài, nhưng pipeline SentenceTransformer hiện đang lấy dense. Sparse lexical matching không tự giải quyết query Việt với văn bản Trung không có token chung; alias/dịch và dense vẫn cần được đánh giá. [Nguồn: BGE-M3 documentation](https://bge-model.com/bge/bge_m3.html).

**Có thể cải thiện:** DOCS_RECALL từ evidence sâu trong tài liệu, thuật ngữ khác cách viết và corpus Trung; CHUNKS_RECALL hưởng lợi từ pool có thêm doc đúng. Weighted fusion có thể cải thiện precision pool.

**Chi phí:** tăng cửa sổ làm tăng số embeddings; multivector tăng lưu trữ/compute. Các nhánh mới chạy trên pilot trước khi triển khai hàng triệu documents. Mọi embedding/text/model đổi cần cache riêng và provenance.

### P5 — So reranker và học từ hard negatives: 1–2 ngày công cho A/B; fine-tune là giai đoạn riêng

1. Giữ nguyên candidate identities và source text; so BGE hiện tại với Qwen3-Reranker-0.6B. Thử4B khi pilot0.6B chưa đạt và ngân sách cho phép. Qwen công bố các model đa ngôn ngữ, hỗ trợ instruction và context dài. Đây là cơ sở chọn model ứng viên, chưa chứng minh thắng ViBioMIR. [Nguồn: Qwen3 Embedding/Reranking](https://qwenlm.github.io/blog/qwen3-embedding/).
2. Giữ instruction/format cố định trên dev; log raw scores theo đúng semantics. **Hiệu chỉnh lại threshold cho từng model**: -2.5 của BGE không chuyển sang Qwen. So ranking quality và F2 sau calibration riêng trên cùng holdout.
3. Nếu passage đúng đã trong pool nhưng vẫn bị xếp/gate sai, chuẩn bị positives và hard negatives cùng bệnh nhưng khác ý hỏi, đối tượng, giai đoạn, xét nghiệm hoặc trị số. Dùng nhãn tin cậy; tránh biến positive chưa được gán nhãn thành negative.
4. Fine-tune reranker trước nếu nút thắt là precision/ranking. Fine-tune embedding nếu nút thắt là candidate recall. Tách query/nguồn giữa train, dev, holdout; đo trên tập giữ kín. Hard-negative mining là hướng có workflow chính thức trong Sentence Transformers. [Nguồn: cross-encoder training](https://www.sbert.net/examples/cross_encoder/training/ms_marco/README.html).

**Có thể cải thiện:** DOCS/CHUNKS_PRECISION và recall sau threshold khi model phân biệt đúng chủ đề với đúng ý hỏi. Model mới không phục hồi passage ngoài pool. Fine-tune cần tập nhãn riêng lớn hơn mẫu audit; chưa có cơ sở chốt số ngày huấn luyện hoặc mức tăng.

## 5. Khả năng cải thiện và mức điểm cần hiểu thế nào

| Hướng | Chỉ số có cơ hội tăng | Cơ sở ưu tiên | Giới hạn |
|---|---|---|---|
| Selector nhiều chunk/doc, ordering và cap | Chunk recall/F2; precision tùy chọn | Cao về tính đáng thử: nhiều candidates bị caps loại, ZIP đã có | Chưa có gold để biết candidate bị loại là đúng |
| Passage pool có diversity/hybrid | Chunk recall; doc recall sau stage2 | Cao về cơ chế: reranker chỉ thấy ~103/400 docs | Chỉ giúp trong fused pool hiện tại |
| Extraction/chunking theo token | Chunk precision và recall | Trung bình–cao, cần error audit | Đổi corpus/cache và tốn embedding mới |
| Multi-window + song ngữ + fusion | Doc recall, rồi chunk recall | Trung bình; phù hợp corpus Việt–Trung và stage1 head-only | Nhiều compute; query drift và noise có thể giảm precision |
| Reranker mới/fine-tune | Precision và F2 sau selection | Trung bình, có cơ sở nghiên cứu | Phụ thuộc nhãn, miền/ngôn ngữ và hiệu chỉnh lại gate |
| Cache/batching/observability | Thời gian, số vòng thử và tái lập | Cao về lợi ích vận hành | Không trực tiếp tăng score relevance |

“Cao/trung bình” là mức ưu tiên dựa trên bằng chứng và chi phí kiểm chứng, không phải xác suất thành công hay cam kết tăng điểm.

Các metric đã ghi phù hợp với FINAL_SCORE ≈ (DOCS_F2MACRO + CHUNKS_F2MACRO)/2. Chưa truy cập được scorer để xác minh độc lập; nếu công thức này đúng, các kịch bản **giữ DOCS_F2=0.2802** là:

| CHUNKS_F2 đạt được | FINAL_SCORE tương ứng | Tăng so với 0.1681 |
|---|---:|---:|
| 0.0560, mốc đã ghi | 0.1681 | 0 |
| 0.0800 | 0.1801 | +0.0120 |
| 0.1000 | 0.1901 | +0.0220 |
| 0.1500 | 0.2151 | +0.0470 |

Đây là tính độ nhạy của score, không phải dự báo đạt0.19–0.22. Tăng DOCS_F2 thêm0.04 cũng đóng góp FINAL_SCORE thêm0.02 theo cùng công thức; vì vậy giữ tối ưu document song song với ưu tiên passage. CHUNKS_F2 thấp cho thấy nhiều khoảng cải thiện, nhưng chưa xác định bao nhiêu lỗi thuộc coverage, extraction, ranking hay matching.

F2 tính từng query rồi macro. Không tính F2 từ macro precision/recall để thay thế scorer và không suy số gold bằng tỷ số hai trung bình. Mục tiêu là tìm thêm positives và giảm false positives có kiểm soát, không chỉ tăng precision hay số lượng output.

## 6. Điều kiện nhận thay đổi và bàn giao

1. Mỗi run giữ config, model revisions, text/corpus/query hashes, ZIP SHA-256, phase/time, bảy metric và chi phí inference. Pin revision cả stage1 trước khi rebuild; stage1 hiện chưa pin trong `index.py`.
2. Dùng cùng scorer/phase và holdout để so. Với nhãn pool, báo rõ phạm vi và số positives mới/lost; không gọi pool recall là recall toàn corpus.
3. Với thử nghiệm chỉ chunk, xác nhận tập docs giữ nguyên. Với thử nghiệm chỉ document gate, kiểm ảnh hưởng parent filter đến chunks thay vì giả định chunks độc lập.
4. Đề xuất nhận thay đổi khi FINAL_SCORE tăng trên mốc đối chứng; chunk-focused run phải tăng CHUNKS_F2. Mức giảm recall cho phép chốt trên dev và báo rõ, không tự tuyên bố cả precision/recall cùng tăng.
5. Nếu đủ nhãn từng query, bootstrap theo query để đánh giá độ ổn định; leaderboard chỉ có aggregate thì ghi giới hạn chưa đo được significance.
6. Mỗi vòng chấm chọn số ít biến thể tốt trên dev. Khi hết ngân sách đo hoặc candidate không thắng, giữ bản thắng hiện tại và chuyển hướng theo error breakdown.

Lịch đề xuất: P0/P1 trong hai ngày công đầu; P2/P3 trong các ngày tiếp theo; P4/P5 khi lỗi và điểm đối chứng chứng minh cần thiết. Ước lượng ngày công là thời gian triển khai/pilot, chưa gồm gán nhãn, lượt chấm ngoài hệ thống và thời gian rebuild/full inference. Fine-tune tách riêng sau khi đủ nhãn.

Bàn giao cuối mỗi vòng: `plan/manifest`, `experiments.csv`, configs, diagnostics, ZIP ứng viên và báo cáo trước/sau có phân tích lỗi. Lượt này tạo kế hoạch nghiên cứu; chưa chạy thêm inference hoặc gửi submission.

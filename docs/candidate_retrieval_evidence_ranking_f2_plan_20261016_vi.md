# Kế hoạch tối ưu Candidate Retrieval, Evidence Ranking và F2

Ngày lập: 16/10/2026. Đây là kế hoạch triển khai; chưa chạy thí nghiệm mới hay triển khai cấu hình.

## 1. Mục tiêu và điểm xuất phát

Mục tiêu chính: tăng F2 ở đầu ra document/evidence trên cùng tập query, corpus và scorer. Recall@K, MRR, retention và latency dùng để giải thích kết quả; không thay thế F2 cuối pipeline.

Các mốc hiện có phải được phân biệt:

| Mốc | Phạm vi | Kết quả | Ý nghĩa |
|---|---|---|---|
| ICD-10 reranking final 16/10 | BM25 top-100, 9.277 passages, 72 evaluation queries | Recall@10 0,8071 → 0,8643; chưa qua kiểm định; NO-GO | Chưa chứng minh ICD reranking có lợi đủ chắc chắn |
| Retrieval improvement 16/10 | Cùng pilot; BM25, dense BGE-M3, weighted RRF | Dense Recall@10 0,9571; fusion 0,9429; fusion MRR 0,7746; intent rerank bị loại | Có dense/fusion pilot; chưa chứng minh kết quả toàn corpus hay F2 chính thức |
| Feedback baseline được recheck 09/10 | 593 queries được chấm trong 600 diagnostics; submission có 1.200 queries | DOCS_F2=0,280164632426; CHUNKS_F2=0,055969257307; FINAL=0,168066944867 | Đối chứng có feedback chấm thật, cần nối chính xác ZIP/hash/phase |

Dense có Chunk F2@10 proxy 0,3475, cao hơn fusion 0,3424; fusion có Doc F2 proxy 0,5078, cao hơn dense 0,5062. Vì vậy không chọn fusion chỉ dựa trên MRR, cũng không suy F2 chính thức từ kết quả pilot.

72 query evaluation cũ đã được phân tích qua nhiều vòng: giữ làm regression set, không tiếp tục gọi là holdout mới. Tập 9.277 passages cũng không đại diện đầy đủ độ khó của corpus phục vụ thực tế.

## 2. Luồng triển khai

Corpus/passages cố định → document/passage candidate retrieval → fusion và phân bổ passage → cross-encoder ranking → chọn documents/evidence → validator → scorer.

Tối ưu ba nhóm riêng: C (candidate), R (ranking), S (selection). Chỉ kết hợp sau khi đã đo đóng góp riêng trên development. ICD/intent heuristic hiện có giữ vai trò comparator phụ; không tự động đưa vào cấu hình chính.

## 3. T0 — Khóa baseline, scorer và hợp đồng thí nghiệm

### Công việc

1. Nối feedback baseline với ZIP SHA-256, query IDs, corpus revision, chunks hash, candidate cache, model revision và selector thực tế. File selector đang lưu threshold -2,5 không mặc nhiên là bản được chấm 0,1681.
2. Chọn một đường chạy end-to-end cho mọi arm. Dùng các thành phần `r2ai` đã có ranking cache/selector/validator để dựng đối chứng; chỉ chuyển sang V4 sau khi kiểm tra tương đương đầu ra.
3. Xác minh cách nối official document ID và passage về nguồn; phân biệt content dedup với việc giữ các official ID khác nhau.
4. Tiếp nhận scorer, tokenizer và gold nếu sẵn có; nếu không, chuẩn bị quy trình chấm các ZIP đã khóa trên cùng phase. Dataset card công khai không thay thế scorer.
5. Kiểm normalization, parent matching, overlap, duplicate handling, macro averaging, empty-gold và prediction rỗng theo scorer thực tế. Feedback hiện tại có 7 query `excluded_empty_gold`; không tự coi chúng là negative.
6. Tái tính baseline từ per-query diagnostics. Khi có scorer/gold, chạy lại baseline và yêu cầu khớp trong sai số số học/độ làm tròn được quy định.

### Điểm chặn và đầu ra

- `baseline_manifest.json`, `scorer_contract.md`, `baseline_metrics.json`, `experiment_registry.csv`.
- Nếu thiếu scorer/gold hoặc chưa có lượt chấm mới, vẫn triển khai benchmark, cache và pilot, nhưng đánh dấu F2 chính thức của arm mới là `UNMEASURED`.
- Không suy điểm arm mới từ matched-positive retention của baseline.

## 4. T1 — Benchmark mới và truy vết mất evidence

### Công việc

1. Dùng 107 query cũ và feedback đã phân tích làm development/regression. Ưu tiên xem 61 query có doc TP nhưng chunk F2 bằng 0 để tìm giả thuyết; không dùng nhóm đã chọn theo lỗi này làm đại diện cho toàn bộ tập test.
2. Tạo benchmark mới mục tiêu 200 query chưa dùng trong các vòng trước: khoảng 120 development, 80 holdout; điều chỉnh quy mô theo chi phí nhãn. Tách nhóm query gần trùng trước khi chia; ghi seed và IDs trước tuning.
3. Query chính thức chủ yếu tiếng Việt: phân tầng thêm theo ngôn ngữ nguồn VI/EN/ZH, cross-language, độ dài, nhiều vấn đề, thuốc/liều lượng, phủ định, câu hỏi cần nhiều tài liệu. Giữ các nhóm query EN/ZH pilot ở báo cáo riêng.
4. Pool để gán nhãn là union từ BM25, dense, hybrid, passage siblings và nguồn đã biết; chấm ẩn tên hệ thống. Lưu doc relevance, evidence text/source offsets, intent match, mức đủ ngữ cảnh và `unjudged` rõ ràng.
5. Dùng nhãn pool cho F2 trong phạm vi pool; không xem tất cả phần ngoài pool là non-relevant. Muốn kết luận F2 chính thức cần gold/scorer hoặc feedback mới tương ứng.
6. Ghi trace theo query: source không có nội dung → passage không chứa evidence → doc ngoài pool → passage ngoài shortlist → ranking thấp → gate/cap/dedup loại → output không match.

### Metrics

- Candidate: doc recall và evidence coverage ở K=50/100/200/400; unique parents; phần positive mới của từng retriever; ANN recall so với exact search trên mẫu nếu dùng ANN.
- Ranking: MRR/nDCG trên judgments phù hợp, positive survival qua các độ sâu.
- End-to-end: F2 trên mọi query đủ điều kiện theo scorer, kể cả query không có containing passage. Có thể báo thêm retrieval metrics có điều kiện, nhưng không bỏ các lỗi coverage khỏi F2 cuối.

### Đầu ra

`benchmark_manifest.json`, `judgments`, `split_ids`, `stage_survival.parquet`, `error_taxonomy.csv`.

## 5. T2 — Tối ưu Candidate Retrieval

### Ma trận C

| Arm | Thay đổi | Đối chứng được giữ |
|---|---|---|
| C0 | Candidate pipeline thực tế đã khóa ở T0 | Ranking và selector baseline |
| C1 | BM25-only | Cùng corpus, final candidate budget và downstream |
| C2 | Dense BGE-M3-only | Cùng corpus, final candidate budget và downstream |
| C3 | BM25 + dense weighted RRF | Cùng corpus, final candidate budget và downstream |
| C4 | Cấu hình candidate tốt nhất + passage allocation theo parent | Cùng budget cặp rerank với comparator |
| C5 | Tăng pool/rerank budget | Đo riêng lợi ích tăng compute |

Nếu C0 đã trùng một arm thì tái dùng kết quả tương đương thay vì chạy trùng.

### Công việc

1. Audit dense artifact: corpus/chunk IDs, thứ tự embedding, text/title format, model revision, CLS pooling, normalization, truncation và missing rows. Chỉ tái dùng cache khớp fingerprint.
2. Chạy C1/C2/C3 ở final pool 100 trước. Thử trọng số BM25 0,25/0,5/0,75, dense là phần còn lại; RRF k=20/60 trên development. Cấu hình pilot 0,5/k20 là ứng viên khởi đầu.
3. Sau đó thử pool 200/400 trên shortlist cấu hình. Báo riêng kích thước mỗi nhánh retrieval, union trước fusion và final pool; tránh gọi cả ba cùng là `top-K`.
4. Phân biệt hai điểm cắt: document pool và passage pool. Thử tăng document pool khi mất đúng document; thử passage union BM25+dense và parent-first khi doc đã có nhưng evidence bị mất.
5. So global shortlist với chiến lược ưu tiên một passage cho các parent phù hợp rồi bổ sung passage 2–4. Cùng tổng budget để đo hiệu ứng diversity; quota không ép các parent yếu vào output.
6. Giữ corpus/chunk text cố định vòng đầu. Nếu source hoặc boundary là nguyên nhân chính, mở một pilot text-window riêng trước khi rebuild.
7. Tái dùng BGE-M3 hiện có. Thử sparse/multi-vector hoặc dịch/expansion chỉ khi trace chỉ ra cần thiết, ở vòng sau có budget riêng.

### Đầu ra và chọn cấu hình

`candidate_ablation.csv`, `candidate_pool.parquet`, `candidate_provenance.json`, coverage theo stage/source language, chi phí index và latency.

Chọn C* theo F2 development sau cùng ranking/selector cố định; Recall@K giúp giải thích. Không chọn chỉ theo MRR hoặc top-100 coverage.

## 6. T3 — Tối ưu Evidence Ranking trên pool cố định

### Công việc

1. Khóa candidate identities/text của C* để so ranking. Đối chiếu thứ tự retrieval với BGE-reranker-v2-m3 trên cùng pool.
2. Chấm đầy đủ passage candidates cần so; không chỉ chấm evidence đầu tiên mỗi document. V4 hiện rerank `evidence[:1]`, nên phải bổ sung score cho từng evidence nếu dùng V4 cho thí nghiệm này.
3. Log số tokens thực của query + title + passage, truncation và vị trí evidence. Chạy A/B riêng về title/context/max length trong giới hạn model; rerank lại khi input text thay đổi.
4. Giữ clinical ask cùng các điều kiện liên quan như tuổi, thai kỳ, bệnh nền, phủ định, liều lượng. Không dùng focused-query heuristic cũ làm mặc định vì pilot đã có regression.
5. Nhóm negatives cần kiểm: cùng bệnh/sai ý hỏi; cùng thuốc/sai liều; cùng triệu chứng/khác đối tượng; passage thiếu ngữ cảnh. Fine-tune hoặc model khác chỉ là bước sau nếu nhãn cho thấy ranking còn là nút thắt.
6. Tách score evidence khỏi score document. Thử doc aggregation `max` và top-2 trên dev với cùng evidence scores; đánh giá tác động đến F2 document và evidence.
7. Cache theo query/text/title/model revision/tokenization/max length/precision. Đổi selector được replay CPU; đổi model hoặc input yêu cầu inference tương ứng.

### Đầu ra

`evidence_scores.parquet`, `ranking_ablation.csv`, `truncation_audit.csv`, `ranking_errors.md`. Chọn R* theo F2 sau selector cố định; báo ranking metrics kèm theo.

## 7. T4 — Tối ưu Evidence Selection cho F2

### Công việc

1. Replay từ score cache để thử riêng doc gate, chunk gate, relative margin và max outputs. Giá trị threshold lấy từ development; raw logits không được coi là xác suất.
2. Thử chunk cap 1/2/3 mỗi doc và budget 60/80/120 khi validator/luật submission cho phép. So parent-first với global score-order tại cùng budget; tách khỏi thí nghiệm tăng budget.
3. Thử dedup có kiểm soát. Feedback cũ cho thấy cap2/budget60 làm mất 695 baseline matches và đổi dedup có thể làm mất 152; cần trace tương tác cap/gate/dedup.
4. Theo dõi giữ/mất 2.190 matched baseline predictions như regression diagnostic. Giữ toàn bộ số này không bảo đảm F2 tăng vì predictions mới làm đổi precision và matching assignment.
5. Chỉ thử context window cho nhóm boundary fail: lấy text liên tục từ nguồn, giữ parent/offset đúng, score lại input và chấm lại output text. Không ghép text rời hoặc tự sinh evidence.
6. Adapter phải xuất đúng tập evidence đã được selector chọn. V4 hiện xuất `evidence[:1]`; cần hỗ trợ nhiều evidence/doc trước khi đo arms cap>1.
7. Không chuyển threshold -2/-2,5 sang model hoặc input format mới mà không hiệu chỉnh lại.

### Đầu ra

`selection_ablation.csv`, `selection_config.json`, `selection_diagnostics`, `validated_submission.zip`, predicted counts và số positive giữ/mất/thêm theo judgments có sẵn.

## 8. T5 — Đo đóng góp thật và khóa ứng viên

Chọn tham số chỉ trên development. Các cấu hình sau được khóa trước khi chạy holdout mới:

| Arm | Candidate | Ranking | Selector | Mục đích |
|---|---|---|---|---|
| E00 | C0 | R0 | S0 | Baseline |
| E10 | C* | R0 | S0 | Đóng góp candidate |
| E01 | C0 | R* | S0 | Đóng góp ranking |
| E11 | C* | R* | S0 | Tương tác candidate và ranking |
| E111 | C* | R* | S* | Gain thêm từ selection và kết quả cuối |

R* được áp dụng với cùng quy tắc trên C0 cho E01; mọi arm ghi budget thực tế. Nếu R*/S* không tốt hơn thì giữ R0/S0, không bắt buộc thêm stage.

### Protocol đánh giá

- So sánh chính đăng ký trước: E111 với E00; các arm còn lại giải thích đóng góp, không dùng holdout để chọn lại cấu hình. Nếu sửa theo holdout, cần test độc lập tiếp theo.
- DOCS_F2MACRO, CHUNKS_F2MACRO, doc/chunk precision và recall, FINAL_SCORE dùng đúng scorer. F2 mỗi query là `5PR/(4P+R)`, rồi macro trên tập eligible của scorer; không tính F2 từ macro P/R.
- Feedback cho thấy matched predictions và gold chunks found khác nhau: baseline có 2.190 matched predictions nhưng 2.214 gold chunks found. Precision và recall phải dùng counts tương ứng của scorer, không ép one-to-one TP/FN.
- Paired bootstrap 10.000 lần theo query, hoặc theo nhóm query gần trùng, cho delta F2/FINAL nếu có per-query results. Chỉ có aggregate score thì không tự dựng paired CI.
- Báo cùng query IDs/phase/scorer/corpus và SHA của ZIP. F2 nhãn pool và known-positive proxy nằm trong cột riêng.
- Đo riêng index-build cost và online latency; warm/cold, mean/P95, GPU VRAM, RAM, số cặp rerank và bytes cache. Benchmark throughput trên 50 query trước full run để dự toán tài nguyên.

### Tiêu chí nhận thay đổi được đề xuất

1. Primary: FINAL_SCORE tăng; mục tiêu thực dụng ban đầu là delta ≥0,005 tuyệt đối, cần khóa trước test. CI 95% của delta >0 khi có đủ per-query results; đây là tiêu chí đề xuất, không phải dự báo đạt điểm.
2. Arm tập trung evidence phải tăng CHUNKS_F2; DOCS_F2 giảm không quá 0,005 tuyệt đối theo guardrail đăng ký trước. Báo riêng recall regression và nhóm ngôn ngữ nguồn.
3. Nếu gain chưa đủ chắc chắn: ghi `INCONCLUSIVE`, không ép quyết định thắng/thua bằng proxy.
4. Latency/RAM/VRAM phải nằm trong budget được ghi trước full run. Budget phục vụ chưa có SLA nên cần chốt sau benchmark T0, không áp lại máy móc giới hạn 1,5x từ ICD pilot.
5. Chỉ gọi là tăng F2 chính thức khi có scorer/gold tương ứng hoặc feedback mới của đúng arm. Bản GO trong pilot cũ không đồng nghĩa cấu hình đã được deploy.

## 9. Thứ tự thực hiện và ước lượng

| Thứ tự | Tác vụ | Ước lượng ngày công kỹ thuật | Phụ thuộc |
|---|---|---:|---|
| 1 | T0: baseline/scorer/artifact audit | 0,5–1 | ZIP/hash/feedback và dữ liệu hiện có |
| 2 | T1: benchmark, split, tracing | 1–2 | Nhãn/scorer; review thủ công tính riêng |
| 3 | T2: candidate ablation trên pilot rồi corpus target | 1–2 | T0/T1, index hợp lệ |
| 4 | T3: evidence ranking/cache/truncation | 1–2 | Pool đã khóa, GPU benchmark |
| 5 | T4: selector replay và output adapter | 0,5–1 | Score cache |
| 6 | T5: holdout, ZIP validation, báo cáo | 0,5–1 | Cấu hình đã khóa, đường chấm có sẵn |

Tổng khoảng 4,5–9 ngày công kỹ thuật, chưa gồm thời gian gán nhãn, xây index/embedding toàn corpus và chờ scorer/feedback. Sau T0 sẽ có dự toán theo số passages, số cặp và throughput đo thật. Nhãn development và xây cache có thể được thực hiện đồng thời khi quy trình review cho phép.

Bàn giao cuối: manifest dữ liệu/scorer, benchmark và phạm vi nhãn, ablation C/R/S, per-query deltas, error analysis, chi phí tài nguyên, cấu hình khóa, ZIP hợp lệ và kết luận GO/NO-GO/INCONCLUSIVE cho từng thành phần. Kế hoạch hiện tại chỉ chuẩn bị kết quả review được; lượt nộp leaderboard và triển khai là bước riêng.

## 10. Nguồn kiểm tra

- [ICD-10 final evaluation](../outputs/vibiomir_icd10_reranking_final_20261016/ICD10_RERANKING_FINAL_EVALUATION.md).
- [Dense/fusion pilot report](../outputs/vibiomir_retrieval_improvement_20261016/RETRIEVAL_IMPROVEMENT_REPORT.md).
- [Scoring feedback recheck](../outputs/scored_feedback_recheck_20261009/SCORING_RECHECK_REPORT.md).
- [ViBioMIR dataset card](https://huggingface.co/datasets/AIGuruTinix/ViBioMIR/blob/main/README.md): query tiếng Việt, nguồn VI/EN/ZH; trang này chưa cung cấp đủ implementation scorer để tái lập matching.
- [BGE-M3 documentation](https://bge-model.com/bge/bge_m3.html): dense CLS normalization, sparse và multi-vector capabilities; không suy mức cải thiện ViBioMIR từ khả năng model.
- [BGE reranker model card](https://huggingface.co/BAAI/bge-reranker-v2-m3): đối chiếu interface scoring của model hiện dùng.

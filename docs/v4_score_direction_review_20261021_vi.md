# Review hướng cải thiện điểm V4 — 21/10/2026

## Kết luận

V4 có cơ sở hợp lý để cải thiện chất lượng: dùng đúng corpus, bổ sung nội dung official còn thiếu, chia passage có provenance, tìm kiếm toàn bộ eligible passages và sửa đường selection/export. Nhưng hiện chưa có phép đo chứng minh V4 tăng DOCS_F2/CHUNKS_F2 so với baseline đã chấm. Hoàn thành mọi gate kỹ thuật không đủ để kết luận tăng điểm.

Rủi ro đáng ưu tiên nhất trước full inference là policy A0: selector và validator buộc mỗi document xuất ra phải có evidence, trong khi tổng evidence bị giới hạn 60. Vì vậy `max_docs=200` thực tế chỉ cho tối đa **60 documents/query**. Đây là lựa chọn có thể đánh đổi document recall lấy precision; chưa có ablation chứng minh lợi ích đó. Không tự sửa cấu hình đang dùng để build dense trong lượt review này.

Phạm vi thực hiện: đọc code/config/report; truy vấn chỉ đọc theo primary/parent index trên sparse store; đối chiếu CSV feedback và ZIP baseline; ghi báo cáo mới. Không gọi model, chạy full verifier, assemble, inference hoặc export. Job dense hiện tại không bị thay đổi.

## Cách hiểu đúng mục tiêu cuộc thi

- Document chỉ được tính đúng nếu ID nằm trong gold của chính query. Tài liệu tương tự hoặc trùng nội dung nhưng khác official ID không tự được tính đúng.
- Chunk phải đúng query/doc_id và đạt token-LCS/gold-length >= 0,4 sau chuẩn hóa BTC. Reranker đánh giá liên quan ngữ nghĩa không trực tiếp tối ưu điều kiện này.
- Mẫu số overlap là độ dài gold, nên thêm đúng ngữ cảnh nguồn có thể tăng coverage. Không có phạt trực tiếp theo số token chunk trong công thức được cung cấp, nhưng còn caps, duplicate handling và giới hạn submission cần đối chiếu scorer. Không suy ra nên xuất cả bài hoặc nối text tùy ý.
- Một chunk có thể tìm thấy nhiều gold chunks; không được mặc định chunk recall bị chặn bởi `submitted_chunks / gold_chunks`.
- F2 ưu tiên recall nhưng vẫn phạt false positives. Với document của một query, `F2 = 5*t / (4*g + d)`, trong đó t là true positives, g là số gold docs và d là số docs nộp. Tăng d chỉ có lợi khi tìm thêm đủ tài liệu đúng.
- Macro-average phải tính P/R/F2 từng query rồi lấy trung bình từng metric. Không tính F2 từ hai macro P/R, không dùng micro counts để thay macro.

Các quy tắc scoring ở trên lấy từ nội dung người dùng cung cấp. Feedback lịch sử còn ghi duplicate threshold 0,8 và tokenizer riêng, nhưng bytes tokenizer và full gold chưa có trong bộ tài liệu local đã kiểm tra. Bởi vậy một scorer dùng whitespace/BGE tokens chỉ là proxy nếu chưa chứng minh tương đương.

## Review các nhóm công việc

| Nhóm | Giá trị đối với điểm | Bằng chứng và giới hạn |
|---|---|---|
| Xác minh official corpus, giữ official IDs | Tránh lấy nhầm corpus hoặc đúng nội dung nhưng sai identity | Snapshot official có 1.220.598 URLs. Audit cũ ghi index có 833.892 IDs thuộc snapshot này, cùng nhiều IDs ngoài scope. Sparse V4 có 1.073.525 eligible official sources. Đây là coverage theo ID, không phải gold recall. |
| Crawl/recovery và PubMed | Tạo cơ hội truy hồi gold trước đây chưa có nội dung | Có cơ sở vì inventory official có 240.000 PubMed URLs thiếu trong corpus local cũ. Chưa đo end-to-end contribution trên cùng scorer/phase. |
| Phase 1 cleaning/quality | Loại boilerplate, nguồn lỗi; giữ text để match evidence | Hữu ích nếu không loại nhầm gold. V4 giữ 2.091/2.092 parents của historical matched chunks; một parent bị thiếu cần chẩn đoán. Không chứng minh coverage toàn gold. |
| Phase 2 chunking A/B/C | Có thể giảm evidence bị đứt hoặc mất ngữ cảnh | Pilot 107 labels: A Recall@10 = 0,7664; C = 0,8131. C tốt hơn trên proxy này, nhưng CI delta C–B Recall@10 bao gồm 0. Đây là BM25 trên corpus nhỏ với manual span-containment relevance, không phải scorer LCS/full corpus. |
| Phase 3 exact/near dedup, canonicalization | Giảm trùng lặp và compute khi áp dụng đúng | Baseline full hiện không tiêu thụ Phase 3 để gộp official IDs. Không gán gain cho baseline từ bước chưa dùng; nếu áp dụng phải giữ khả năng trả riêng các gold IDs. |
| Task 2/3 integration, CLI/API, provenance | Đưa đúng passage/text qua các entrypoints | PASS integration/repeatability không đo relevance. Một subset không làm giàu positives không thích hợp để kết luận F2. |
| Sparse/dense full trên đĩa | Cho phép tìm kiếm toàn eligible release và tránh bỏ passage cuối | Sparse đã có SPARSE_FULL_PASS. Dense đang hoàn thành; encode đủ rows chưa chứng minh candidates được chọn tốt hơn. |
| Rerank/selector/cache/export | Có thể tăng precision và chọn đúng đoạn; cache giúp thử policy rẻ | Cap ngầm 60 docs, allocation document-order và tối đa 3 evidence/doc còn rủi ro. Các arm A0/A1/A2 đã có output khác nhau, chưa có F2 tương ứng. |
| Integrity/resume/hash/tests | Ngăn hỏng artifact, mất query, chạy lại tốn GPU | Cần cho vận hành nhưng không trực tiếp chứng minh điểm tăng. Không nên mở rộng vô hạn công việc hạ tầng trước khi đo chất lượng. |

Corpus card của nhà phát hành xác nhận 1.200 queries, 1.220.598 URLs và 240.000 nguồn tiếng Anh/PubMed: https://huggingface.co/datasets/AIGuruTinix/ViBioMIR/blob/main/README.md . Các số liệu index local lấy từ `outputs/vibiomir_official_verify_20261012/verification.json` và sparse reports; không suy coverage từ số docs local tổng cộng.

## Findings ưu tiên

### R1 — Cao: document output đang phụ thuộc chunk budget

`configs/v4_phase2_full_a0_gpu3.yaml`: `max_docs=200`, `max_chunks=60`, `chunks_per_doc=1`. `select_results()` chỉ append document khi có `kept`; validator còn từ chối document không có evidence. Bởi vậy giới hạn thực là 60 documents. Tăng `chunks_per_doc` lên 2 trong khi giữ tổng 60 có thể còn giảm số parents xuất ra.

Tính mới trên feedback lịch sử: 298/593 câu có >60 gold docs; median 61, mean 71,81, max 190. Đây là rủi ro recall có thật. Tuy nhiên oracle macro DOCS_F2 dưới cap60 vẫn tới 0,8136 và không câu nào có baseline F2 vượt oracle cap60, nên **cap60 không chứng minh V4 sẽ giảm điểm hoặc không thể vượt baseline 0,2802**. Cần đo actual positives sống sót, không chỉ nhìn cardinality ceiling.

Hướng kiểm chứng: thử budgets document/chunk riêng nếu luật submission cho phép; hoặc thử mức cap chung khác, so actual F2 trên cùng conditions. Hiện yêu cầu mỗi doc có chunk là implementation contract của V4; đoạn scoring người dùng đưa không xác lập đây là yêu cầu BTC. Không tự nới validator trước khi đối chiếu schema.

### R2 — Cao: passage recall có thể mất trước reranker

Pipeline lấy document BM25 400, passage BM25 200, dense passages 200 rồi fusion giới hạn 400 mixed doc/passage keys. Đây không phải bảo đảm 400 unique parents hoặc mọi passage của 400 parents đều được rerank.

Allocation A0 lấy theo document order, tối đa 3 evidence/doc và tổng 200 pairs. Document chỉ có hit ở nhánh document nhận fallback đúng một passage theo parent. Chỉnh output cap 2 không tự tạo thêm passage candidates hoặc scores. Parent-first đã có trong code nhưng chưa đo chất lượng trên full corpus và không thể đảm bảo tốt hơn A0.

Hướng kiểm chứng: đo historical known-positive survival tại store → branch candidates → fused parents/passages → 200 pairs → selected/export. Khi mới có nhãn một phần, gọi đây là known-positive survival, không gọi official recall/F2. Nếu evidence mất trước rerank, không giải quyết bằng thêm GPU cho chính 200 pairs cũ.

### R3 — Cao: relevance score và chunk match chưa được nối bằng phép đo đúng

Baseline đã chấm: DOCS_F2 0,2801646; CHUNKS_F2 0,0559693; FINAL 0,1680669 trên 593 scored queries. Trong 31.294 predicted chunks: 21.736 có `unknown_doc`; 7.368 có parent được scorer nhận nhưng không match đoạn; 2.190 match. Có 61 câu đúng document nhưng CHUNKS_F2 bằng 0.

Đây là hai vấn đề khác nhau: chọn đúng parent và chọn đúng span/text. Không quy tất cả `unknown_doc` thành ID hỏng hoặc ngoài official corpus: tài liệu có thể hợp lệ trong corpus nhưng không có gold evidence cho query.

AUC rerank logit đối với chunk-match trong selected set cũ chỉ 0,5476. Nó cảnh báo không nên mặc định score cao sẽ đạt LCS, nhưng không chứng minh reranker hiện tại kém trên toàn candidate pool và không dự đoán V4 sau khi đổi dữ liệu.

Hướng kiểm chứng: ưu tiên các câu có đúng parent nhưng sai đoạn; dùng exact-source spans, kiểm evidence ở biên passages và khác biệt extraction. Có thể thử 1/2 đoạn không trùng hoặc context expansion hợp lệ trên một arm nhỏ. Không bắt đầu bằng re-embed toàn bộ corpus hoặc fine-tune.

### R4 — Trung bình: chỉ đổi dữ liệu vẫn có thể làm mất positive cũ

Phép kiểm mới chỉ đọc sparse store và baseline ZIP, đối chiếu đúng query/chunk_order/doc_id với feedback:

| Diagnostic | Số lượng |
|---|---:|
| Historical chunks được chấm đúng | 2.190 |
| Parents duy nhất của các chunks đó | 2.092 |
| Parents vẫn có trong V4 eligible store | 2.091 |
| Old matched chunks có parent vắng trong V4 | 1 |
| Old chunk sau chuẩn hóa nằm nguyên trong nguồn V4 | 1.846 |
| Old chunk sau chuẩn hóa nằm nguyên trong một passage V4 | 1.346 |
| Có parent nhưng không tìm thấy nguyên old chunk trong nguồn | 343 |

Parent thiếu: `1952175`. Chưa xác định nguyên nhân trong lượt này. Không tự khôi phục hoặc sửa release.

Không tìm được toàn đoạn cũ không có nghĩa fail LCS 40%: có thể đổi extraction, whitespace/punctuation hoặc segmentation và vẫn match gold. Ngược lại, có text trong store không chứng minh query retrieval/selection giữ nó. Các con số này giúp khoanh vùng regression; không dùng làm F2.

### R5 — Trung bình: thời gian inference full chưa được đo trên full corpus

`DiskDense.search()` hiện quét toàn ma trận trên CPU, chuyển block sang float64 và tính norm/cosine từng query. GPU dùng query encoder/reranker; không thực hiện phần tìm kiếm dense này. Ma trận FP32 khoảng 11,26 GiB được duyệt mỗi query; số liệu này không có nghĩa mỗi query nhất thiết đọc toàn bộ từ ổ đĩa vì OS có thể cache.

Pilot 12.347 vectors không đủ để hứa ETA full 2.951.224 vectors × 1.200 queries. Đo latency representative queries trên full index trước khi quyết định workload. Chỉ tối ưu backend nếu benchmark cho thấy cần thiết và có parity check; không tự mở một migration mới trong lúc dense đang chạy.

## Thứ tự đề xuất để giảm chi phí trước deadline

1. Giữ dense hiện tại hoàn thành, tái sử dụng artifact. Chưa có lý do từ review này để crawl/embed lại toàn bộ hoặc đổi model.
2. Sau khi được phép mở bước tiếp theo, hoàn tất kiểm integrity cần thiết và benchmark nhỏ trên full index; đo stage latency trước khi ngoại suy 1.200 câu.
3. Khóa một pilot chất lượng nhỏ trước full inference, chẳng hạn 20 official queries có positive cũ và manual evidence; ưu tiên nhóm đúng doc/sai chunk, thêm đại diện tiếng Việt/Anh/Trung và câu dài. Sample chẩn đoán không đại diện toàn contest. Chạy retrieval trên **full corpus**, không chỉ 1.000 docs.
4. Lưu đủ branch candidates, fused/pair candidates, exact model inputs, scores và selected results. Đo mất positive ở từng tầng. Chỉ rerank pairs cần thiết một lần cho các policy có thể dùng chung cache; đổi allocation/input sẽ có pairs mới phải score riêng.
5. Replay CPU số arm nhỏ: A0; parent-first cùng 200 pairs; document/chunk budgets được tách hoặc điều chỉnh đúng schema; 1/2 evidence với tổng budgets ghi rõ. So cùng query/pool/conditions, không gán gain cho allocation khi đồng thời tăng compute.
6. Dùng full gold/scorer hợp lệ để đo macro DOCS_F2/CHUNKS_F2 nếu có. Nếu chỉ có partial feedback, báo survival và diagnostics; không coi unjudged là negative để tạo official F2. Phải có feedback đúng ZIP/hash/phase mới kết luận điểm contest tăng.
7. Chạy đủ 1.200 câu với policy đã chọn, cache/resume; xuất một artifact đủ deadline, giữ baseline cũ. Dành thời gian cho nhận feedback và sửa một vòng có mục tiêu. Không đưa thêm tuning ICD10, Phase 3 hay kiến trúc mới vào baseline chỉ vì đã có code.

Review hiện tại chưa triển khai các bước 2–7. Không hứa số điểm mới hoặc xác suất tăng điểm khi chưa có phép đo tương ứng.

## Bằng chứng tham chiếu

- `outputs/v4/score_direction_review_20261021/review_evidence.json`: joins historical positive parents và cardinality bound tính mới.
- `outputs/v4/score_direction_review_20261021/positive_text_preservation.json`: exact normalized containment diagnostic tính mới.
- `outputs/scored_feedback_recheck_20261009/RECHECK_SUMMARY.json`: metrics cũ, selected-set AUC, gold/scorer limitations.
- `outputs/vibiomir_phase2_20261015_retrieval_v2/reports/retrieval_evaluation_v2/retrieval_comparison.json`: proxy A/B/C trên 107 labels, trong đó config có 87 official query IDs và 20 IDs bổ sung.
- `docs/v4_evidence_fixes_execution_20261016_vi.md`: A0/A1/A2 integration, output counts, thiếu F2 từng arm.
- `docs/v4_task4_post_dense_plan_review_20261021_vi.md`: integrity, run ledger, replay/cache và gate gaps sau dense.
- `src/medrag/evidence/selector.py`, `src/medrag/adapters/validation.py`, `src/medrag/core/pipeline.py`, `src/medrag/retrieval/core.py`, `src/medrag/retrieval/disk_dense.py`: hành vi runtime được review trực tiếp.

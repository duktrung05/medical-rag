# Chẩn đoán DOCS_F2, CHUNKS_F2 và hướng thay đổi workflow

Ngày đối chiếu: 26/10/2026. Phạm vi: code, receipts, manifests và diagnostics hiện có; không chạy inference mới hoặc thay champion.

**Nhận định: cần thay cách ra quyết định và đánh giá trong workflow hiện tại. Chưa có bằng chứng phải bỏ toàn bộ kiến trúc phân cấp parent–passage.** Hệ thống đang loại ứng viên ở nhiều tầng, dùng passage score thay cho document relevance, rồi giới hạn evidence trong context đã chọn sớm. Sửa riêng fusion, cap hoặc allocator vì thế thường không đưa evidence đúng tới output.

## Mốc thực tế và cách đọc điểm

Receipt cap-10 do người dùng cung cấp, ZIP SHA-256 đã đối chiếu khớp:

| Thành phần | Precision | Recall | F2 macro |
|---|---:|---:|---:|
| Documents | 0,1946 | 0,3695 | 0,2788 |
| Chunks | 0,2084 | 0,2117 | 0,1860 |

FINAL = 0,2324. Đây là điểm BTC được lưu từ người dùng, không phải một lần truy vấn BTC độc lập trong lượt review này. Không tính lại macro F2 bằng cách thế precision/recall trung bình vào công thức F2: các phép lấy trung bình không giao hoán.

DOCS_F2 chấm danh sách document IDs. Passage sai làm ảnh hưởng DOCS_F2 thông qua quyết định xếp hạng/chọn document; nó không khiến một document ID đúng tự động thành sai. CHUNKS_F2 còn phụ thuộc đoạn text xuất ra và quy tắc matching evidence.

## Evidence mất ở đâu?

Đếm lại CSV audit của 87 manual evidence và CSV pool recovery:

| Phép kiểm tra | Số evidence có mặt / 87 |
|---|---:|
| Có trong source hiện hành | 87 |
| Có passage trong index chứa toàn evidence | 87 |
| Parent có trong refined candidate pool | 76 |
| Passage trong refined pool chứa toàn evidence | 71 |
| Parent có trong cap-10 documents | 76 |
| Có context đúng trong R5 candidate set | 75 |
| Context cố định dùng ở cap-10 chứa evidence | 73 |
| Parent được cấp chunk slot | 67 |
| Chunk cuối chứa đầy đủ evidence | 65 |
| Pool recovery mở rộng có passage chứa evidence | 87 |

Các hàng không phải một chuỗi tập con: mở rộng seed thành context có thể lấy lại evidence nằm ngoài seed. Tập 87 đã được dùng chẩn đoán, không đại diện cho complete gold hoặc một holdout độc lập mới.

22 output misses được phân nhóm không trùng: **11 thiếu parent trong candidate pool, 9 có document nhưng thiếu chunk slot, 2 có slot nhưng sai context**. Một ca có thể mắc nhiều lỗi; q566 vừa thiếu slot vừa thiếu context đúng trong pool cũ. Không suy tất cả 11 parent misses là raw retrieval misses nếu thiếu trace trước fusion.

Pool recovery 87/87 dùng depth 4.000 cho riêng 11 query thiếu parent và enumerate passages của các documents baseline. Đây là diagnostic có điều kiện theo cohort. Nó chưa chứng minh policy depth 800 hiện chờ GPU cũng đạt 87/87, và chưa có kết quả scoring/output từ pool này.

## Bảng lỗi, bằng chứng và hướng cải thiện

| Ưu tiên / trạng thái | Vấn đề | Bằng chứng cụ thể | Hướng tư duy cải thiện |
|---|---|---|---|
| P0 — còn trong đường champion | Cắt mixed document/passage items trước khi gom parent | q610: dense rank 142, mixed-item rank 406, unique-parent rank 285; q1159 cũng được parent-floor thu hồi | Tách budget unique parents khỏi budget passages. Giữ raw branch traces; gom ở cấp parent trước khi áp parent cap, đồng thời giữ passage identities. Không quy mọi fusion drop về lỗi duplicate: q991/q1035 vẫn ngoài top-400 unique parents. |
| P0 — còn | Parent vào pool nhưng passages của nó chưa được khảo sát đủ | 76 manual parents có trong refined pool nhưng chỉ 71 evidence nằm trọn trong candidate passages; index có đủ 87 | Liệt kê indexed passages theo parent rồi lọc với budget rõ ràng. Đo evidence coverage trước/sau shortlist; chỉ score bổ sung các passages đã lọt retrieval không khắc phục phần chưa vào pool. |
| P0 — còn | Khóa một context/parent trước khi evidence scorer so sánh alternatives | `choose_contexts` giữ nguyên S2 chunk cho baseline; outsiders lấy context theo seed logit. q672/q893 có context đúng trong R5 set nhưng fixed context sai | Chọn evidence window trong nhiều vùng của parent, rồi dựng context quanh vùng thắng. Giữ baseline context như một candidate/fallback; không bắt scorer tìm evidence ngoài input mà nó được nhìn. |
| P0 — còn | Document relevance và evidence quality chưa có mục tiêu học riêng | `pipeline.finish_rerank` dùng max passage logit làm document score; S2 cấp slots theo thứ tự document. Cap-10 có điểm window mới nhưng vẫn dùng generic BGE reranker | Định nghĩa nhãn riêng query–document và query–exact-span/context. Hai scorer có thể dùng chung backbone; điều cần tách là mục tiêu và dữ liệu kiểm chứng, chưa nhất thiết phải mua/chạy hai model lớn. |
| P0 — còn | Thu hồi document không đảm bảo evidence nhận slot | 9/22 manual misses có document nhưng thiếu slot. R7 đưa q610/q1159 vào documents ở rank 104/68, vẫn không có chunk | Chọn 60 slots bằng evidence score đã kiểm chứng. Cho evidence mạnh có đường đưa parent vào output thay vì phụ thuộc hoàn toàn document prefix. Kiểm tính nhất quán chunk-parent membership ở bước cuối. |
| P1 — còn | Ba cửa sổ head/middle/tail chỉ nhìn một phần của context đã khóa | Cap-10 `window_ranges` sinh tối đa ba windows; không khảo sát mọi indexed passage. Đổi 11.445 slots chỉ tăng CHUNKS_F2 0,0012 | Sliding windows với coverage được đo, cheap shortlist rồi cross-encoder. So scorer trên windows cùng representation; context export dựng theo source offsets. Tăng cửa sổ chỉ có ích khi đoạn đúng vào pool và scorer chọn đúng. |
| P0 — còn | Thiếu direct judgments để học document precision và chọn challenger | Multi-passage challenger đổi 2.788 documents/1.066 queries, validation known-positive gain/loss +1/-1. Packet 1.554 rows đang `AWAITING_DIRECT_JUDGMENTS` | Chấm nội dung query–document ở các ca entrant/victim, thêm mẫu nền từ candidate pool để giảm lệch do chỉ xem swaps. Nhãn thiếu phải là unjudged trong evaluation. Known-positive retention không đủ đo precision hoặc full recall. |
| P1 — xác minh thêm từ code hiện tại | Logistic ranker vẫn học unlabeled thành target 0, dù giảm weight | `fit_model`: 207.477 unlabeled rows nhận target 0, weight 0,02; tổng weight 4.149,54 so với 1.254 positive, khoảng 3,31 lần | Đây là heuristic weak supervision; không diễn giải log-odds thành xác suất relevance đã calibrated. Bổ sung judged negatives, đo false positives trực tiếp trước khi tune margin/weight. Tỷ lệ weight không tự chứng minh model sai nhưng bác bỏ giả định unlabeled gần như không tác động. |
| P1 — còn hạn chế dữ liệu/model | Điểm liên quan chủ đề chưa đảm bảo trả lời đúng ý hỏi | Report q442: thay document nói ảnh hưởng/tần suất X-quang bằng document nói phát hiện u răng. q684: historical-positive entrant nói đau bên phải trong khi query có đau bên trái | Tạo judged hard negatives cùng chủ đề nhưng sai intent, vị trí, tần suất, đối tượng hoặc điều kiện hỏi. Cho document scorer đọc nội dung/top passages có chọn lọc; thêm thống kê logits không tạo ra thông tin ngữ nghĩa mới. Không biến mismatch thành rule cấm tuyệt đối nếu chưa chấm ngữ cảnh. |
| P0 — lỗi lịch sử đã sửa, giới hạn nhãn vẫn còn | Từng chuyển nhãn old chunk sang text mới và gọi historical anchors là gold | Exact-identity re-audit chỉ còn 4 historical judgments khớp nguyên S2 text: 1 positive, 3 negative; 71.996 S2 chunks chưa có exact historical judgment | Giữ nhãn theo `(query, doc, exact text hash)`. Positive evidence chứng minh parent có nội dung liên quan; negative một passage không chứng minh cả parent negative. Historical anchor preservation, manual containment và official F2 phải báo riêng. Không tiếp tục dùng kết luận R5 dựa trên nhãn chuyển theo parent. |
| P1 — một phần đã sửa | Sai offsets, split và gate từng khiến experiment kết luận quá mạnh | F03 đã sửa 320 source slices sai xuống 0/1.933; F01/F06 cũng đã sửa. Token visibility cũ chưa tính lại; các historical labels đã được xem nhiều vòng | Giữ corrected offsets và registry split chung; không coi lỗi audit cũ là bằng chứng runtime đang cắt sai source. Chốt policy trên development, validation xác nhận; gross loss ghi riêng với gain. Không gọi historical held-out đã xem là benchmark mới độc lập. |
| P1 — giới hạn kiến trúc và vận hành | Tăng depth/full run trước khi xác nhận shortlist giữ evidence làm chi phí phình lên | Pool manual có 259.327 passages. R11 dừng budget; end-to-end recovery chưa chạy retrieval/scoring vì thiếu CUDA, chưa có ZIP/receipt | Dùng cascade và benchmark theo số parents, passages, windows, pairs và thời gian. Đo coverage sau từng prune. Full run chỉ trả lời chất lượng của cả treatment; cần trace để biết tầng nào phải sửa nếu điểm không tăng. |

Corpus coverage là một câu hỏi riêng: review ghi 1.073.525 eligible sources trên 1.220.598 official URLs, với 147.067 chưa Phase1 PASS và 6 exclusions. Chưa biết bao nhiêu gold thuộc phần thiếu. Tập manual 87 có đủ source/index nên không có căn cứ rebuild index cho các ca này; cũng không thể lấy cohort đó chứng minh toàn corpus đủ gold.

## Parent-first có phải thiết kế sai?

Trong code, `rerank_allocation="parent_first"` là cách chia lượt chấm: đi qua một passage của mỗi parent trước rồi mới đi depth tiếp theo. Đó không đồng nghĩa với một mô hình document retriever được học đầy đủ, cũng không có nghĩa mọi passage trong parent đã được đánh giá. Champion còn có supplemental scoring, nên không đúng khi nói champion chỉ chấm đúng một passage/parent; giới hạn còn lại là **chỉ chấm cached candidates đã có**.

Phân cấp document → passage là một hướng có cơ sở trong nghiên cứu [Dense Hierarchical Retrieval](https://aclanthology.org/2021.findings-emnlp.19/). Bài này không chứng minh pipeline hiện tại tốt, nhưng không ủng hộ kết luận rằng mọi parent-first workflow đều có trần thấp.

Tương tự, max passage score không luôn sai về nguyên tắc: nếu một passage trực tiếp trả lời câu hỏi thì đó có thể là tín hiệu document tốt. [PARADE](https://arxiv.org/abs/2008.09093) cũng nêu kết quả aggregation phụ thuộc dạng nhu cầu thông tin; các cách đơn giản có thể hợp hơn khi relevance tập trung trong một passage. Kết quả đó không cho phép kết luận cứ đổi max thành mean/top-k hoặc transformer aggregation là tăng điểm trên dữ liệu này.

**Điều cần bỏ là sự phụ thuộc cứng: parent rank chưa tốt → passages đúng không được xem → context sai bị khóa → evidence scorer không thể sửa.**

## Workflow nên hướng tới

Đây là đề xuất kiến trúc cần so sánh với champion, chưa phải kết quả đã đo:

```text
query
  ├─ document BM25 / document candidates
  └─ passage BM25 + dense / passage candidates
          ↓
  union parents, giữ passage identities và raw traces
          ↓
  parent shortlist + một budget bảo lưu passages mạnh từ nhánh passage
          ↓
  lấy thêm passages theo parent → sinh windows → cheap shortlist
          ↓
  document scorer                  evidence scorer
  query + nội dung nhiều vùng      query + window/span cụ thể
          ↓                              ↓
  chọn documents theo relevance   chọn contexts và tối đa 60 chunk slots
          └──────── hợp nhất, kiểm mọi chunk parent có trong docs ────────┘
```

Nhánh passage đã có ở retrieval hiện tại. Thay đổi đề xuất là giữ các passages mạnh đi tiếp qua những lần cắt theo parent/document, thay vì lấy passage chỉ để gợi ý parent rồi làm mất tín hiệu/vùng nội dung của nó. Budget nhánh này phải khóa trước trên development, không chèn manual IDs hoặc gold evidence vào inference.

Document path cần chọn cả số lượng documents theo relevance đã đánh giá trực tiếp. Không giữ vĩnh viễn mọi document cũ nếu mục tiêu dài hạn là tăng precision; bảo toàn baseline thích hợp cho một challenger hẹp nhưng có thể giữ lại false positives. Ngược lại, chọn evidence không được kéo document IDs thiếu liên quan vào output chỉ vì một generic logit cao.

## Hướng thực hiện để tránh tiếp tục sửa vụn

1. **Giữ một baseline, hoàn thiện bảng đo theo stage.** Cap-10 là baseline đã có receipt. Với mỗi query–evidence đã judge, ghi source/index, raw union, fusion, parent shortlist, passage shortlist, window visibility, slot và final match. Bước nào mất phải được đo ở đúng bước đó; thiếu trace ghi UNKNOWN.
2. **Tận dụng judgments đã chuẩn bị.** Packet 1.554 rows/300 queries có thể bắt đầu cho document branch, kèm kiểm chứng nhãn và mẫu nền. Đánh giá evidence phải nhìn đúng context/span mới sẽ xuất, không kế thừa judgment từ text cũ. Đây là thiếu dữ liệu đánh giá trực tiếp, không phải thiếu thêm một cap để sweep.
3. **So hai giả thuyết kiến trúc bằng cùng ngân sách.** Một arm giữ hierarchy nhưng mở coverage trong parent; một arm thêm đường bảo lưu passages mạnh. Cùng corpus, query split, số inference pairs và candidate traces. Chấm riêng document ranking và evidence selection để biết gain đến từ đâu. Chưa cần rewrite toàn bộ repo hoặc rebuild corpus.
4. **Khi pool đã có evidence nhưng scorer vẫn chọn sai, chuyển công sức sang học relevance.** Dùng judged hard negatives trong cùng parent và giữa các parents, tách mục tiêu document/evidence. Không tiếp tục tăng retrieval depth cho lỗi nằm sau retrieval.
5. **Một full candidate trả lời một câu hỏi đã định trước.** Candidate end-to-end hiện đã khóa policy depth 800 và đang chờ GPU; không sửa nó theo kết quả audit này rồi coi như cùng experiment. Nếu thực hiện candidate đó, giữ nguyên policy, ghi traces và chấm BTC theo kế hoạch của nó. Dù thắng hay thua, nó chỉ kiểm nghiệm treatment cụ thể, chưa chứng minh kiến trúc đã tối ưu.

Không có căn cứ cam kết mốc FINAL 0,4 từ một sửa đổi trên. Cap-10 chứng minh allocation có thể cải thiện nhưng gain còn nhỏ; multi-passage challenger NO_GO cho thấy aggregation của cached scores chưa cung cấp đủ tín hiệu document precision. Hướng đầu tư có căn cứ hơn là coverage có đo lường, judgments đúng đối tượng và scorers phân biệt được trả lời đúng trọng tâm.

## Bằng chứng và kiểm chứng trong lượt này

- [Receipt champion](../outputs/v4/evidence_window_cap10_full_20261025/scoring_receipt_user_20261026.json); đã hash ZIP và đối chiếu receipt.
- [Audit 87 evidence](../outputs/v4/manual87_recovery_audit_20261026/audit.json), [87 dòng](../outputs/v4/manual87_recovery_audit_20261026/all87.csv), [22 misses](../outputs/v4/manual87_recovery_audit_20261026/missing22.csv); đã đếm lại flags và loss groups từ CSV.
- [Pool recovery](../outputs/v4/manual87_parent_recovery_20261026/manual87_target_coverage.csv); đã đếm lại 87/87 candidate parent/span coverage và baseline preservation.
- [Document challenger manifest](../outputs/v4/doc_f2_multipassage_20261026/manifest.json); đã đếm lại 2.788 swaps trên 1.066 queries và tính tổng weighted unlabeled mass từ training counts/policy.
- [Báo cáo document challenger](v4_doc_f2_multipassage_execution_20261026_vi.md), [cap-10 execution](v4_evidence_window_cap10_full_execution_20261026_vi.md), [R7 downstream](v4_r7_score_gate_execution_20261025_vi.md), [evaluation fixes](v4_evidence_evaluation_allocation_20261025_vi.md).
- Code đã đọc: `src/medrag/core/pipeline.py`, `src/medrag/retrieval/core.py`, `src/medrag/evidence/selector.py`, `scripts/v4/run_evidence_window_allocation.py`, `scripts/v4/run_document_relevance_challenger.py`, `src/medrag/ranking/document_relevance.py`.

Chỉ thêm báo cáo này. Các thay đổi code/docs đã có sẵn trong workspace được giữ nguyên. Không chạy training, inference hoặc một lần chấm BTC mới; không tuyên bố đã re-audit toàn bộ source text/token visibility trong lượt này.

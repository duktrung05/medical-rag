# Kiểm tra mất evidence và hướng khắc phục V4 — 21/10/2026

Kết quả hiện tại: pilot `parent_first` trên 6 queries full corpus đã hoàn tất và cải thiện known-positive survival; xem phần cập nhật thực thi ở cuối báo cáo. Không có cơ sở rerank lại toàn bộ 1.200 queries hoặc crawl/embed lại corpus. Full inference và F2 mới chưa chạy.

## Phạm vi và bằng chứng

- Dùng 20 checkpoints full corpus tại `outputs/v4/task4/full_index_acceptance_20261021/attempt_20261021_1550/smoke_queries/`, sau gate `FULL_INDEX_PASS`; đây là A0 `document_order`, không phải kết quả của config independent/parent_first.
- Join feedback baseline với đúng ZIP SHA `7eeacdd8273502fe5d9ba1dcb76d0689d9c3ba9c737749c4b39bc3c0e89a6e20`, kiểm đúng query hash, chunk_order, doc_id và text hash. Trong 20 queries có 10 câu chứa 45 historical matched chunks, thuộc 45 cặp query/document khác nhau.
- Toàn bộ 6.419 candidate evidence được đối chiếu với registered passages, parent và source offsets. Kiểm 4.000 pair scores với exact query hash, passage ID, model input hash và cache score; replay A0 khớp toàn bộ selected checkpoints. Hash inputs và production code không thay đổi trước/sau.
- Bộ 20 câu chỉ giao với nhóm 61 câu “đúng document nhưng chunk F2=0” ở query 1200. Nó không đủ để chẩn đoán toàn bộ nhóm 61 câu hoặc chọn policy contest.

Script tái hiện: `scripts/v4/audit_evidence_survival.py`. Kết quả cuối: `outputs/v4/evidence_survival_audit_20261021_v2/{summary.json,per_query.csv,historical_positive_stages.csv,selection_replays.csv,input_hashes.json}`. Thư mục không có hậu tố v2 là lượt phân tích đầu trước khi thêm diagnostic global selection/context; dùng v2 cho báo cáo này.

```bash
python3 scripts/v4/audit_evidence_survival.py \
  --output outputs/v4/evidence_survival_audit_NEW_DIRECTORY
```

Script từ chối ghi đè output. Đường dẫn output mới là bắt buộc khi chạy lại.

## 1. Có mất parent trước rerank

| Trạng thái của parent của 45 historical matched chunks | Số trường hợp |
|---|---:|
| Có trong store nhưng không nằm trong fused candidates | 4 |
| Có trong fused candidates nhưng không được rerank | 7 |
| Có rerank nhưng không được chọn document bởi A0 | 3 |
| Có document được chọn bởi A0 | 31 |

Tất cả 45 parents vẫn có trong eligible store. “Parent được chọn” chưa khẳng định xuất đúng đoạn hoặc được scorer hiện tại chấm đúng.

200 pairs/query của A0 chỉ phủ **118–193 parents/query**, tổng **2.990 cặp query/parent** trong 20 câu. `parent_first` cấp một pair mỗi parent trước khi cấp đoạn thứ hai có thể phủ **3.994 cặp query/parent** trên cùng frozen candidate pool. Cả 7 parents đang chưa được rerank đều nằm trong first-round top 200 này. Đây là bằng chứng về allocation membership, chưa phải scores hoặc quality của một lần chạy parent_first.

Ví dụ query 130, parent `553119`: có dense rank 96, đứng thứ 197 trong thứ tự fused document; passage chứa nguyên old matched prediction nhưng trạng thái `not_reranked_budget`. Rerank lại chính 200 pairs A0 sẽ tiếp tục bỏ parent này.

Để đánh giá parent_first trên cùng 20 câu cần ít nhất **1.004 pair scores mới** cho parents chưa có score. Đây là lower bound: exact passage được chọn có thể khác. Không phải mặc định chấm lại 4.000 pairs. Cache hiện khóa query, parent, passage, exact title+text, model revision, max_length và scoring contract; chỉ tái sử dụng khi tất cả khớp.

**Hướng sửa ưu tiên:** dùng config independent/parent_first đã triển khai; chưa cần viết lại allocation. Kiểm pilot có 200 distinct parents khi pool đủ, lưu pair order trước rerank và đo known-positive survival. Nếu thiếu scores, chỉ score exact inputs còn thiếu trong cache phù hợp contract.

## 2. Retrieval/fusion và fallback còn thiếu thông tin

Có 4 historical positive parents không có trong fused pool: `(130,2796009)`, `(130,2761519)`, `(754,1644439)`, `(771,563024)`. Store vẫn giữ parent; 3/4 có passage chứa nguyên old matched prediction. Trace hiện chỉ lưu số hits và fused candidates, không lưu raw hit lists từng nhánh. Chưa thể quy nguyên nhân cho BM25, dense hoặc fused limit 400.

Trong 20 câu, **1.282** cặp query/parent chỉ xuất hiện qua document BM25; tất cả được cấp đúng một fallback passage. **1.269/1.282** parents còn có nhiều passages trong store. Đây là giới hạn cơ hội chọn span, chưa chứng minh fallback đang sai ở từng parent.

**Hướng sửa:** bổ sung diagnostics cho pilot: hit lists có parent/passage/rank/score theo từng nhánh, fused order, fallback candidates, pair admission/rejection và selection reason. Với 4 parents trên, đo chúng vắng từ branch retrieval hay bị cắt lúc fusion trước khi tăng budgets. Thử top-2 fallback hoặc bổ sung passage trong parent chỉ khi có bằng chứng mất đoạn ở fallback; giữ cùng tổng 200 pairs để không lẫn gain từ tăng compute. `parent_first` tăng breadth nhưng nếu gần 200 parents sẽ không còn ngân sách cho nhiều đoạn/parent; phải đo đánh đổi breadth/depth.

Không đưa historical gold IDs hoặc old matched text vào runtime retrieval: các nhãn chỉ dùng để đánh giá offline.

## 3. Có trường hợp đoạn gần old positive đã được score nhưng không được xuất

Query **754**, parent **182675** được chọn ở document rank 14 sau rerank:

| Source offsets | Rerank score | Vai trò |
|---|---:|---|
| 2433–3914 | 1,8046875 | Được chọn bởi cap một chunk/document |
| 3730–5138 | 1,4189453125 | Chứa nguyên old matched chunk sau chuẩn hóa, đã score nhưng không xuất |
| 1219–2627 | 0,40673828125 | Đã score, không xuất |

Đây là ví dụ selection theo relevance score không bảo đảm giữ lại text từng đạt scorer. Nó không chứng minh tất cả reranker scores sai hoặc expanded text chắc chắn tăng F2.

Trên 45 anchors, số old matched predictions được chứa nguyên sau chuẩn hóa:

| Tầng | Số anchors |
|---|---:|
| Trong source đầy đủ | 37 |
| Trong ít nhất một registered passage của parent | 29 |
| Trong candidate evidence sau fusion/fallback | 26 |
| Trong evidence đã rerank | 21 |
| Trong chunks A0 xuất ra | 19 |

Không chứa nguyên old prediction không đồng nghĩa fail LCS 40%. Old prediction không phải gold reference; thay extraction/chia đoạn có thể vẫn match gold.

**Hướng sửa:** tách vấn đề đúng parent và đúng span trong pilot. Kiểm đoạn đứng thứ hai/ba, biên passage và source context của các trường hợp cụ thể trước khi thay model hoặc fine-tune. Không tính LCS giữa query với chunk để thay scorer: scorer so chunk với gold của cùng query/doc_id.

## 4. CPU replay chưa ủng hộ bật thẳng hai chunks/document

Các arm dưới dùng chung A0 candidates và scores; không giả làm inference parent_first. Số liệu là tổng 20 queries.

| Arm | Documents | Chunks | Parents có exported chunk | Old anchors chứa nguyên trong output |
|---|---:|---:|---:|---:|
| A0 hiện tại | 1.200 | 1.200 | 1.200 | 19 |
| Independent d120, c1 | 2.398 | 1.200 | 1.200 | 19 |
| Independent d200, c1 | 2.990 | 1.200 | 1.200 | 19 |
| Independent d200, c2, document order | 2.990 | 1.200 | 815 | 15 |
| Diagnostic global score, c2 | 2.990 | 1.200 | 963 | 17 |

Fix max_docs giúp giữ thêm 3 historical positive parents trong relevant_docs, nhưng không đổi exported chunks ở arm c1 trên cùng trace. D200 chỉ đạt 118–193 docs/query vì A0 mới score được từng ấy parents; không thể replay thành 200 parents có score mà không score thêm.

Hai chunks/document cùng tổng cap 60 lấy mất suất của parents khác. Global selection c2 có ít mất breadth hơn document-order c2 nhưng cũng chưa cho tín hiệu tốt hơn theo exact anchor containment. Chưa có căn cứ chọn một trong hai c2 làm mặc định.

## 5. Context expansion là ứng viên thử nhỏ, chưa là bản sửa đã nghiệm thu

CPU diagnostic cố định lấy source slice bao seed và passages liền trước/liền sau, chỉ cho parents A0 đã chọn. Exact old-anchor containment tăng từ **19 lên 23** trên 45 anchors; thêm bốn trường hợp `(53,487926)`, `(242,538821)`, `(754,182675)`, `(1144,183582)`. Đây là unscored context diagnostic, không phải submission mới hay F2.

Production validator hiện yêu cầu `passage_id`, text và offsets khớp nguyên registered passage. Không thể giữ passage_id cũ rồi thay text bằng expanded text. Nếu pilot span cho tín hiệu tốt, cần contract riêng cho derived source span: parent, exact offsets, source hash/provenance, seed passage IDs; validator phải kiểm exact contiguous source slice. Không cần re-embed corpus để thử context từ source sẵn có.

Phân biệt hai cách dùng score: nếu reranker nhận expanded text thì input thay đổi, phải score input mới; nếu chỉ mở rộng lúc export sau khi chọn seed, score cũ chỉ thuộc seed, không được gắn nhãn expanded text đã rerank. Cách thứ hai vẫn cần contract/grounding đúng và đánh giá chất lượng. Kiểm caps, duplicate handling, giới hạn độ dài BTC khi có đầy đủ schema, và ZIP size trước khi quyết định export.

**193/4.000 pairs hiện vượt 512 tokens trước truncation**. Không mặc định kéo dài reranker input là tốt hơn; phải đo truncation, coverage và latency trên arm nhỏ. Context không khôi phục parent đã mất từ retrieval/allocation.

## Trình tự khắc phục đề xuất

1. **Đã hoàn tất trong lượt này:** CPU audit, A0 parity, cache/grounding audit và selection replay. Giữ nguyên full index/dense và production code.
2. **Pilot allocation có mục tiêu:** queries `53,130,380,754,771,1144` đã có full-corpus traces, bao phủ toàn bộ 7 unscored positives, 4 absent-fused positives và ví dụ chọn sai span được phát hiện. Dùng A0 làm đối chứng và independent/parent_first c1 đã có; có thể thêm controls `46,242`. Đây là tập chẩn đoán, không phải tập đo macro F2 đại diện.
3. **Lưu đủ trace và dùng cache:** runner pilot phải ghi raw branch hits, pre-rerank evidence order, exact inputs và pair decisions. Trace A0 hiện thiếu ordering nên không được sửa label thành parent_first để giả parity. Nếu phải chạy lại retrieval để lấy ordering thì chạy tập nhỏ trên index hiện có; không rebuild. Chỉ gọi model cho exact missing pairs, khóa một cache contract và ghi hit/miss counts.
4. **Sửa đúng tầng theo kết quả:** 4 parents absent-fused cần branch diagnosis; fallback nghèo cần thử passage alternatives; span đã score nhưng bị chọn sai cần so seed/context hoặc allocation chunk. Chưa tăng đồng thời mọi budgets, đổi model hay bật c2 mặc định.
5. **Chốt policy bằng scorer hợp lệ:** nếu chỉ có feedback một phần thì báo known-positive survival, unjudged counts, source coverage và latency. Không gán unjudged là negative để tính F2. Muốn xác nhận tăng DOCS_F2/CHUNKS_F2 cần full gold/tokenizer/scorer hoặc feedback đúng phase và đúng ZIP hash. Sau đó mới quyết định full 1.200 queries.

## Giới hạn và kiểm chứng

Tokenizer BTC và gold reference text đầy đủ chưa có trong artifacts được kiểm tra. Whitespace LCS trong CSV này lấy **độ dài old matched prediction** làm mẫu số, chỉ để chẩn đoán bảo toàn text; không phải công thức scorer token-LCS/gold-length. Không dùng threshold 0,4 của BTC lên proxy này để gán nhãn đúng/sai. Các phương án context mới chưa được chấm chính thức.

Kiểm chứng đã thực hiện: 20/20 A0 selection parity, 4.000/4.000 cached scores khớp inputs, 6.419 candidate spans khớp nguồn, stage counts đối soát đủ 45 anchors, production/input SHA giữ nguyên; thuật toán bitset LCS so với DP độc lập trên 300 cặp token ngẫu nhiên cho kết quả 300/300 khớp. Không thay runtime nên không chạy lại GPU hoặc toàn bộ integration suite.

Trạng thái tại thời điểm audit ban đầu: remediation pilot chưa chạy. Phần cập nhật bên dưới ghi kết quả thực thi sau đó; F2 mới vẫn UNMEASURED.

## Cập nhật thực thi pilot parent-first

Pilot đã hoàn tất sau phần audit trên đúng full index đã nghiệm thu, với 6 query IDs `53, 130, 380, 754, 771, 1144`. Runner ghi raw hits của document BM25, passage BM25 và dense; fused items, fallback decisions, exact 200 rerank pairs, cache hit/miss, ranked candidates và selection. Artifacts nằm tại `outputs/v4/parent_first_evidence_pilot_20261021_v2/`. Namespace không có hậu tố v2 là hai lần launcher dừng trước khi chấm pair do lỗi khởi tạo lock/wrapper trace; không dùng làm kết quả.

Kết quả vận hành:

- 6/6 queries hoàn tất trên GPU 3 đã khóa; tổng 1.200 pairs.
- Tái sử dụng 897 scores A0 và chấm mới đúng 303 pairs; cache cuối có 4.303 rows, không chấm lại 897 pairs đã có.
- Tổng thời gian truy vấn 1.878,66 giây; query 130 chiếm 836,44 giây. Nút thắt chính vẫn là retrieval full index trên CPU.
- Mỗi query có 200 distinct reranked parents, chọn 200 documents và 60 chunks theo config independent d200/parent-first. Đây là pilot chẩn đoán, không phải lựa chọn cuối của document cap.

So sánh trên 37 cặp query/document từng có chunk baseline được BTC chấm đúng:

| Arm trên cùng 6 queries | Documents | Chunks | Positive parents còn trong output | Old matched chunks còn nguyên trong output |
|---|---:|---:|---:|---:|
| A0 with_chunks/document_order | 360 | 360 | 24/37 | 15/37 |
| Independent trên scores A0 | 897 | 360 | 26/37 | 15/37 |
| Independent d200/parent_first | 1.200 | 360 | **33/37** | **19/37** |

Như vậy fix max_docs riêng tăng 2 positive parents; đổi allocation sang parent_first tăng thêm 7 parents và 4 exact old-anchor chunks trên tập này. Đây là known-positive survival, không phải official precision/recall/F2. Các candidates mới chưa được gán negative.

Replay document cap trên trace parent-first cho thấy số positive parents giữ được: cap60 `26/37`, cap80 `28/37`, cap100 `32/37`, cap120 `32/37`, cap160 `33/37`, cap200 `33/37`. D120 xuất ít documents hơn nhưng chưa biết các documents cắt đi có đúng hay không ngoài 37 historical positives này. **Giữ d120 và d200 làm hai arm để chấm**, chưa có bằng chứng precision hoặc F2 cho việc chọn d120 tốt hơn d200. Hai arm có thể replay cùng scored trace trên CPU; không cần rerank riêng chỉ vì đổi max_docs.

### Kết luận cho fusion

Bốn positive parents không có trong fused400 đã được truy về raw branches:

- `(130,2796009)` và `(771,563024)` có trong dense top200 nhưng bị cắt ở item-level fusion400.
- `(130,2761519)` và `(754,1644439)` không có trong top limits của cả ba nhánh.

Phép phân tích fusion ban đầu cộng tất cả passage scores, khác runtime dùng `max(document score) + max(passage score)`. Đã sửa helper và kiểm đúng parent ordering của cả 6 baseline traces. Artifact tính lại là `remediation_analysis_v2.json`; artifact `remediation_analysis.json` cũ được giữ để truy vết và không dùng làm kết luận.

CPU replay đúng aggregation của runtime cho thấy fusion800 đưa hai dense parents trở lại pool, nhưng chúng chỉ đứng parent rank 374 và 250; có 28/37 positives nằm trong top200 parent order của fusion800, so với 33/37 ở fusion400. Đây là diagnostic thứ tự parent trước rerank, chưa phải inference hoặc F2 của fusion800. Vì vậy **giữ fused_candidate_limit=400 cho cấu hình đang dùng**, không tăng lên 800 chỉ vì thấy pool có thêm parents. Thử cách phân bổ candidates giữa các nhánh trong arm riêng nếu muốn cứu hai dense parents; hai parents vắng khỏi cả ba nhánh cần chẩn đoán candidate retrieval trước.

### Kết luận cho context

Mở rộng source slice qua passage liền trước + seed + liền sau làm exact old-anchor containment tăng `19 → 21/37`, khôi phục `(53,487926)` và `(1144,183582)`. Cả hai old anchors đều vượt ranh giới passage; không có registered passage đơn lẻ chứa nguyên anchor.

Không bật context expansion vào production trong pilot này. Runtime hiện chỉ cho phép exact registered passages; dùng derived span cần schema/provenance/validator riêng. Ngoài ra chưa có tokenizer/gold/scorer BTC để đo gain và duplicate effect. Đây là arm tiếp theo nếu cần cải thiện CHUNKS_F2 sau khi có phép chấm phù hợp.

Kết luận triển khai hiện tại: **giữ fusion400; dùng parent_first + independent; so d120/d200 bằng phép chấm phù hợp; giữ chunks_per_doc=1 và max_chunks=60; context expansion tiếp tục ở trạng thái experimental.** Full 1.200 inference và F2 vẫn chưa chạy.

## Cập nhật triển khai derived adjacent context

Đã triển khai arm context thử nghiệm sau selection. Mỗi chunk xuất ra được dựng thành một source slice liên tục từ seed passage đã rerank và tối đa một passage liền trước/liền sau (`context_window=1`, tối đa 4.096 ký tự). Document ranking, `document_evidence`, 200 rerank pairs và score của seed không thay đổi. Derived chunk mang contract `adjacent_source_span_v1`, source hash, exact offsets, seed passage, danh sách passages tham gia và bản sao seed evidence. Trạng thái score là `derived_from_reranked_seed`; không tuyên bố expanded text đã được rerank.

Validator không tin text/offset do runner ghi. Nó lấy lại seed và toàn bộ passages của parent từ corpus store, dựng lại deterministic span và yêu cầu object khớp chính xác trước khi serializer chỉ xuất `doc_id` và `chunk_text`. Các sửa giả mạo text, offset, derived ID, seed ID, member IDs, status hoặc score đều bị từ chối. Mode mặc định `chunk_context=passage` vẫn giữ output cũ.

Các file chính:

- `src/medrag/evidence/context.py`: dựng derived span và provenance.
- `src/medrag/adapters/validation.py`: xác minh lại derived contract từ corpus.
- `scripts/v4/replay_adjacent_context.py`: replay từ scored parent-first traces, không gọi model.
- `configs/v4_phase2_full_independent_d{120,200}_context_gpu3.yaml`: hai config arm context; control cũ không bị thay đổi.

CPU replay trên cùng 6 pilot queries đã PASS cho cả hai document caps:

| Arm | Documents | Chunks | Derived spans đã validate | Expanded / unchanged | Positive parents | Seed chứa old anchor | Context chứa old anchor |
|---|---:|---:|---:|---:|---:|---:|---:|
| d120 adjacent | 720 | 360 | 360 | 327 / 33 | 32/37 | 19/37 | **21/37** |
| d200 adjacent | 1.200 | 360 | 360 | 327 / 33 | 33/37 | 19/37 | **21/37** |

Hai trường hợp mới được khôi phục vẫn là `(53,487926)` và `(1144,183582)`. Artifacts nằm tại `outputs/v4/adjacent_context_replay_d120_20261021/` và `outputs/v4/adjacent_context_replay_d200_20261021/`. Cả hai ghi `model_calls=0` và `official_f2=UNMEASURED`.

Giới hạn vận hành mới phát hiện: trên 6 câu, median derived chunk dài 2.576 ký tự, lớn nhất 4.084. ZIP chẩn đoán 6 câu khoảng 336–339 KB; ngoại suy tuyến tính lên 1.200 câu khoảng 67–68 MB, vượt `zip_max_bytes=50 MB`. Đây chỉ là ngoại suy vì độ dài và tỷ lệ nén của full set có thể khác, nhưng đủ để **chưa chạy full/export arm 4.096 ký tự**. Trước full run cần sweep context cap và kiểm ZIP dự kiến; cap 2.304 trên cùng pilot cho khoảng 42 MB ngoại suy nhưng chỉ giữ mức `20/37`, còn cap 3.072 giữ `21/37` nhưng khoảng 59 MB. Cần scorer chính thức để chọn đánh đổi coverage/kích thước, không chọn bằng historical-anchor proxy đơn độc.

Regression hiện tại: 75 tests V4 ngoài `test_cli_api_tail.py` pass; riêng file API/CLI này bị treo ở fixture HTTP/subprocess của môi trường và đã dừng sau 60 giây, không phát sinh assertion failure. Nhóm trực tiếp liên quan context, selector, validator, ZIP và disk pipeline có 43 tests pass trước khi thêm disk-context assertion; lần regression cuối bao gồm assertion đó trong 75 tests. Full 1.200 inference và official F2 vẫn chưa chạy.

## Cập nhật P0: replay đường đi từng positive

Replay chi tiết 37 historical positives trên 6 pilot queries đã hoàn tất, không gọi model. A0 có 7 fused positive parents chưa được rerank; parent-first đưa số này về 0. Nút thắt mới là 7 positive parents đã score và chọn document nhưng bị cắt khỏi 60 chunk slots. Oracle ceiling trên cùng fused400 là `21 → 26/37` nếu chỉ đổi chunk allocation, và `21 → 28/37` nếu đồng thời cho phép đổi candidate passage nhận một trong 200 rerank slots. Ba trường hợp tiếp theo cần retrieval/fusion; sáu trường hợp không thể exact-recover từ registered source.

Báo cáo đầy đủ: `docs/v4_p0_positive_path_replay_20261021_vi.md`. Artifacts chuẩn: `outputs/v4/p0_positive_path_replay_20261021_v3/`.

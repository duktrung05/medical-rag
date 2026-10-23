# A0 Evidence Loss Audit — Lightweight Replay Only

**Ngày audit:** 2026-10-22  
**Deadline contest:** 2026-10-30  
**Kết luận vận hành:** giữ nguyên A0. Không promote selection policy nào từ audit này.

## 1. Kết luận ngắn

Audit đã truy vết đủ **7/7 historical-positive parents đã được rerank nhưng không xuất hiện trong 60 chunks** trên sáu query pilot khóa (`53, 130, 380, 754, 771, 1144`). Không có retrieval call, model call hoặc GPU work mới.

- Cả 7 parent đều vào frozen fused400, nhận đúng một parent-first pair trong 200 pairs, có reranker score, và được giữ trong `max_docs=200`.
- Cả 7 mất chunk tại cùng một stage cơ học: **chunk allocation cap 60**, vì parent rank sau rerank là `70–126`.
- Không ca nào trong 7 bị loại ở document selection d200. Việc gọi cả 7 là “document-selection loss” sẽ trộn hai tầng độc lập.
- Chỉ **5/7** có scored seed/context chứa exact historical anchor và có thể được cứu về mặt exact-containment chỉ bằng đổi slot top-60.
- Hai ca còn lại, `(130,546076)` và `(771,494605)`, là **source mismatch**: exact historical anchor không còn trong registered current source, passage hoặc adjacent context. Cấp slot cho chúng vẫn không phục hồi exact anchor.
- Trong 5 ca recoverable, bằng chứng semantic chỉ ủng hộ rõ nhất một ca reranker under-ranking (`130/553119`) và ở mức khả dĩ một ca (`771/484565`). Không được suy ra rằng mọi positive lịch sử đứng sau rank 60 đều là lỗi reranker.
- Oracle ceiling cho cùng fused pool và cùng scored seeds là `21 -> 26/37` exact anchors nếu biết trước label để chuyển slot. Đây là **khả năng lý thuyết +5**, không phải một policy online hợp lệ.
- Các policy replay không dùng label đã thử trước đây không cứu được ca exact-recoverable nào và đều không vượt A0. Vì vậy khả năng cải thiện có tồn tại về mặt allocation ceiling, nhưng **chưa có selection signal đủ tin cậy**.

## 2. Phạm vi, contract và bảo vệ A0

Audit chỉ đọc:

- frozen config `configs/v4_submission_p2_a0.yaml`, SHA-256 `bd944d1e986ed6bf5e25767454168b14bb82684d802ac7645d3f95b671f4cc03`;
- existing branch/query traces tại `outputs/v4/parent_first_evidence_pilot_20261021_v2/`;
- P0 replay chuẩn tại `outputs/v4/p0_positive_path_replay_20261021_v3/`;
- cached scored-parent pool và policy replays tại `outputs/v4/p1_chunk_allocation_20261021_v2/`;
- score diagnostics tại `outputs/v4/p1_5_score_diagnostics_20261021_v2/`;
- P2 cached passage replay chỉ để đối chiếu giới hạn của passage choice.

Không chạy full-index retrieval, không load dense encoder/reranker, không dùng GPU, không attach/inspect/interrupt tmux `p5_cpu`, và không sửa config, checkpoint, baseline hoặc output A0. File duy nhất được tạo bởi task này là báo cáo này.

P0 ghi `model_calls=0`, official F2 `UNMEASURED`; P1 ghi `model_calls=0`, `retrieval_calls=0`, `a0_p0_parity=true`; P1.5 ghi `model_calls=0`, `retrieval_calls=0`. Toàn bộ input hashes do P0 và P1.5 khai báo đã được kiểm lại và đều khớp.

Phạm vi bằng chứng chỉ là 37 historical anchors trên sáu query full-index pilot. Historical positives là diagnostic labels không đầy đủ, không phải official gold và không đủ để ước lượng generalization trên 1.200 queries.

## 3. Định nghĩa rank để tránh nhập nhằng

Mỗi parent trong frozen A0 parent-first nhận một passage ở vòng đầu:

- **Passage branch rank**: rank của chính scored passage trong raw branch trace. Nếu parent chỉ đến từ document BM25, passage là deterministic fallback và không có passage-level branch rank.
- **Pair position**: vị trí passage được nhận trong frozen 200 rerank pairs. Với vòng parent-first một-passage-per-parent, vị trí này bằng **pre-rerank parent rank**.
- **Post-rerank parent rank**: rank của parent sau khi sắp theo score của scored passage; đây là rank quyết định 60 chunk slots.
- **Cutoff gap**: `score(rank 60) - score(parent)`. Gap dương nghĩa là parent ở dưới cutoff.

Như vậy “passage đã rerank” không đồng nghĩa “passage được xuất”. Document selection giữ tới 200 parents, sau đó chunk allocation chỉ cấp 60 slots.

## 4. Truy vết chính xác 7 positives

| Query / doc | Scored passage | Passage branch rank | Pair position / pre-parent rank | Post-parent rank | Reranker score | Rank-60 score / gap | Exact source/context | Loss và nguyên nhân |
|---|---|---:|---:|---:|---:|---:|---|---|
| `53 / 756661` | `p2_84221813f89f02c6b1eb0e0b07254c8268220b10397d2441b93f8434d81aa8a1` | dense `105`; không BM25 | `176 / 176` | `99` | `1.8134765625` | `2.736328125 / 0.9228515625` | source: yes; registered passage: no; adjacent context: yes | Mất tại chunk cap: thiếu 39 slots. Allocation-only exact-recoverable. Passage liên quan ở mức triệu chứng rộng; chỉ là under-ranking borderline, không phải lỗi reranker đã xác nhận. |
| `130 / 546076` | `p2_b55ddfb53a6664e9e2cc18214fdf2173da757aad9cf3251852e8fd407f79a1ef` | passage BM25 `7`; dense `160` | `36 / 36` | `99` | `-0.1116943359375` | `1.9267578125 / 2.0384521484375` | source: **no**; passage: no; context: no | Bị cap 60 về mặt output, nhưng nguyên nhân không thể exact-recover là **source mismatch**. Chuyển slot không thể tái tạo historical anchor. |
| `130 / 553119` | `p2_9fa5fce1c04c89d31446bf7faab5847707b14c11214cf881cf3a7e8b8cfc2508` | dense `96`; không BM25 | `197 / 197` | `75` | `1.0595703125` | `1.9267578125 / 0.8671875` | source/passage/context: yes | Mất tại chunk cap: thiếu 15 slots. Allocation-only exact-recoverable. Đây là **reranker under-ranking mạnh nhất**: passage giải thích trực tiếp tắc tuyến lệ, chảy nước mắt và trẻ sơ sinh, dù retrieval/fusion support yếu. |
| `130 / 2968260` | `p2_904b3e4f8c265c05e21cd9c1ee2a643d92dacdbbeab0f1ad3acea97b60d710da` | dense `64`; không BM25 | `149 / 149` | `126` | `-1.9013671875` | `1.9267578125 / 3.828125` | source/passage/context: yes | Mất tại chunk cap: thiếu 66 slots. Exact-recoverable theo text, nhưng passage tiếng Trung khác đối tượng và câu trả lời phủ định bệnh được hỏi. Pair dài 538 tokens nhưng nội dung lâm sàng chính vẫn trước cutoff 512. Score thấp có thể là đúng; không gán lỗi reranker. |
| `754 / 2713777` | `p2_704e384d8369d954b51461c85b84612c7744a2a5d35f9ca5791b962b18a5a470` | dense `70`; không BM25 | `114 / 114` | `70` | `-1.544921875` | `-1.0322265625 / 0.5126953125` | source/passage/context: yes | Mất tại chunk cap: thiếu 10 slots. Exact-recoverable theo text, nhưng passage nói CA125 `40.41`, query hỏi `64.80`, và không có diễn giải/bước tiếp theo. Không đủ bằng chứng under-ranking. |
| `771 / 484565` | `p2_e37d221b3f3ed2eb40f0f5a3cef52eacb98d60a5f70bba876510ebbf408a3cc6` | passage fallback từ document BM25 parent rank `77`; không dense/passage-BM25 hit | `137 / 137` | `100` | `3.3046875` | `4.0703125 / 0.765625` | source: yes; registered passage: no; adjacent context: yes | Mất tại chunk cap: thiếu 40 slots. Allocation-only exact-recoverable. Passage nói trực tiếp về xuất tinh sớm và điều trị; **under-ranking khả dĩ**, nhưng không có dense support và vẫn dưới cutoff 0.766. |
| `771 / 494605` | `p2_98de89ca996547baa10ce1df35b7eacf34d3d435aa1d0a1c3d128c5e81c49758` | scored passage: passage BM25 `155`, dense `187`; parent best: document BM25 `87`, dense `59` | `74 / 74` | `84` | `3.619140625` | `4.0703125 / 0.451171875` | source: **no**; passage: no; context: no | Bị cap 60 về mặt output, nhưng exact historical evidence không tồn tại trong registered source. Đây là **source mismatch**; chuyển slot không thể exact-recover anchor. |

### Phân loại nguyên nhân

| Loại | Kết quả |
|---|---|
| Reranker under-ranking | Mạnh nhất: `130/553119`; khả dĩ: `771/484565`; borderline: `53/756661`. Không có căn cứ gán nhãn này cho cả 7. |
| Document selection | `0/7` mất ở d200. Cả 7 đều được chọn trong 200 documents. |
| Chunk allocation | `7/7` bị loại cơ học vì post-rerank rank > 60; `5/7` exact-recoverable nếu có policy hợp lệ cấp lại slot. |
| Source mismatch | `2/7`: `130/546076`, `771/494605`. Selection không thể khôi phục exact historical anchor. |

## 5. Khả năng cải thiện mà không đổi 200 rerank pairs

### Oracle ceiling

Giữ nguyên fused400, đúng 200 pairs và đúng scored seeds:

- hiện tại: `21/37` historical anchors exact sau adjacent context;
- oracle đổi 60 slots: tối đa `26/37`, gain `+5`;
- năm ca có thể thêm là `53/756661`, `130/553119`, `130/2968260`, `754/2713777`, `771/484565`.

Oracle dùng historical labels để biết trước slot nào cần đổi, nên không được dùng làm inference policy. Hai source-mismatch cases không nằm trong gain `+5`.

### Replay các policy đã thử

Tất cả arm sau dùng cùng 200 scored parents/query; labels lịch sử chỉ được mở sau selection để tính metric:

| Policy | Exact anchors | Mean LCS proxy | Exact cases mới được cứu trong 5 ca recoverable | Kết luận |
|---|---:|---:|---:|---|
| A0 score order | `21/37` | `0.649946` | baseline | Reference |
| A1 rerank-only | `21/37` | `0.649946` | `0` | Cùng tập chunk với A0 |
| A2 hybrid `0.5/0.3/0.2` | `13/37` | `0.416731` | `0` | Fail |
| A2 hybrid `0.4/0.3/0.3` | `11/37` | `0.344765` | `0` | Fail |
| A2 hybrid `0.6/0.2/0.2` | `15/37` | `0.470525` | `0` | Fail |
| A3 `40 hybrid + 10 dense + 10 coverage` | `13/37` | `0.433730` | `0` | Fail |

Các arm A2/A3 tăng lexical query coverage nhưng làm mất lần lượt 8, 10, 6 và 8 exact anchors đang có trong A0. Kết quả này bác bỏ việc promote ngay weighted fusion hoặc quota dense/coverage. Dense similarity, fused rank, BM25 presence và query coverage hiện không tách an toàn các true-relevant cases khỏi topical, wrong-patient, wrong-number hoặc source-mismatch cases.

Kết luận: **có headroom allocation +5 theo oracle, nhưng chưa có bằng chứng một label-free policy hiện có biến headroom đó thành gain thực.** Official CHUNKS_F2 và independent relevance holdout chưa được đo, nên claim cải thiện vẫn `BLOCKED_BY_EVALUATION`, không phải blocked vì thiếu rerank score.

## 6. Tối đa hai policy đáng thử tiếp

Hai policy dưới đây chỉ là candidates để replay/evaluate trên development rồi independent holdout. Chúng không được thêm vào A0 đang chạy.

### Policy 1 — Conservative residual reserve (`55 + 5`)

- Giữ nguyên 55 parents đầu theo reranker score.
- Dành tối đa 5 slots cho ranks 56–120 theo một score residual được định nghĩa trước từ frozen signals: reranker margin, cross-branch agreement, query-language match, truncation/boilerplate flags và numeric consistency.
- Có hard safety floor theo reranker margin; nếu không candidate nào qua floor thì backfill đúng score order.
- Feature set, weights/floors phải được chọn trên development relevance labels độc lập; không được fit trên 37 historical anchors hoặc bảy IDs này.

Lý do đáng thử: thận trọng hơn A3 `40+10+10`, giữ ít nhất 55/60 A0 slots và trực tiếp nhắm vùng residual. Rủi ro: các ca recoverable trải tới rank 100, và các topical mismatches cũng có retrieval signal tốt; do đó policy có thể vẫn không cứu được gain mà không làm giảm precision.

### Policy 2 — Redundancy-aware safe replacement

- Bắt đầu từ exact A0 top-60.
- Chỉ đánh dấu slot thay thế khi chunk top-60 gần trùng nội dung/source với một chunk đã chọn khác theo deterministic text similarity trên frozen text.
- Thay slot redundant thấp nhất bằng candidate chưa chọn có score gap trong ngưỡng đã khóa và qua language/numeric/source-quality gates; nếu không có candidate đủ điều kiện thì giữ A0.
- Không dùng query/doc IDs, historical-anchor text, anchor containment hoặc historical-positive flags ở inference.

Lý do đáng thử: tìm slot từ redundancy thay vì hạ hàng loạt high-score evidence như A2/A3. Rủi ro: audit này chưa đo prevalence của cross-document redundancy và chưa chứng minh một trong năm recoverable cases sẽ được chọn; policy phải qua development/holdout mới được coi là hữu ích.

## 7. Điều kiện đánh giá và quyết định promote

Không policy nào được promote trước khi có independent holdout evaluation. Gate tối thiểu:

1. Khóa policy/threshold trước khi mở holdout labels.
2. Giữ đúng frozen 200 pair identities và cached scores; không tăng model calls.
3. Xác minh 60 unique parents/query, một chunk/parent, source grounding và A0 replay parity.
4. Báo cả overall/per-query CHUNKS precision, recall, F2; không chỉ anchor containment hoặc mean LCS proxy.
5. Báo regressions theo language, wrong-patient, numeric mismatch, truncation và source mismatch.
6. So sánh với A0 bằng paired bootstrap hoặc confidence interval đã khóa trước; không chọn policy từ holdout.
7. Nếu không có independent relevance/gold labels, trạng thái vẫn là `NO_PROMOTION`.

Historical positives chỉ được dùng để audit sau selection. Tuyệt đối không hardcode bảy query/doc IDs, passage IDs hoặc old-anchor text vào runtime selector.

## 8. Score completeness và blockers

- Rerank score cho 7/7 cases: **đầy đủ** trong existing parent-first traces.
- Missing rerank score cần inference: **0**.
- Full retrieval/model run cần cho audit này: **0**.
- Blocker kỹ thuật: không có.
- Blocker quyết định policy: chưa có independent relevance/CHUNKS-F2 holdout evaluation.

## 9. Artifact provenance

Các artifact kết luận chính:

- `outputs/v4/p0_positive_path_replay_20261021_v3/positive_paths.jsonl`
- `outputs/v4/p0_positive_path_replay_20261021_v3/failure_groups.json`
- `outputs/v4/p0_positive_path_replay_20261021_v3/allocation_ceiling.json`
- `outputs/v4/p0_positive_path_replay_20261021_v3/manifest.json`
- `outputs/v4/p1_chunk_allocation_20261021_v2/scored_parent_pool.jsonl`
- `outputs/v4/p1_chunk_allocation_20261021_v2/arm_comparison.json`
- `outputs/v4/p1_chunk_allocation_20261021_v2/validation.json`
- `outputs/v4/p1_5_score_diagnostics_20261021_v2/target_parent_diagnostics.csv`
- `outputs/v4/p1_5_score_diagnostics_20261021_v2/diagnostics.json`

P0 manifest status: `P0_POSITIVE_PATH_REPLAY_PASS`. P1 validation status: `PASS`. Không artifact nào ở trên bị sửa trong audit này.

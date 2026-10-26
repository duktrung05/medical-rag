# Review dự án và cấu hình thực tế — 25/10/2026

## Kết luận review

Champion hiện tại vẫn là S2, điểm người dùng báo từ BTC: FINAL 0.2318,
DOCS_F2MACRO 0.2788, CHUNKS_F2MACRO 0.1848. Review độc lập replay đúng
1.200/1.200 document lists và chunk arrays từ frozen refined checkpoints.
ZIP SHA-256 khớp `081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.

Vấn đề lớn có hai phần: ranking/allocation chưa tốt và hệ thống đánh giá local
chưa đo đúng mục tiêu đủ để chọn hướng tối ưu đáng tin cậy. Một số kết luận
R5/R6/R8 trước đây đã được diễn giải mạnh hơn bằng chứng cho phép.

Review này chỉ thêm báo cáo và bằng chứng; chưa sửa code runtime, evaluator,
config, model hoặc submission. Các kết quả cũ được giữ nguyên để truy vết.

## 1. Cấu hình nào thực sự đang được dùng?

| Đường chạy | Cấu hình/entrypoint | Ý nghĩa thực tế |
|---|---|---|
| CLI/API mặc định | `src/medrag/cli.py`, `src/medrag/api.py`, `configs/v4_core.yaml` | Bounded prepared snapshot, dense/reranker tắt; không phải S2 |
| Frozen competition precursor | `scripts/v4/run_p5_2_split.py`, `configs/v4_submission_p2_a0.yaml` | Full eligible release, BGE-M3 + BGE reranker gốc, 200 pairs/query |
| Champion S2 | `scripts/v4/refine_a0_submission.py`, refined manifest và S2 manifest | Supplemental scores + context expansion + adaptive docs, nhiều tham số nằm ngoài YAML |
| R3/R5 | Các namespace/model thử nghiệm | Fine-tuned R3 không nằm trong champion S2 |
| R6/R8 | Scripts audit/experiment riêng | Không phải deployment config; không có submission thắng S2 |

Thông số effective S2:

- Index: `outputs/v4/task4/full_index_gpu3_20261019`.
- Eligible sources: 1.073.525; passages: 2.951.224.
- Encoder: BAAI/bge-m3, revision `5617a9f61b028005a4858fdac845db406aefb181`, max_length 512.
- Reranker: BAAI/bge-reranker-v2-m3 gốc, revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, max_length 512.
- Retrieval budgets: document BM25 400, passage BM25 200, dense passages 200.
- RRF k=60, top-400 là mixed document/passage items trước khi gom parents.
- Precursor parent-first rerank: 200 pairs/query; S2 sau đó score bổ sung toàn bộ
  cached passages: 240.000 original + 153.401 supplemental = 393.401 pairs.
- Chọn ranked top-200 parents, xuất tối đa một chunk/parent, 60 chunks/query.
- Context adjacent window=1, cap 4.096 ký tự; refinement mở rộng tiếp cap 5.500 ký tự.
- Documents: pre-gate -4; đếm score >= -2, target clip vào [80,140], bảo vệ mọi chunk parent.
- Output thực: 140.095 documents, 72.000 chunks; docs/query min 60, mean 116.746, max 140.
- ZIP thực: 91.173.448 bytes; refinement validator cap 100.000.000 bytes.

Đường dữ liệu champion:

```text
eligible index
  → doc BM25 400 + passage BM25 200 + dense 200
  → RRF top-400 mixed items → group parents
  → precursor 200 pairs + supplemental all cached passages
  → rank parents / select top-200
  → allocate first 60 chunk slots / select seed per parent
  → adjacent context 4096 → expansion 5500
  → doc gate -4 → adaptive docs [80,140], protect chunk parents
  → S2 ZIP
```

Điểm gây rối đã xác minh: README vẫn mô tả trạng thái trước full index; core
default khác competition path; YAML competition vẫn ghi cap ZIP 50 MB trong khi
S2 exporter dùng 100 MB; effective S2 policy nằm trong Python, không có một
config manifest duy nhất điều khiển toàn bộ đường chạy. Chạy default CLI/API
không tái tạo được S2 chỉ bằng cách dùng config mặc định.

## 2. Findings theo mức ưu tiên

### F01 — P0: Nhãn R5 không còn gắn đúng chunk đang được đánh giá

Code: `scripts/v4/evaluate_r5_same_input.py:31`, `:45`, `:99`.

Feedback được xác minh trên `(query, old doc, old text hash)`, sau đó rút xuống
`(query, doc) -> bool` và gán cho chunk S2 mới. Hai model thực sự score cùng
input mới, nhưng nhãn vẫn thuộc input cũ.

Đo lại trên 17.707 pairs đã ghép:

- 1.568 positive, chỉ 1 pair có nguyên text cũ.
- 16.139 negative, chỉ 3 pairs có nguyên text cũ.
- 1.261 positive mới còn chứa anchor cũ; 307 positive mới không còn chứa anchor đó.
- 14.571 negative mới chứa đoạn negative cũ. Điều này không đủ để suy ra cả context
  mới là negative: context có thể bổ sung evidence đúng ở vùng khác.

Tác động: AUC/AP tăng trong R5 là measured đối với nhãn chuyển theo parent,
không đủ chứng minh chất lượng evidence ranking tăng như đã diễn giải trước.
307 trường hợp không chứa anchor không tự động là false positives; chúng là
chunk mới chưa có nhãn evidence trực tiếp.

Hướng sửa: giữ nhãn theo exact text identity; changed spans để unjudged trừ khi
được đánh giá lại; tách document relevance, evidence relevance và preservation.

### F02 — P0: Historical matched prediction đang được sử dụng như gold evidence

Code: `scripts/v4/audit_r6_stage_oracle.py` registry; R8 `exact_ids`.

Registry gồm 2.190 outputs cũ từng match BTC và 87 manual evidence labels.
2.190 text này là predictions từng đạt matching threshold; không phải complete
gold evidence texts. Review feedback gốc ghi overlap = token LCS/gold tokens,
threshold 0.4, duplicate threshold 0.8; tokenizer bytes và scorer source không có.

Do đó exact containment 1.323/2.277 là anchor-preservation proxy. Span ngắn có
thể vẫn đáp ứng BTC nhưng không chứa nguyên prediction cũ; span dài có thể giữ
anchor nhưng giảm precision/duplicate behavior. Whitespace LCS/old anchor length
cũng khác denominator và tokenizer của BTC.

R6 source-miss không chứng minh mất gold evidence. Trong 309 labels có chunk
parent nhưng thiếu exact anchor, 266 không còn nguyên anchor trong current source.
Không thể gán cả 266 cho extractor lấy sai passage.

Hướng sửa: báo riêng ba thước đo: preservation trên old outputs, manual evidence
match, và official scorer khi có. Không gọi preservation recall là official recall.

### F03 — P1: R6 mapping normalized text → source offsets sai

Code: `scripts/v4/audit_r6_stage_oracle.py:72`.

Hàm normalize NFKC, HTML-unescape và lowercase trước rồi lưu index của chuỗi đã
biến đổi như index source gốc. NFKC/HTML có thể làm đổi số ký tự.

Kiểm tra trên artifact thực: 1.933 labels được báo contained trong source;
320 trường hợp `normalize(source[start:end]) != normalize(evidence)`.

Ví dụ tái hiện: source `A &amp; B\n  Điều trị VIÊM dạ dày.`, needle `điều trị viêm`:
hàm trả (8,21), slice là `B\n  Điều trị `, không phải evidence.

Tác động: source offsets của registry và token visibility dựa trên offsets này
cần sửa/re-audit. Count exact containment dựa trên string search vẫn dùng được;
S2 chunks/source offsets riêng không bị chứng minh sai bởi lỗi audit này.

Hướng sửa: giữ mapping về original character spans trong từng phép normalization;
assert normalized slice bằng evidence trước khi ghi offset; thêm test Unicode
ellipsis/ligature, HTML entity và combining marks.

### F04 — P1: R8 scoring khiến A1 gần như không thể chọn span ngắn hơn

Code: `scripts/v4/run_r8_evidence_extraction.py:86`, `:102`, `choose_a1`.

70% score là query-term coverage; density chia cho `min(text_tokens,80)`.
Khi cả đoạn dài và span ngắn đều >=80 lexical tokens, đoạn dài không chịu thêm
length penalty, thường giữ ít nhất số query terms của span nằm trong nó.

Đo trên toàn candidate parquet:

- 150.552 span mới nằm trong baseline chunk.
- Chỉ 5 có lexical score cao hơn baseline, không span nào đạt margin +0.20.
- A1 locked cuối chỉ thay 22 chunks trên 526 labeled queries: 17 development, 5 validation.

Kết quả A1 +0 validation phản ánh policy gần như không thay output, không phải
chứng minh extraction không thể cải thiện.

Hướng sửa: chấm spans trong cùng representation, score relevant evidence thay vì
độ phủ từ query, calibration trên development; baseline là fallback với loss budget.

### F05 — P1: Candidate pool R8 đánh đổi mất evidence trước khi selector hoạt động

Review oracle độc lập trên actual R8 parquet:

| Cohort | A0 exact | Short-span label union | Short spans có thể gain | A0 hits không span ngắn nào giữ được |
|---|---:|---:|---:|---:|
| All | 1323 | 1309 | 240 | 254 |
| Development | 917 | 891 | 161 | 187 |
| Validation | 406 | 418 | 79 | 67 |

Label union không đảm bảo một span/parent. Oracle chọn thật một candidate/parent
cho R8 đạt 1.308 với short-only, 1.561 với baseline+short. Trên validation,
baseline+short đạt 485, tức còn +79 so với 406. Đây là diagnostic ceiling chưa
dùng labels để đổi policy/output.

Ép mọi exported chunk <=512 đã gộp scoring input budget với export text size.
R6 1.636 oracle sử dụng expanded contexts dài hơn, vì vậy hai candidate domains
không tương đương. Trên 2.189 historical labels có source, chỉ 55 anchor inputs
tự vượt 512; mất 254 A0 hits không thể giải thích chỉ bằng anchor quá dài.

Hướng sửa: tách short scoring windows khỏi source-grounded export contexts,
đo candidate recall trước policy search, giữ baseline candidate và kiểm candidate
pool không tự phá preservation khi mục tiêu là conservative challenger.

### F06 — P1: Gate development chưa fail-closed; validation/loss accounting còn yếu

Code: R8 `policy_score`, `select_dev:549`, `evaluate:662`.

- Cả 6 A1 policies và 12 A2 policies không đạt development eligibility.
- Code vẫn `max(...)` và khóa policy tốt nhất trong nhóm đã bị loại thay vì `selected_policy=null`.
- `documents_unchanged` và `source_grounding_valid` trong final gate là literal
  `True`, không phải kết quả runtime validator dùng lại trong evaluate.
- Validation tính cả A2 grid rồi chọn challenger A1/A2 theo kết quả validation.
  Muốn dùng validation như một confirmatory gate cần khóa challenger trước.
- Query-loss đếm net exact count giảm; query gain/loss cùng lúc có thể bị che
  nếu net count không đổi. IDs có được track nhưng gate chưa dùng gross-loss queries.

Hiện cuối cùng NO_GO đúng, không có ZIP sai. Lỗ hổng này có thể cho quy trình
thử nghiệm tiếp tục dù development đã bác bỏ. Cần `DEVELOPMENT_NO_GO` và gate
từ validator thực; khóa cả treatment trước validation, ghi gross gain/loss riêng.

### F07 — P1: Chưa có split chung giữa các vòng

R3 dùng `diagnose_a0_quality.partition`; R8 dùng hash seed/split khác.
Đo trực tiếp: 102/167 R8 validation queries có trong R3 train triples.

R8-A hiện dùng lexical signals nên không tự động bị model leakage từ R3.
Nhưng nếu vòng sau dùng R3 trên chính R8 validation, các query đó không còn
held-out đối với model. Historical labels đã được xem qua nhiều vòng nên cũng
không phải pristine holdout.

Hướng sửa: một query split registry xuyên suốt data prep/model/policy/scorer;
record train/eval query membership và fail khi overlap.

### F08 — P1: R6 raw branch attribution có false absence và oracle hơi quá lạc quan

Code: R6 `:456`, `oracle_allocate:283`.

`candidate.ranks` chỉ được gom từ các items sau fused top-400. Có rank chứng
minh branch presence; không có rank không chứng minh raw branch absence, vì
item của branch đó có thể bị fusion cắt trong khi parent vào nhờ branch khác.
Các raw branch absent/conditional retention ngoài explicit pilot phải là UNKNOWN.

Oracle đếm label Boolean theo parent có thể cộng các labels chỉ xuất hiện ở hai
context khác nhau. Đo lại same S2 pool: label union 1.636 nhưng one-context/parent
feasible tối đa 1.635; conflict tại query 672/doc 504074. Sai lệch nhỏ nhưng cho
thấy oracle chưa thực thi đúng ràng buộc. Các O2/O3 parent counts cũng cần được
gọi là parent ceilings, không phải evidence improvements.

Raw full trace thiếu nên chưa định lượng được toàn bộ RAW_PARENT_MISS vs FUSION_DROP.
Không thể suy từ 13.587 unmatched rows rằng retrieval bỏ 13.587 positives: tập đó
gồm cả historical negatives và đầu ra cũ không có nhãn relevance mới.

### F09 — P1: Chunk allocator production vẫn dựa trên document ranking

Code: `src/medrag/evidence/selector.py:141`, `:147`; pipeline parent score=max seed logit.

Selector chạy documents theo thứ tự và cấp slots đến 60; không có evidence
quality allocator riêng được champion xác nhận. S2 adaptive docs thực hiện sau
khi chunk parents đã được chọn, nên S1/S2 chỉ cải thiện docs; CHUNKS_F2 giống hệt.

Đã quan sát: 324 known-anchor labels có parent trong S2 docs nhưng không có chunk;
278 có context hiện hữu chứa anchor. Đây là cơ hội có bằng chứng, nhưng phải
chọn được chúng bằng tín hiệu inference, không dùng gold labels lúc chọn.

Hướng sửa: parent evidence score được xác thực trên evidence labels; một arm
chỉ đổi allocation và giữ context cũ, một arm chỉ đổi context/seed. Thử lexical
swap thất bại chưa bác bỏ semantic/evidence-aware allocation.

### F10 — P2: Runtime/reproducibility và corpus coverage còn thiếu một chuẩn chung

- Effective S2 cấu hình rải ở YAML, Python, manifests và artifact paths hardcode.
- README/core CLI/API chưa phản ánh champion competition workflow.
- Các implementation gần đây còn untracked; cần checkpoint/commit để có code revision
  tái lập, đồng thời giữ artifacts/models lớn ngoài git theo manifest.
- Config 50 MB và exporter 100 MB mâu thuẫn; frozen hash runner không nên sửa trực
  tiếp, cần release config mới rõ version.
- Full release chỉ gồm 1.073.525/1.220.598 official URLs. 147.067 chưa Phase1 PASS
  và 6 semantic exclusions không được index. Chưa biết số gold bị ảnh hưởng;
  không được gọi đây là index đủ mọi official URL hoặc quy hết cho lỗi retrieval.

## 3. Những kết quả nào còn đáng tin?

| Kết quả | Cách dùng đúng sau review |
|---|---|
| S2 official metrics do user cung cấp | Mốc champion; association qua local receipt, không independently fetched |
| S2 replay/hash/cardinality | PASS độc lập 1200 queries, 140095 docs, 72000 chunks |
| R6 exact containment 1323 | Historical/manual anchor preservation proxy, không official evidence recall |
| R6 document parent present | Known-positive parent survival; sample selection bias rất lớn |
| R6 offsets/token visibility | Phải sửa mapping và kiểm lại trước khi dùng attribution chính xác |
| R5 same-input AUC/AP gain | Gain với inherited-parent labels; chưa chứng minh evidence ranking gain |
| R8 NO_GO | Đúng với extractor/lexical allocator cụ thể đã chạy |
| R8 NO_GO → chuyển hẳn R7 | Không suy ra được từ kết quả này |
| 34 tests PASS | Structural/regression checks hiện có; không phủ label validity/normalization semantics |

## 4. Vì sao official score vẫn thấp?

S2 docs precision/recall = 0.1946/0.3695; chunks = 0.2050/0.2108.
Both false positives và thiếu evidence còn lớn. Known-positive parent survival
local ~86% không mâu thuẫn với official recall ~37%: local registry chỉ gồm
những outputs từng tìm đúng từ một pipeline cũ và manual pool nhỏ, bỏ qua nhiều
gold chưa từng được tìm thấy. Đây không phải random complete gold sample.

Reranker score, AUC và preservation không trực tiếp tối ưu macro F2 của scorer.
Model nhìn short passage rồi output long expanded context; length/domain/label
alignment đều thay đổi. Trong khi đó các vòng thử lặp trên cùng labels, đổi split
và nhiều gate không cùng semantics nên khó biết một thay đổi thật sự tốt hay chỉ
tốt hơn trên proxy. Những lỗi này giải thích vì sao đầu tư compute chưa chuyển
thành leaderboard gain tương ứng; không thể phân bổ chính xác phần trăm official
loss cho từng stage khi thiếu current gold/scorer/per-query feedback.

Mục tiêu 0.4 yêu cầu DOCS_F2+CHUNKS_F2=0.8, so với 0.4636 hiện tại. Đây là tăng
lớn ở cả hai tầng; chưa có bằng chứng cho phép dự báo một fix nhỏ sẽ đạt mốc đó.

## 5. Một đường thực nghiệm rõ ràng từ đây

1. **Sửa lớp đo lường trước.** Exact-identity labels, preservation/manual/official
   metrics riêng, normalization offset mapping, feasible oracle, split chung,
   fail-closed development gate. Điều kiện xong: các lỗi F01/F03/F06 có tests
   bắt được; re-audit không sửa frozen baseline; báo cáo nào superseded ghi rõ.
2. **Chuẩn hóa champion release.** Một effective config S2, một command replay
   từ frozen artifacts khớp 1200/1200, manifest model/code/config/data, registry
   experiment status `champion/challenger/diagnostic/rejected` và score receipts.
   Default demo/production và competition entrypoints được ghi riêng rõ ràng.
3. **Một pilot evidence đúng nghĩa.** Giữ docs S2, baseline context fallback;
   score short windows, export contexts source-grounded phù hợp scorer, tách
   allocation-only và context-only. Candidate recall được đo trước policy,
   semantic inference chỉ chạy khi split không overlap model train.
4. **R7 targeted khi cần.** Lưu raw branch hit lists và pre/post-fusion identities
   cho nhóm query thiếu verified relevant parents; không dùng thiếu trace làm
   retrieval miss. Chọn full R7 hoặc R8 tiếp dựa trên measured loss/gain sau các
   bước trên, không dựa vào một lexical NO_GO.

Không triển khai các bước sửa ở review này. Chưa tạo challenger mới.

## 6. Kiểm chứng trong lượt review

- Đọc default CLI/API config, competition config, selector/fusion/refinement,
  R3 training prep, R5 label join, R6 registry/matching/oracle và R8 policy/eval.
- Replay S2 độc lập đầy đủ 1.200 queries; ZIP hash đúng.
- Rejoin toàn bộ historical scored feedback để đo label transfer.
- Check original-source slices cho toàn bộ registry offsets khả dụng.
- Scan actual R8 candidate parquet để tính candidate ceiling và lexical margin.
- Kiểm oracle same-parent trên actual R5 contexts.
- Chạy targeted suites R6/R8/refinement: 34 passed.
- Không full-index retrieval, GPU inference, training hoặc thay baseline.

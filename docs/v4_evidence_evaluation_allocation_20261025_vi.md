# Khóa evaluation và chuẩn bị một pilot allocation — 25/10/2026

S2 vẫn là champion: BTC do người dùng báo FINAL **0.2318**, DOCS_F2 **0.2788**,
CHUNKS_F2 **0.1848**. Chưa có challenger được fit hoặc kết quả relevance mới.
Trạng thái cập nhật: **pilot annotation đã dừng trước khi chấm nhãn**. Xem
[quyết định sau audit 1.200 queries](v4_s2_allocation_signal_decision_20261025_vi.md).

## Giai đoạn A: sửa lớp đo lường

- **F01:** `evaluate_r5_same_input` và consumer `evaluate_r5_window_scoring`
  ghép nhãn theo `(query_id, doc_id, sha256(exact UTF-8 text))`. Không chuyển nhãn
  positive/negative sang context khác. Thiếu hai lớp nhãn trên validation trả
  `INSUFFICIENT_EXACT_IDENTITY_LABELS`, không tính gate từ nhãn kế thừa.
- **F03:** normalization giữ original start/end qua NFKC, combining marks,
  Hangul, HTML entities, lowercase expansion và whitespace. Chỉ ghi offset
  khi normalized source slice khớp normalized evidence. Có tests cho ligature,
  ellipsis, HTML numeric/named references và Unicode combining.
- **F06:** R8 chỉ chọn policy đủ điều kiện development. Nếu không có, ghi
  `DEVELOPMENT_NO_GO`, treatment/policies null và không mở validation. Treatment
  A1/A2 được chọn trên development trước validation; validation chỉ chạy policies
  đã khóa. Query có gross loss vẫn tính loss dù đồng thời có gain bằng số lượng.
  Validator đọc source thật để kiểm offsets, hashes, parent membership và exact
  baseline contexts. Historical preservation không thể qua gate independent relevance.

Audit mới tại `outputs/v4/evidence_evaluation_lock_20261025/`:

| Kiểm tra trên artifact thật | Kết quả |
|---|---:|
| Historical labels khớp nguyên text S2 | 4: 1 positive, 3 negative |
| S2 chunks chưa có exact historical judgment | 71.996 |
| Source offsets cũ có dữ liệu | 1.933 |
| Source slices sai trước sửa | 320 |
| Source slices sai sau sửa | 0 / 1.933 |
| R8 development policies đủ điều kiện | 0; selected policy null |
| Query train R3 / overlap với evaluation mới | 335 / 0 |

Hash S2 giữ nguyên:
`081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.

Registry split chung có 789 development, 210 validation và 201 held-out queries.
Development giữ nguyên boundary A0/R3; tập validation cũ được chia trước thành
validation và held-out. Các query/labels historical đã từng được xem: held-out
ở đây dành cho **judgments mới**, không tuyên bố pristine benchmark holdout.
Split R8 cũ chỉ giữ cho diagnostic lịch sử, không dùng để đánh giá R3 tiếp.

Các báo cáo cũ được giữ nguyên. Audit mới supersede R5 inherited-parent metrics,
R6 source offsets và derived token visibility, R8 development lock/treatment
selection. **Token visibility cũ đã bị vô hiệu cho kết luận mới; chưa tính lại
token visibility.** Không gọi preservation hoặc oracle là official recall/F2.

## Giai đoạn B: pilot allocation đã chuẩn bị, sau đó dừng

Phần dưới là thiết kế lịch sử để truy vết. Không dùng bộ 13.546 contexts làm
việc annotation hiện tại. Audit trên toàn bộ S2 không tìm thấy swap điểm reranker
có margin dương; pilot sẽ đòi lượng nhãn lớn trước khi có tín hiệu khả thi.

Namespace: `outputs/v4/s2_evidence_allocation_20261025/`.

Khóa trước **120 queries**: 60 development, 30 validation, 30 held-out.
Pilot được lấy deterministic từ registry, không chọn bằng kết quả relevance.
Bộ chấm có **13.546 contexts**: development 6.992, validation 3.057,
held-out 3.497. Chưa có nhãn nào được điền tự động.

Retrieval, ranked parents, S2 document lists và candidate context files giữ nguyên.
Một số query có dưới 200 parents trong frozen S2; giữ chính xác danh sách thực tế,
không bổ sung parents để làm đủ 200. Trong pilot có 10 query dưới 200, min 175.
Allocation chỉ chọn trong S2 exported documents thuộc ranked top-200, vì chọn
parent ngoài S2 docs sẽ làm hỏng điều kiện documents unchanged.

Trước khi có judgments, mỗi parent được khóa một context từ pool S2 hiện hữu:
baseline parents giữ đúng chunk text S2; parent chưa có chunk dùng context có
seed logit cao nhất, tie-break bằng context ID. Không chạy extraction hoặc thử
context variants. Việc khóa một context/parent giữ context selection cố định khi
so sánh cơ chế cấp 60 slots.

Challenger duy nhất là CPU logistic allocator với ba features đã khóa:
seed reranker logit, log context characters, log số seed support. Fit duy nhất
trên independent context judgments ở development, 400 steps, step 0.1, L2 0.01;
không grid search, không fine-tune reranker và không GPU inference. Đây là lựa
chọn pilot có thể bác bỏ, chưa phải phương pháp được chứng minh có gain.

Development phải có positive net relevance gain trong loss budget mới tạo
`LOCKED_BEFORE_VALIDATION`. Validation dùng nguyên model; held-out chỉ mở sau
validation GO với cùng model hash. Mỗi stage ghi gross gains/losses và query-level
changes. Loss budget = `floor(0.02 * evaluated queries)`: 1 query ở development,
0 ở validation và held-out. Query mất một relevant context vẫn bị tính loss,
kể cả khi gain một relevant context khác.

Metric là macro số chunk được chấm relevant / 60 trên pool cố định, không phải
official evidence recall hoặc official F2. DOCS giữ nguyên theo thiết kế; không
dự báo CHUNKS_F2/FINAL từ metric này. Chưa tạo submission ZIP; các release/export
validators cần chạy khi có challenger vượt các gate và chuẩn bị submission.

## Handoff chấm độc lập và lệnh chạy

Reviewer dùng `review_development.jsonl` và điền `labels_development.csv`;
không cung cấp `private_*.json` hoặc model scores/ranks/arm identity cho reviewer.
Review items chỉ chứa opaque ID, query và exact context. Đánh giá toàn bộ context
có evidence trả lời query hay không; không copy nhãn parent hoặc historical anchor.

CSV yêu cầu đủ mọi item: `relevant` là `true` hoặc `false`, `reviewer_id` có giá
trị, `judgment_basis` là `direct_context_review`. Unknown, duplicate, foreign ID
và thiếu judgments đều fail closed. Provenance này là attestation của reviewer;
code không tự chứng minh reviewer đã chấm độc lập. Validation/held-out review
files được giữ riêng; không mở nhãn trước khi stage trước đủ điều kiện.

```bash
# Re-audit A vào namespace mới nếu chạy lại, không ghi đè lịch sử.
.venv-phase2-batch/bin/python -m scripts.v4.lock_evidence_evaluation \
  --output outputs/v4/evidence_evaluation_lock_20261025_reaudit

# Bộ pilot đã tạo; prepare chỉ dùng khi tạo namespace mới.
.venv-phase2-batch/bin/python -m scripts.v4.run_evidence_allocation prepare \
  --output outputs/v4/s2_evidence_allocation_20261025_new

# Chỉ chạy sau khi reviewer hoàn tất labels tương ứng.
.venv-phase2-batch/bin/python -m scripts.v4.run_evidence_allocation fit-dev \
  --labels outputs/v4/s2_evidence_allocation_20261025/labels_development.csv
.venv-phase2-batch/bin/python -m scripts.v4.run_evidence_allocation evaluate-validation \
  --labels outputs/v4/s2_evidence_allocation_20261025/labels_validation.csv
.venv-phase2-batch/bin/python -m scripts.v4.run_evidence_allocation evaluate-held-out \
  --labels outputs/v4/s2_evidence_allocation_20261025/labels_held_out.csv
```

Mỗi namespace khóa input/code hashes, query membership, features và policy.
Sửa policy/code cần namespace mới; không reuse model/validation lock sau khi sửa.
Không đổi default CLI/API hoặc champion exporter. Không commit/stage các thay đổi
cleanup đã có sẵn trong workspace.

## Kiểm tra

Regression tập trung: **63 tests PASS** trong 5 suites về identity, offsets,
fail-closed evaluation, allocation lifecycle và frozen submission refinement.
Bao gồm synthetic lifecycle test dev → validation → held-out; kết quả synthetic
không phải kết quả của challenger trên dữ liệu thật.

Audit A và pilot preparation đã chạy trên local artifacts/source thật. Readiness
recheck xác minh source/structure và locked hashes ở cả ba partitions; CSV nhãn
trống bị reject và chưa có model lock hoặc submission mới.

Kiểm tra mở rộng: **293 tests PASS trong 20,42 giây** với:

```bash
.venv-phase2-batch/bin/python -m pytest -q tests/v4 \
  --ignore=tests/v4/test_cli_api_tail.py
```

Hai tests trong `test_cli_api_tail.py` chưa được xác minh. Lượt full V4 bị treo;
chẩn đoán riêng bằng `faulthandler_timeout=10` cho thấy đang chờ trong
`starlette.testclient.TestClient.__enter__` / AnyIO portal, ở test đầu tiên.
Lệnh chẩn đoán timeout sau 25 giây. Chưa xác định nguyên nhân gốc; không tuyên bố
full suite PASS và không sửa API/dependencies ngoài phạm vi. BM25 suite đã PASS
trong lượt 293 tests. `git diff --check` và Python compilation đều PASS; Ruff
không có trong môi trường test hiện hữu.

**Cập nhật sau audit toàn bộ S2:** Giai đoạn B dừng trước annotation. Bộ CSV
đã chuẩn bị là artifact lịch sử, không còn là công việc cần giao reviewer.
Kết luận hẹp: điểm parent reranker cũ không thể đổi 60 slots với margin dương.
Chưa có kết luận về một evidence signal mới hoặc official F2.

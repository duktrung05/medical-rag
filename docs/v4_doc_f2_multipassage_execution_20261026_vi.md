# Thực thi document relevance branch — 26/10/2026

## Quyết định

Trạng thái: **NO_GO, không tạo submission ZIP**. Champion vẫn là cap-10 với
BTC DOCS_F2 `0,2788`, precision `0,1946`, recall `0,3695`.

Candidate đã chạy đủ 1.200 queries bằng cached scores. Nó giữ nguyên 140.095
documents, 72.000 chunks và số documents của từng query, nhưng thay 2.788
document IDs trên 1.066 queries. Gate bị chặn vì validation chỉ đạt `+1/-1`
direct-positive documents, net `0`; điều kiện khóa trước là positive net gain.

## Thay đổi đã thực hiện

Document score cũ là score cao nhất của một passage. Scorer mới dùng 12 feature
ở cấp parent: max/second/top-2/top-3 passage logits, số passages, số passages
qua các ngưỡng, RRF score, số retrieval branches và reciprocal rank theo document
BM25, passage BM25, dense. Logistic model được fit chỉ trên development queries.

Nhãn positive gồm:

- manual query-document evidence đã review;
- historical chunk có exact identity và được BTC feedback đánh positive.

Các document còn lại được giữ đúng nghĩa `unlabeled`, chỉ mang sample weight
`0,02`; không được tuyên bố là negative. Policy khóa trước khi chạy: tối đa ba
swaps/query, entrant phải có max passage score `>= -4`, document log-odds phải
cao hơn victim ít nhất `1,0`, không được loại chunk parent.

## Kết quả local

| Split | Labeled docs | Baseline hit | Candidate hit | Gain | Loss | Net |
|---|---:|---:|---:|---:|---:|---:|
| Development | 1.423 | 1.220 | 1.230 | 11 | 1 | +10 |
| Validation | 397 | 340 | 340 | 1 | 1 | 0 |
| Held-out | 380 | 326 | 328 | 2 | 0 | +2 |

Manual document loss trên validation + held-out là `0`. Structural contracts
đều PASS và relevant chunks giống champion theo từng query. Tuy vậy tập nhãn chỉ
liệt kê positive đã biết, nên các số trên đo positive retention, chưa đo document
precision và không phải dự báo DOCS_F2.

## Hai ca validation giải thích giới hạn

- q442 hỏi ảnh hưởng của chụp X-quang răng 10 lần/ngày. Candidate loại document
  187394 có đoạn trả lời trực tiếp về ảnh hưởng và tần suất chụp X-quang, để lấy
  document 82492 có nội dung chụp X-quang răng nhưng chủ yếu nói phát hiện u răng.
  Đây là một loss có ý nghĩa, không phải lỗi identity.
- q684 hỏi mệt mỏi, đau mạn sườn trái và đi ngoài thất thường. Candidate lấy
  document 638679 về đau bụng bên phải thay document khác; entrant là historical
  positive nhưng laterality/domain match yếu hơn một số unlabeled documents.

Multi-passage statistics giúp thu hồi positive bị max-score ranking bỏ sót, nhưng
không đủ để phân biệt nội dung trả lời đúng trọng tâm. Đặc biệt, nhiều passage
liên quan rộng có thể được model ưu tiên hơn một passage duy nhất trả lời trực tiếp.

## Artifacts

- Implementation: `src/medrag/ranking/document_relevance.py`.
- Full runner: `scripts/v4/run_document_relevance_challenger.py`.
- Judgment packet builder: `scripts/v4/build_document_judgment_packet.py`.
- Contract tests: `tests/v4/test_document_relevance.py`.
- Run namespace: `outputs/v4/doc_f2_multipassage_20261026/`.
- `manifest.json`: policy, hashes, coefficients, split metrics và gate.
- `document_changes.csv`: toàn bộ 2.788 replacements và margins.
- `label_evaluation.csv`: gross gains/losses theo split và provenance.
- `direct_judgment_packet.csv`: 1.554 blinded query-document rows thuộc 300
  queries cố định (180 development, 60 validation, 60 held-out).
- `direct_judgment_key.csv`: doc IDs và entrant/victim roles, tách khỏi packet
  để người chấm không nhìn thấy arm.
- `direct_judgment_manifest.json`: hướng dẫn grade 0/1/2 và hashes.

Targeted regression: **6 passed**. Python compile và `git diff --check` PASS.
Không dùng GPU, không chạy inference mới và không tiêu một lượt BTC.

## Kết luận kỹ thuật

Không promote hoặc nới gate sau khi xem validation. Kết quả cho thấy aggregation
từ các passage scores có tín hiệu recall nhưng chưa có tín hiệu đủ mạnh cho
precision. Bước có khả năng cải thiện DOCS_F2 tiếp theo phải bổ sung direct
query-document judgments ở vùng swap boundary và chấm nội dung document/top
passages cùng query. Packet này đã được materialize và đang ở trạng thái
`AWAITING_DIRECT_JUDGMENTS`. Tiếp tục đổi cap, margin hoặc positive-unlabeled
weights trên cùng validation sẽ là tuning proxy, không tạo thêm bằng chứng về
DOCS_F2.

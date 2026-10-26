# S1/S2 document selection — 24/10/2026

## Mốc chính thức

Người dùng báo điểm của `rerank_refined/submission.zip`, SHA-256
`164fbed38d3d0afdab29a736f6a654d59886fead72560e600a946def3136967d`:
FINAL 0,2274; DOCS_F2 0,2699; CHUNKS_F2 0,1848; document
precision/recall 0,1498/0,4126; chunk precision/recall 0,2050/0,2108.
Receipt local ghi rõ đây là feedback do người dùng cung cấp, chưa tải độc lập từ BTC.

## S1 — fixed cap 120

- Pre-gate reranker `-4`, sau đó lấy prefix tối đa 120 documents.
- Mọi parent đang xuất chunk luôn được bảo vệ.
- 140.684 documents, trung bình 117,2367/query; min/median/max 60/120/120.
- 72.000 chunks, byte-for-byte giống mốc 0,2274 theo từng query.
- ZIP 91.175.297 bytes; SHA-256
  `d80e8b9b4978b49246f8820d1e0f5d1344105c2720431776046426530a13fbc0`.

Known-positive document retention so với positives đang có trong mốc mới:

| Tập | Historical | Manual |
|---|---:|---:|
| Development | 1.213/1.247 (97,27%) | 53/55 (96,36%) |
| Validation | 659/679 (97,05%) | 21/21 (100%) |

## S2 — adaptive score-count

Rule được chọn trên development và sau đó kiểm validation:

1. Giữ pre-gate `-4` của mốc mới.
2. Đếm documents có reranker score `>= -2`.
3. Clip target vào `[80, 140]`; nếu pre-gate còn ít hơn target thì không thêm
   lại document đã bị gate loại.
4. Chọn ranked prefix theo target và luôn bảo vệ mọi chunk parent.

Kết quả:

- 140.095 documents, trung bình 116,7458/query; min/median/max 60/131/140.
- 72.000 chunks, byte-for-byte giống mốc 0,2274 theo từng query.
- ZIP 91.173.448 bytes; SHA-256
  `081f8cfdf77948bc5693d255ece0ddc6141a04584ea8b86f314a5e8568d1649a`.
- So với S1, S2 xuất nhiều docs hơn ở 667 query, bằng ở 46 và ít hơn ở 487;
  thêm 12.125 và bỏ 12.714 query-document pairs. Đây là policy động thực sự,
  không chỉ là một cap cố định khác tên.

Known-positive document retention:

| Tập | Historical | Manual |
|---|---:|---:|
| Development | 1.217/1.247 (97,59%) | 55/55 (100%) |
| Validation | 663/679 (97,64%) | 21/21 (100%) |

S2 giữ nhiều known-positive hơn S1 ở gần như cùng tổng document budget. Đây là
proxy chẩn đoán, không phải official F2; chỉ BTC scoring mới xác định arm thắng.

## Kiểm tra hoàn tất

- 58 targeted tests PASS.
- ZIP CRC, schema, đủ 1.200 query và cap 100.000.000 bytes PASS.
- Không query nào có document ID trùng.
- Toàn bộ chunk parents nằm trong relevant documents.
- Chunk arrays của cả S1 và S2 giống hệt reference 0,2274 theo thứ tự, doc ID và
  text; vì vậy hai probe chỉ thay document selection.

Nên chấm S1 trước để đo fixed-cap effect, rồi chấm S2 để đo riêng lợi ích của
adaptive allocation. Không suy điểm từ known-positive retention.

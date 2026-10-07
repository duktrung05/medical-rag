# ZIP thử nghiệm selector — 20261006

File nộp: `selected/submission.zip` (39.31 MB). Validator: **VALID**.

| Thuộc tính | Giá trị |
|---|---:|
| Query | 1.200 |
| Document/query | 200, giữ nguyên baseline |
| Chunk trước | 144,000 |
| Chunk sau | 120,262 |
| Chunk bị loại | 23,738 (16.48%) |
| Chunk/query trung bình | 100.22 |
| Query không có chunk | 0 |

Luồng chạy: evidence ZIP baseline → đối chiếu corpus → BGE reranker thật trên GPU →
checkpoint/cache → selector raw threshold -6 → ZIP → validator.

Đã kiểm tra đủ query ID, document predictions giữ nguyên, text của mọi chunk được
chọn là text gốc trong baseline, tất cả candidates đã được rerank và mọi chunk
được chọn có score >= -6. Test/lint cho code cập nhật đều qua (43 tests).

Selector loại 12,403 chunks dưới ngưỡng và
11,335 chunks trùng text. Không có nhãn dev/holdout để hiệu
chỉnh threshold; đây là cấu hình thử nghiệm cố định. Chưa có điểm leaderboard.
Bản này tái dùng candidate evidence của ZIP baseline, không mở rộng retrieval pool.

Cache đầy đủ: `rankings.jsonl.gz`. Cấu hình và diagnostics: `selected/`.
Môi trường thực tế và version packages: `environment.json`, `requirements.freeze.txt`.

# P5.2 — Split runner cho P2-A0

`scripts/v4/run_p5_2_split.py` tách submission P2-A0 thành ba bước. Cấu hình
`configs/v4_submission_p2_a0.yaml` được kiểm tra bằng SHA-256 cố định; runner
không sửa trọng số, ngân sách, parent-first allocation, selection hay context.

1. `prepare` chạy V4 BM25, dense, RRF và fallback trên CPU, rồi dùng
   `Pipeline.prepare_rerank` tạo đúng 200 cặp cho mỗi query. Mỗi shard chứa query,
   candidates, payload `title\npassage`, cache key và SHA-256 của input. Chế độ
   `pilot` phát lại **branch hits đã khóa** của sáu query để kiểm tra chính xác
   fusion và các bước sau, không chạy lại dense encoder. Chế độ `full` mới chạy
   dense encoder trên CPU với cùng model/revision và config gốc.
2. `score` đọc các shard trên máy GPU ngoài, kiểm tra score cache theo contract
   model/revision/max length/index/corpus và metadata từng cặp, chỉ gọi
   `BGEReranker` cho cache miss. Score shard ghi atomically, có hash của pair
   shard và identity/hash của từng cặp. Có thể chuyển chỉ `lock.json`,
   `pair_manifest.json`, `pairs/` và `key_contract.sqlite` tới máy GPU; bước
   score không cần corpus hoặc full index. Score shard bên ngoài được đưa về
   qua lệnh `import`.
3. `finalize` kiểm tra đủ scores, hash, count, thứ tự và finite values, dùng
   `Pipeline.finish_rerank` để sắp xếp, chọn documents/chunks và mở rộng context.
   Sau đó chạy validator và ZIP exporter V4 hiện có.

Mỗi bước có thể chạy lại: lock và pair manifest phải khớp, shard đã hoàn thành
không bị ghi đè nếu nội dung khác. `final/` chứa checkpoint từng query. Runner
chỉ ghi `run_manifest.json` và ZIP sau khi toàn bộ query trong lock hoàn tất.

## Pilot đã xác minh

Từ thư mục repository, dùng môi trường dự án:

```bash
export PYTHONPATH=.:outputs/pubmed_runtime_deps_20261012
PY=.venv-phase1/bin/python
OUT=outputs/v4/p5_2_split_pilot_20261022
$PY scripts/v4/run_p5_2_split.py prepare --output "$OUT"
$PY scripts/v4/run_p5_2_split.py score --output "$OUT" --seed-cache outputs/v4/p2_missing_pair_scores_20261021/scores.sqlite --device none
$PY scripts/v4/run_p5_2_split.py finalize --output "$OUT"
```

Để kiểm tra retrieval CPU thật trên đúng sáu query, đặt
`HF_HOME=$PWD/.cache/huggingface` và chạy `prepare --mode pilot-live` với một
`OUT` mới; sau đó dùng `score` và `finalize` cùng `--mode pilot-live`. Runner
đối chiếu identity, thứ tự và SHA-256 của 200 input mỗi query với pilot lock.
Lần chạy `pilot-live` ngày 22/10/2026 đã hoàn tất 6/6 query và 1.200/1.200
cặp khớp; selected objects, BTC payload và ZIP SHA-256 đều trùng bản replay.
Artifact ở `outputs/v4/p5_2_split_pilot_live_20261022/`.

Kết quả: 6/6 query, 1.200/1.200 cặp từ cache, 1.200 documents, 360 chunks.
Selected objects khớp chính xác với P2-A0 replay; canonical BTC payload có
SHA-256 `b7802660f332f9a2a773354a30d4983876745e6b7dd63386143b5d9d073906e6`,
trùng pilot lock. `run_manifest.json` và `export/submission.zip` ở `OUT`.

## Chấm trên GPU ngoài

Sau `prepare`, chuyển các file được liệt kê ở trên sang cùng cấu trúc `OUT`
trên máy GPU. Chạy `score --output OUT --device cuda:0` (thêm `--mode full` khi
lock là full). Chuyển các `scores/<query_id>.json` về CPU và import từng file:

```bash
$PY scripts/v4/run_p5_2_split.py import --output "$OUT" --query-id 53 --score-file /path/to/53.json
```

`import` từ chối shard sai contract, pair hash, identity, thứ tự, số lượng,
non-finite score và score mâu thuẫn với cache đã ghim. Chạy lại `finalize` khi
mọi query có score shard. Full 1.200 query yêu cầu `--mode full` ở từng lệnh;
chưa chạy chế độ đó trong P5.2.

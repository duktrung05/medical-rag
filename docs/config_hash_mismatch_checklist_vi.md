# Checklist kiểm tra config hash — 2026-09-30

Nguyên nhân chênh lệch đã xác nhận: `run_identity.json` của hybrid/rerank chứa SHA256 của YAML dùng xuống dòng CRLF, còn config hiện tại và `manifest.json` chứa SHA256 của cùng nội dung YAML dùng LF. Hai hash không thể so sánh trực tiếp như cùng một chuỗi byte.

| Artifact rerank | SHA256 |
|---|---|
| Config hiện tại / phiên bản LF | `84626175616a6c0d20ee057f4c8cc31121c0f1ff6f1b7b5a84449cb6937aa1c8` |
| `manifest.json` | `84626175616a6c0d20ee057f4c8cc31121c0f1ff6f1b7b5a84449cb6937aa1c8` |
| Config hiện tại chuyển LF thành CRLF | `ebf3646a0a1208e7d05c3ff11c8b6670cc49e1f1518c789215cdb32630505ebc` |
| `run_identity.json` | `ebf3646a0a1208e7d05c3ff11c8b6670cc49e1f1518c789215cdb32630505ebc` |

**Checklist đã thực hiện**

- [x] Hash byte config hiện tại khớp manifest cho cả bốn stage.
- [x] Hash CRLF của config rerank khớp chính xác run identity.
- [x] Hybrid có cùng nguyên nhân: hash LF `1cf7c943e53c647ba6ab4f059e0a1bf133c5b685ffb952525ee3d9d237a08d27`; hash CRLF/run identity `17213098b1e9a3b5b6693d8a1d08ef3cb046adb5351dc4fdf10adc0a97382e44`.
- [x] BM25/dense không có mismatch giữa manifest và run identity.
- [x] Các trường chung khác giữa manifest và run identity khớp: stage, corpus, samples, split, số query và probe.
- [x] Corpus hash tính lại khớp cả hai metadata hybrid/rerank: `b219a1f7f55df30a759753db9f01c29397e68bd8f62125a1b5347da0781ba04a`.
- [x] Samples hash tính lại khớp cả hai metadata hybrid/rerank: `18ee5573930b6fc8a4ffe34e94c16485441445a3028f70078851ea105511c98c`; có 2.210 validation query.
- [x] Rankings checksum tính lại khớp manifest của cả bốn stage.
- [x] Config đã load của hybrid/rerank khớp các tham số trong `config_snapshot`; chỉ khác ba đường dẫn tuyệt đối Windows/Linux: corpus, sparse index và dense index. Việc này không xác nhận checksum nội dung index; corpus đã được kiểm tra riêng.
- [x] Manifest, run identity, rankings và metrics của hybrid/rerank khớp nguyên byte với bản frozen.
- [x] Replay CPU baseline `baseline_20260930` thành công cho cả bốn stage; checksum snapshot và retrieval metrics đều khớp.

**Cách tái kiểm tra nguyên nhân**

Chạy từ thư mục gốc `medical-rag`:

```bash
python - <<'PY'
import hashlib
import json
from pathlib import Path

config = Path('configs/vimed_hybrid_rerank.yaml').read_bytes()
lf = config.replace(b'\r\n', b'\n')
crlf = lf.replace(b'\n', b'\r\n')
identity = json.loads(Path('outputs/vimed_validation/rerank/run_identity.json').read_text())
manifest = json.loads(Path('outputs/vimed_validation/rerank/manifest.json').read_text())
digest = lambda value: hashlib.sha256(value).hexdigest()
assert digest(config) == manifest['config_sha256']
assert digest(crlf) == identity['config_sha256']
print('LF SHA256:', digest(lf))
print('CRLF SHA256:', digest(crlf))
print('Confirmed: line-ending difference explains the config hash mismatch')
PY

python -m scripts.replay_baseline --baseline baseline_20260930
```

**Ảnh hưởng và giới hạn kết luận**

`scripts/benchmark_vimed_validation.py` tính hash bằng `read_bytes()`. Khi resume, worker so sánh toàn bộ identity và sẽ báo `Existing run config/data differ; choose a new output directory` cho hybrid/rerank hiện tại. Dependency loader kiểm tra manifest nhưng chưa đối chiếu manifest với run identity, nên hai đường kiểm tra không đưa ra cùng kết luận.

Chuyển đổi LF/CRLF giải thích chính xác hash khác nhau. Thời điểm và thao tác làm metadata chuyển sang hash LF chưa được xác định từ kiểm tra này. Rankings checksum và replay thành công xác nhận tính toàn vẹn artifact/metric đang lưu; chúng không chứng minh toàn bộ quá trình inference lịch sử có thể chạy lại giống hệt trên máy khác.

**Checklist xử lý cho lần cập nhật tiếp theo**

- [ ] Giữ hash gốc và tạo provenance reconciliation riêng cho artifacts cũ, ghi nguyên nhân LF/CRLF và bằng chứng hash; không thay metadata của snapshot frozen.
- [ ] Với run mới, lưu cả raw config SHA256 và hash sau chuẩn hóa LF; version hóa quy tắc hash. Hash LF chỉ xử lý line endings, không đồng nhất mọi cách biểu diễn YAML.
- [ ] Kiểm tra identity/manifest thống nhất trước khi reuse cache; chỉ chấp nhận khác line endings khi có config gốc hoặc bằng chứng reconciliation xác thực. Không bỏ qua khác biệt tham số.
- [ ] Thêm tests cho LF/CRLF tương đương, thay đổi tham số thực sự bị từ chối, và artifact thiếu bằng chứng reconciliation bị từ chối.
- [ ] Chuẩn hóa line endings cho config mới qua `.gitattributes`, đồng thời giữ nguyên snapshot lịch sử.

Trạng thái: đã xác định nguyên nhân và kiểm tra artifact; sửa cơ chế hash/resume chưa được triển khai trong checklist này. Lần kiểm tra này chỉ thêm tài liệu, không gọi GPU/model.

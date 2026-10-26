# Khôi phục candidate passage coverage cho 87 manual evidence — 26/10/2026

## Kết quả

Đã tạo candidate pool cho toàn bộ 87 query có manual evidence. Mỗi parent đang
có trong cap-10 được mở rộng thành **toàn bộ registered passages** của parent đó.
11 parent không có trong cap-10 được tìm bằng query gốc trên document BM25,
passage BM25 và dense; parent được gom theo RRF trước khi áp cap 4.000. Pool
mới hợp nhất với documents cap-10 và giữ nguyên toàn bộ documents baseline.

| Chỉ số | Kết quả |
|---|---:|
| Query–parent pairs đã kiểm tra | 87/87 |
| Manual parents có trong candidate pool | **87/87** |
| Manual spans đầy đủ trong candidate passages | **87/87** |
| Documents cap-10 được giữ nguyên | Có |
| Query cần retrieval bổ sung | 11 |
| Tổng retrieval CPU | 2.783,36 giây |
| Unique parent entries trong 87 pools | 53.326 |
| Registered passage entries trong 87 pools | 259.327 |

Coverage được xác minh bằng query ID, parent ID, source SHA-256 và corrected
source offsets trong cohort lock. Evidence text chỉ được đọc sau khi pool được
ghi để đo xem passage nào chứa span; không được dùng để xếp hạng hoặc chọn parent
hay passage. 11 query retrieval dùng đúng raw query, index fingerprint đã khóa,
ba nhánh retrieval và FP64 dense reference. Cache được nạp offline.

## Phạm vi kết quả

Kết quả này hoàn tất **bước khôi phục evidence vào candidate pool**. Nó chưa
chấm 259.327 passages, chưa cấp lại 60 chunk slots và chưa thay submission.
Vì vậy không được gọi 87/87 là final output recall hoặc official CHUNKS_F2.

Lượt retrieval nhắm vào 11 query có manual parent vắng mặt trong cap-10; 76
parent còn lại đã có trong documents baseline nên chỉ cần enumerate passages.
Đây là cohort diagnostic khóa theo manual evidence. Các con số 87/87 mô tả đúng
cohort này, không phải một estimate độc lập cho mọi query.

## Artifact và chạy lại

- Script: `scripts/v4/recover_manual87_parent_pool.py`.
- Report đầy đủ và 87 hàng attribution: `outputs/v4/manual87_parent_recovery_20261026/manual87_parent_recovery_report.json`.
- Bảng gọn theo evidence: `outputs/v4/manual87_parent_recovery_20261026/manual87_target_coverage.csv`.
- Pool riêng cho từng query: `outputs/v4/manual87_parent_recovery_20261026/pools/`.
- 11 retrieval traces: `outputs/v4/manual87_parent_recovery_20261026/traces/`.

Chạy từ repository root trong môi trường scorer đã khóa:

```bash
HF_HOME="$PWD/.cache/huggingface" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  .venv-p5-gpu-a6000/bin/python -m scripts.v4.recover_manual87_parent_pool
```

Output dùng namespace write-once để giữ hashes và provenance.

## Bước kế tiếp

Pool hiện có 259.327 passages. Không đưa hết passages này vào cross-encoder cùng
lúc. Bước tiếp theo là xây một shortlist passage/context có budget rõ ràng, rồi
đo lại xem shortlist giữ được bao nhiêu trong 87 manual spans trước khi sửa slot
allocator. Với các evidence đã có parent, việc mở rộng đầy đủ passages đã bỏ
qua lỗi chọn seed/context ban đầu; lớp lọc tiếp theo cần chứng minh nó không làm
mất lại coverage vừa phục hồi.

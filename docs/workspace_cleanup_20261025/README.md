# Dọn workspace V4 — 25/10/2026

Cleanup chuyển các output thử nghiệm cũ, aborted runs, recovery snapshots và
docs lịch sử ra ngoài checkout. Không xóa vĩnh viễn dữ liệu.

## Kết quả

- Đã chuyển **279 mục / 12.511 files**, tổng **16.903.075.359 bytes**
  (khoảng 16,9 GB / 15,74 GiB) khỏi cây làm việc.
- Archive: `/data_hdd_16t/trungnguyen12/medical-rag/project-archives/v4_cleanup_20261025T090704Z/`.
- Gồm 37 root reports/orphan B1 script, 51 docs/refactor files cũ, các
  experiment/submission cũ, aborted runs và pilot/recovery snapshots.
- 479 code/config/test/protected artifact hashes không đổi.
- S2 replay sau cleanup khớp **1.200/1.200 queries**, **140.095 documents**,
  **72.000 chunks**; source store và full-index targets còn truy cập được.
- Verification: **58 V4 tests + 2 repository checks PASS**.

Code V4/configs/models không bị đổi bởi cleanup; chỉ documentation và công cụ
cleanup được thêm/cập nhật. Hai candidate vẫn giữ vì có core reference:
`outputs/v4/phase2_batch_subset/` và `outputs/v4/p5_1_a0_submission_preflight_20261022/`.

## Manifest và khôi phục

- [Plan và inventory từng file](plan.json): original path, lý do, size/inode,
  hash của file nhỏ, dependencies và các mục được giữ lại.
- [Vị trí archive](archive_location.json): thư mục ngoài checkout chứa dữ liệu.
- [Kết quả kiểm tra](verification.json): protected files và inventory sau chuyển.

Trong archive có `cleanup_manifest.json`, `move_journal.json`, bản sao code/docs
trước cleanup và Git diffs. Dữ liệu lớn được rename trên cùng filesystem, giữ
nguyên inode và nội dung.

Khôi phục về đúng paths ban đầu từ repository root:

```bash
python3 scripts/tooling/cleanup_v4_workspace.py restore
```

Restore từ chối ghi đè file đã tồn tại. Khi chỉ cần một experiment, tra manifest
và khôi phục đúng subtree tương ứng từ `<archive>/tree/<original-relative-path>`.
Một số symlinks trong history dùng absolute old paths; restore toàn experiment
về original paths trước khi chạy lại.

## Phạm vi giữ lại

Code V4 và các helpers còn dùng chung, configs/tests, dữ liệu gốc, model cache,
virtual environments, S2 cùng inherited lineage, full index/backing stores,
feedback lịch sử và inputs R6/R8 hiện tại được giữ tại chỗ.

Các scripts lịch sử vẫn được giữ để tránh phá imports/tests; output đã chuyển
cần restore trước khi rerun experiment tương ứng. README và workspace guide
là điểm bắt đầu mới để nhận diện đường V4/S2 chính.

Đây là giảm dung lượng/cấu trúc trong checkout, **không phải giải phóng dung lượng
ổ đĩa**, vì archive nằm trên cùng filesystem và dữ liệu vẫn khôi phục được.

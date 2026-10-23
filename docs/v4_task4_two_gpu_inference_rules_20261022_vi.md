# Rule chạy A0 full inference trên hai GPU

Ngày 22/10/2026. Phạm vi: thay rule một GPU cho **A0 inference 1.200 official queries**. B0 vẫn tạm dừng. Tài liệu này là rule vận hành mới; không tự coi là launcher hoặc worker hai GPU đã được triển khai.

## GPU và worker

| Worker | GPU vật lý dự kiến | UUID | CUDA bên trong worker |
|---|---:|---|---|
| A | 3 | `GPU-8a4d1391-9b9b-428a-691a-9c98607f45a7` | `cuda:0` |
| B | 2 | `GPU-3eb95da2-57af-94b3-2ae9-8e5d46d9c079` | `cuda:0` |

Hai UUID lấy từ các report Task 4 trước đó. **Mỗi lượt chạy phải kiểm tra lại** bằng `nvidia-smi` ngay trước khi khởi chạy. Nếu một GPU không hiện đúng UUID, đang có process khác, driver/CUDA không hoạt động hoặc bị chiếm VRAM, dừng preflight và không khởi chạy worker nào. Không kill hoặc chiếm GPU của process khác; không tự đổi sang GPU0/1. `run_single_gpu.py` hiện tại không phải launcher cho cấu hình này.

Mỗi process chỉ nhìn thấy một UUID qua `CUDA_VISIBLE_DEVICES`, rồi dùng `cuda:0` cho dense query encoder và reranker. Không để cả hai worker nhìn thấy cả hai GPU; không dùng DDP/DataParallel; không để hai process cùng ghi một output, checkpoint hoặc `scores.sqlite`.

## Chia query và tách artifacts

- Khóa đúng 1.200 official query IDs/text/hash và config A0 trước khi chạy. Dùng đúng config smoke A0 `configs/v4_phase2_full_a0_gpu3.yaml` để giữ parity. Config này dùng `document_order` và selector A0; P2-A0 với `parent_first`/`independent` là cấu hình khác, không gộp tên.
- Sắp query theo official ID rồi chia xen kẽ vị trí: worker A nhận index chẵn, worker B nhận index lẻ. Tạo manifest cho cả hai shard trước khi mở model; mỗi shard phải có 600 IDs, không giao nhau, hợp lại đúng 1.200 IDs. Việc chia xen kẽ tránh dồn một vùng query IDs vào một GPU.
- Mỗi worker ghi vào thư mục riêng, ví dụ `full_a0_gpu23_20261022/worker_a/` và `worker_b/`; mỗi thư mục có query checkpoints, log/ledger và score cache riêng. Lock ghi config, index, query IDs, code hashes, model revisions, UUID worker và shard ID.
- Giữ nguyên corpus/index, fusion, rerank budget tối đa 200 pairs/query, max length 512, selector A0 và output schema. Không đổi precision, passage/chunk/context policy hoặc retrieval budgets trong lượt này.

## Tài nguyên và an toàn ghi

- Worker A ghim CPU affinity `4-7`; worker B ghim `0-3`. Giữ `nice 10`, I/O priority idle và đặt tối đa 4 CPU threads/worker. Nếu hệ thống không thể cấp đúng hai tập CPU riêng, preflight fail.
- Hai worker có thể cùng đọc full index đã hoàn tất, nhưng chỉ đọc; không assemble, reindex hoặc thay file trong lúc chạy. Theo dõi RSS tổng hai worker, VRAM từng UUID, disk throughput và query latency. Budget tổng RSS tham chiếu của Task 4 là 32 GiB; dừng an toàn nếu chạm budget hoặc có lỗi I/O/CUDA.
- Cache và query checkpoints thuộc riêng từng shard. Mất worker thì chỉ resume shard đó khi toàn bộ lock khớp. Không để query lỗi thành output rỗng; ledger phải giữ trạng thái `pending/running/failed/complete`, lỗi và attempt.
- Worker kết thúc khi toàn bộ 600 query của nó complete, không failed/pending/running. Không xuất ZIP từ worker.

## Ghép, audit và export

Coordinator CPU chỉ chạy sau khi cả hai worker xong. Coordinator phải:

1. Xác minh UUID/config/index/query/code/model locks khớp; shard ID khác nhau và query ID phủ chính xác 1.200/1.200.
2. Audit mọi `reranked` pair trong trace với entry score cache chính xác theo query hash, doc ID, passage ID, input hash và score hữu hạn; kiểm pair count không vượt 200/query. Merge cache từ hai shard vào cache tổng chỉ sau khi kiểm không trùng key có payload/score mâu thuẫn.
3. Replay selection và context từ ranked candidates đã khóa, kiểm khớp selected checkpoint. Validate toàn bộ IDs, source offsets/hash, grounding, 200 documents, 60 chunks, một chunk/document và equality giữa prediction với selection.
4. Tạo một canonical `predictions.json`, một `submission.zip`, mở lại ZIP kiểm CRC/schema/query coverage/size/payload SHA, rồi ghi `SHA256SUMS`, validation report và run manifest. Trạng thái kỹ thuật sau đó là `FULL_V4_EXPORT_PASS / SCORE_PENDING`.

Không được submit nếu còn query thiếu/lỗi, cache audit lệch, replay mismatch, validator fail hoặc ZIP hash chưa có.

## Việc phải triển khai trước khi mở 1.200 query

Runner hiện có `scripts/v4/run_full_baseline.py` xử lý đủ 1.200 queries trong một process, một score cache và gộp inference với export. `scripts/v4/run_single_gpu.py` chỉ khóa GPU3 và chỉ cho child thấy đúng một GPU. Vì vậy rule hai GPU **chưa chạy được bằng code hiện tại**. Cần triển khai worker theo shard, launcher kiểm/khóa hai UUID, coordinator merge/cache audit/export, cùng các checkpoint/ledger bền vững và preflight yêu cầu `FULL_INDEX_PASS`. Chưa mở full inference trong lúc thiếu các thành phần này.

Full index hiện đã nghiệm thu (`FULL_INDEX_PASS`); report acceptance cũ khóa `full_inference_allowed=false` và vẫn phải giữ nguyên làm lịch sử. Khi triển khai launcher/coordinator mới, tạo execution lock/manifest mới, giữ nguyên report cũ và không sửa status cũ thành đã cho phép.

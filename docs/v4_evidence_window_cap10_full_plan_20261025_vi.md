# Full cap-10 candidate — execution plan

Trạng thái: **EXECUTED — FULL_CANDIDATE_READY_FOR_BTC**. ZIP đã qua full
structural/hash gate; official F2 chưa đo. Chi tiết:
[execution report](v4_evidence_window_cap10_full_execution_20261026_vi.md).

## Treatment

Áp dụng nguyên policy cap-10 đã đạt `SCREENED_CANDIDATE` lên đủ 1.200 queries:
S2 documents giữ nguyên; fixed context, BGE revision, ba windows 192 tokens,
max-logit aggregation và tối đa 10 swaps/query giữ nguyên. Không sweep hay đổi
threshold. 60 pilot queries tái sử dụng exact pair/scores đã khóa; chỉ chấm
1.140 queries còn lại.

## Execution

1. Materialize và hash toàn bộ fixed contexts/windows trước model load.
2. Chia deterministic 1.140 queries mới theo vị trí chẵn/lẻ cho hai workers;
   mỗi worker chỉ thấy một A6000 (GPU 2 hoặc GPU 3), chạy parity 128 anchors.
3. Finalize score coverage chỉ khi hai worker manifests đủ và không trùng.
4. Allocate 60 chunks/query với cap 10, xác minh documents giống S2 từng query,
   validate toàn payload, tạo deterministic single-entry ZIP.
5. Giữ S2 làm champion; candidate chỉ mang trạng thái ready for BTC, không tự
   tuyên bố score gain.

## Fail-closed gates

Dừng nếu code/input/pair/score hash đổi, parity vượt 1e-3, score non-finite,
worker overlap/missing, pilot validation/held-out output không replay exact,
documents khác S2, structural validator fail, ZIP quá 100 MB hoặc ZIP payload
khác predictions. Không submit tự động.

Namespace: `outputs/v4/evidence_window_cap10_full_20261025/`.

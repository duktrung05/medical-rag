# P5.3B — Audit numerical và semantic parity

Ngày audit: 22/10/2026. Input là fresh Tesla T4 score ZIP SHA-256
`0434ce1c30a2ab7bfe727eb10a1e7d7c7ca35d1d43afd0cde6efaabf2e036fcc` và
P2-A0 pilot đã khóa. Audit không chạy retrieval, không gọi lại reranker cho 1.200
pairs, không sửa config hoặc score artifacts.

## Quyết định

- `exact_parity`: **NO-GO**. Có 608/1.200 scores khác, 30 pairwise rank
  inversions, document order khác ở cả sáu query và canonical BTC payload khác.
- `semantic_parity`: **GO** theo contract riêng: bỏ qua thứ tự list và duy nhất
  trường số `rerank_score`, tập 200 document IDs và 60 chunk IDs của mỗi query
  giống nhau; text, source offsets, context passage IDs, source hashes, seed
  evidence và provenance còn lại giống tuyệt đối.
- Quyết định release: **NO-GO**. Semantic GO không thay exact NO-GO vì repository
  không chứa source hoặc contract chính thức chứng minh BTC evaluator bỏ qua order.

| Query | Scores khác | Max abs diff | Pair inversions | Document set | Chunk set | Chunk order |
|---:|---:|---:|---:|---|---|---|
| 53 | 31 | 0.005859375 | 0 | exact | exact | exact |
| 130 | 147 | 0.01953125 | 5 | exact | exact | khác |
| 380 | 170 | 0.01171875 | 9 | exact | exact | khác |
| 754 | 84 | 0.01171875 | 5 | exact | exact | khác |
| 771 | 96 | 0.0078125 | 9 | exact | exact | khác |
| 1144 | 80 | 0.013671875 | 2 | exact | exact | exact |

Mean absolute score difference là `0.0017174720764160157`; max là
`0.01953125`. `rank_inversions.csv` chứa đủ 30 cặp đảo thứ tự. Query 53 vẫn
đổi document order do tie/secondary document ordering dù không có inversion
trong thứ tự thuần của 200 pair scores.

## Inference contract và nguyên nhân drift

Cả hai phía dùng `BAAI/bge-reranker-v2-m3` revision
`953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`, raw logits, max length 512 và
input `query + (title + newline + passage)`. Model và tokenizer snapshot hashes
của T4 đã được khóa và pair input hashes khớp 1.200/1.200.

Tokenizer là `XLMRobertaTokenizer` từ cùng snapshot, dùng right padding/right
truncation mặc định. Snapshot khai báo model max length 8.192, còn CrossEncoder
khóa effective max length ở 512. `tokenizer.json` SHA-256 là
`69564b696052886ed0ac63fa393e928384e0f8caada38c1f4864a9bfbf379c15`.

Implementation `BGEReranker` nạp weights bằng `torch.float16`; không dùng
autocast. `CrossEncoder.predict()` của sentence-transformers 4.1.0 mặc định
batch size 32, tokenize với dynamic padding và truncation. Vì P2-A0 cache được
xây tăng dần, các rows reference không có cùng một batch provenance: pilot RTX
A6000 tái sử dụng 897 cached pairs và chấm mới 303 pairs; sáu missing pairs sau
đó được bổ sung trên CPU. Score-cache contract không ghi hardware, package
versions, attention kernel, batch grouping hoặc padding shape theo từng row.
Fresh T4 chấm toàn bộ 200 pairs/query trong một lượt mới. Đây là các nguồn
numerical drift còn khả dĩ; bằng chứng không hỗ trợ lỗi model revision,
tokenizer, truncation limit hay input construction.

FP32 diagnostic dùng 12 pairs khóa trước từ vùng score sát/đảo rank, đúng
sentence-transformers 4.1.0, transformers 4.51.3, tokenizer 0.21.4 và model
revision. Kết quả batch-12 so batch-1 có max difference chỉ
`3.159046173095703e-06`, nhỏ hơn nhiều drift FP16 quan sát. FP32 gần T4 hơn
baseline ở 7/12 pairs, gần baseline hơn ở 1/12. Kết quả này phù hợp với giả
thuyết drift do FP16 hardware/kernel và reference cache có provenance trộn,
nhưng chưa chứng minh duy nhất nguyên nhân vì torch CPU diagnostic là
`2.14.1+cu130`, không phải torch T4 `2.6.0+cu124`, và batch-12 không tái tạo
mọi companion của batch 32 cũ.

Fresh T4 canary xử lý 1.200 pairs trong `40.39864754100006` giây, đạt
`29.703964687979607` pairs/giây. Peak VRAM quan sát bằng `nvidia-smi` là
`2389` MiB; PyTorch peak allocated/reserved lần lượt là khoảng `1459`/`2256` MiB.

## BTC order audit

V4 schema dùng JSON arrays `relevant_docs` và `relevant_chunks`; serializer giữ
thứ tự selected và local validator yêu cầu serialized output bằng đúng selected
order. Evaluator F2 nội bộ trong repository tính precision/recall theo membership
và vì thế không nhạy thứ tự. Tuy nhiên evaluator nội bộ không phải bằng chứng về
BTC evaluator chính thức. Nếu BTC thực sự tính set-based F2 trên toàn list, các
permutation này không đổi DOCS_F2/CHUNKS_F2; nếu BTC có cutoff, rank weighting,
hoặc xử lý list theo vị trí thì điểm có thể đổi. Trạng thái chính thức vì vậy là
`UNVERIFIED`, không suy diễn F2 parity từ semantic parity.

## Artifacts

- `outputs/v4/p5_3b_cpu_gpu_parity_20261022/p5_3b_parity_report.json`
- `outputs/v4/p5_3b_cpu_gpu_parity_20261022/score_differences.csv`
- `outputs/v4/p5_3b_cpu_gpu_parity_20261022/rank_inversions.csv`
- `outputs/v4/p5_3b_cpu_gpu_parity_20261022/fp32_subset_manifest.json`
- `outputs/v4/p5_3b_cpu_gpu_parity_20261022/fp32_probe.json`

Nguồn upstream dùng để xác nhận behavior của CrossEncoder 4.1.0:
`https://raw.githubusercontent.com/UKPLab/sentence-transformers/v4.1.0/sentence_transformers/cross_encoder/CrossEncoder.py`.

# P0 — Replay đường đi của từng historical positive

## Kết luận chính

Replay CPU đã khớp tuyệt đối với A0 baseline, parent-first selection và adjacent-context artifacts trên 6 full-index pilot queries (`53, 130, 380, 754, 771, 1144`). Không gọi retrieval hoặc model mới. Phạm vi có 37 cặp query/document từng chứa chunk được chấm positive trong baseline cũ.

Nút thắt đã chuyển tầng:

- A0 có **7/37 positive parents đã vào fused nhưng không có passage nào được rerank**.
- Parent-first d200 giảm con số trên xuống **0/37**: mọi positive parent được chọn đều có đúng một scored passage.
- Sau sửa allocation rerank, có **7/37 positive parents đã được score và chọn document nhưng không được xuất chunk vì đứng sau 60 chunk slots**.

Vì vậy lỗi còn lại không còn chủ yếu là “selected parent chưa được rerank”. Trên trace parent-first hiện tại, lỗi P0 lớn nhất có thể tác động bằng allocation là cách cấp 60 chunk slots.

## Đường đi hiện tại của 37 positives

| Điểm dừng đầu tiên | Số | Query/document |
|---|---:|---|
| Không có trong top limits của cả ba retrieval branches | 2 | `(130,2761519)`, `(754,1644439)` |
| Có ở dense nhưng bị cắt khỏi fused400 | 2 | `(130,2796009)` dense rank 195; `(771,563024)` dense rank 151 |
| Fused nhưng không được parent-first rerank | 0 | — |
| Đã rerank nhưng không được chọn document d200 | 0 | — |
| Đã score + chọn document, bị cắt bởi top-60 chunks | 7 | `(53,756661)`, `(130,546076)`, `(130,553119)`, `(130,2968260)`, `(754,2713777)`, `(771,484565)`, `(771,494605)` |
| Đã xuất chunk nhưng context không chứa nguyên old anchor | 5 | `(53,128866)`, `(53,741392)`, `(130,546716)`, `(1144,17620)`, `(1144,142195)` |
| Đã xuất và context chứa nguyên old anchor | **21** | xem `failure_groups.json` |

Bảy trường hợp bị top-60 có final document ranks `70, 75, 84, 99, 99, 100, 126`. Chúng đều đã có parent-first pair, rerank score và `document_evidence`; trường `evidence` rỗng chỉ vì 60 slots đã được cấp cho các documents xếp trên.

## Replay baseline A0

Trên đúng 37 positives:

| Trạng thái | A0 | Parent-first d200 |
|---|---:|---:|
| Positive parents được chọn | 24 | **33** |
| Positive parents có exported chunk | 24 | **26** |
| Export chứa nguyên old anchor | 15 | **19** |
| Sau adjacent context chứa nguyên old anchor | 18 | **21** |
| Fused parent không có passage được rerank | 7 | **0** |
| Selected parent không có reranked passage | 0 | **0** |
| Scored + selected nhưng bị top-60 cắt chunk | 0 theo A0 coupling | **7** |

Bảy A0 positives bị thiếu rerank là `(53,756661)`, `(130,553119)`, `(130,2911632)`, `(380,1222276)`, `(380,1402274)`, `(771,484565)`, `(1144,536369)`. Hai positives đã rerank nhưng A0 không chọn là `(130,546076)` và `(130,2968260)`.

A0 buộc document phải có exported chunk nên không thể biểu diễn trạng thái “document được chọn nhưng chunk bị cap”. Independent selection sửa được coupling đó, nhưng làm lộ rõ bước allocation 60 chunks riêng biệt.

## Giới hạn tối đa có thể cứu bằng allocation

Ceiling dưới đây là oracle exact-anchor containment: nó dùng historical label để hỏi “nếu biết trước positive thì tối đa cứu được bao nhiêu”. Đây là giới hạn chẩn đoán, không phải một policy có thể dùng online và không phải official F2.

| Phạm vi thay đổi | Exact anchors tối đa | Gain so với 21 hiện tại | Trường hợp thêm |
|---|---:|---:|---|
| Giữ scored seed, chỉ chuyển 60 chunk slots | **26/37** | **+5** | `(53,756661)`, `(130,553119)`, `(130,2968260)`, `(754,2713777)`, `(771,484565)` |
| Giữ fused400 và 200 rerank pairs, cho phép score candidate passage khác rồi chuyển chunk slots | **28/37** | **+7** | thêm `(53,128866)`, `(1144,142195)` |
| Cho phép sửa retrieval/fusion và allocation | **31/37** | **+10** | thêm `(130,2796009)`, `(130,2761519)`, `(771,563024)` |
| Exact ceiling từ registered source | **31/37** | **+10** | 6 anchors còn lại không tồn tại nguyên văn trong source hiện tại |

Kết luận P0 về allocation: **trên candidate pool hiện tại, allocation có thể nâng exact containment từ 21 lên tối đa 28/37**. Trong đó +5 đến từ allocation top-60, +2 cần đổi passage nhận rerank. Chín trường hợp không thể cứu chỉ bằng allocation trong fused pool: 3 cần retrieval/fusion và 6 không thể tái tạo exact old anchor từ registered source.

Các tầng cứu tối thiểu, phân hoạch đủ 37 positives:

- Đã đúng: 21.
- Chỉ đổi chunk allocation: 5.
- Đổi passage nhận rerank trong candidate pool: 2.
- Đổi retrieval/fusion rồi allocation: 3.
- Không exact-recoverable từ registered source: 6.

## Ý nghĩa đối với bước sửa tiếp theo

P0 không ủng hộ rerank lại toàn bộ. Parent-first đã xóa lỗi “fused positive không được rerank” trên 6 queries. Gain allocation gần nhất nằm ở bộ chọn 60 chunk-bearing parents: cần chọn 60 parents xuất chunk bằng tín hiệu riêng thay vì mặc định lấy 60 documents đầu sau rerank.

Hai trường hợp `(53,128866)` và `(1144,142195)` cho thấy one-passage-per-parent vẫn có giới hạn: parent đúng và đã xuất chunk, nhưng passage được parent-first score không phải candidate passage có context bao old anchor. Hai ca này là phạm vi tối đa cần thêm passage allocation/rerank trong candidate pool hiện tại.

Hai dense hits bị fused400 cắt và một parent không có trong branch top limits nhưng source còn exact anchor nằm ngoài ceiling allocation. Không tăng thẳng fusion lên 800: replay trước đó cho thấy top-200 parent coverage giảm. Cần thay cách hợp nhất theo parent hoặc quota branch trong arm riêng.

## Artifacts và tái hiện

Artifact chuẩn dùng hậu tố `v3`; hai namespace trước là lượt trung gian trước khi tách tên rescue tier cho chính xác:

- `positive_paths.csv`: một dòng/positive với branch ranks, fused/pre-rerank/final ranks, pair position, scored/export/context flags, first failure và rescue tier.
- `positive_paths.jsonl`: trace đầy đủ gồm raw branch hits, fused items, scored passage IDs/scores, exported evidence và derived context.
- `failure_groups.json`: danh sách query/document theo failure stage và rescue tier.
- `allocation_ceiling.json`: các oracle ceilings và điều kiện cố định.
- `summary.json`, `per_query.json`, `manifest.json`, `input_hashes.json`: tổng hợp, hash và replay contract.

Đường dẫn: `outputs/v4/p0_positive_path_replay_20261021_v3/`.

Lệnh tái hiện vào namespace mới:

```bash
.venv-phase2-batch/bin/python scripts/v4/replay_positive_paths.py \
  --output outputs/v4/p0_positive_path_replay_NEW_DIRECTORY
```

Script từ chối ghi đè, kiểm hash các inputs trước/sau, assert A0 parity, parent-first parity, context parity, 37/37 row coverage và các stage totals. Kết quả ghi `model_calls=0`, `official_f2=UNMEASURED`.

Kiểm chứng sau khi tạo artifact: compile pass; 75 tests V4 pass khi bỏ riêng `test_cli_api_tail.py` đã biết bị treo trong fixture môi trường; 37 rows và toàn bộ artifact hashes trong manifest được kiểm lại độc lập.

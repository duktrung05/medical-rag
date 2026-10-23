# Kết quả thu thập PubMed mới — 2 RTX A6000

Hoàn tất và đối soát: 13/10/2026 07:35:35, giờ Việt Nam.

**Đã hoàn tất xử lý 240.000 official ID–URL/PMID.** Thu được 239.770 bản ghi có abstract, 126 bản ghi chỉ có metadata, 104 PMID không được EFetch trả về sau các lượt nhóm nhỏ/riêng lẻ. ESummary cũng không trả summary cho cả 104 PMID. Không kết luận các URL đã bị xóa.

| Kết quả | Số lượng |
|---|---:|
| Tổng official IDs đã đối soát | 240.000 |
| Bản ghi có abstract | 239.770 (99,904%) |
| Chỉ metadata | 126 |
| API không trả bản ghi | 104 |
| Dòng export documents, bao gồm trạng thái thiếu | 240.000 |
| Raw XML gzip batch được xác minh | 1.203 |
| Bản ghi thực có trong raw được tham chiếu | 239.896 |
| Vector document hợp lệ | 239.770 × 1.024 chiều |
| Tiêu đề rỗng trong records thu được sau sửa | 0 |
| Tiêu đề sách bổ sung từ BookTitle | 48 |

Corpus revision: `71be0e1239a19dfd7a717e2bc13e3bec21bd0334`. Model/tokenizer BGE-M3 revision: `5617a9f61b028005a4858fdac845db406aefb181`.

## Giới hạn tài nguyên đã dùng

- Fetch: 4 HTTP workers, 1 request/giây tổng, nice 10, affinity tối đa 8 cores.
- Embedding: GPU 0 và GPU 1; một encoder mỗi card và 4 worker chuẩn bị dữ liệu mỗi card. Hai nhóm dùng tối đa 8 CPU cores tổng, threads thư viện giới hạn 1.
- Encode batch 32, fp16, allocator budget tối đa 12 GiB/card. Đo thực tế cả nhóm GPU dùng khoảng 4,4 GiB RAM, VRAM driver khoảng 1,85 GiB/card; CUDA allocator peak khoảng 1,49 GiB/card.
- Thời gian collector: 1.390,9 giây, khoảng 23,2 phút. Lượt encode đầu sau khi model sẵn sàng khoảng 517,9 giây, khoảng 8,6 phút; sau đó chỉ cập nhật 26/120 checkpoints bị ảnh hưởng bởi sửa 48 tiêu đề sách.
- Logs tài nguyên: `/data_hdd_16t/trungnguyen12/medical-rag/medical-rag/outputs/pubmed_240k_plan_2gpu_20261012/resource_snapshots.jsonl`. GPU đã được giải phóng khi encoder kết thúc; job không chỉnh cấu hình GPU toàn server.

## Kiểm tra chất lượng và giới hạn

Đối soát toàn bộ ID–PMID–official URL, checksum raw/sidecar, cấu trúc XML, text tái tạo từ title/abstract sections, 48 Parquet parts, ID–text hash của vector và normalization/finite values. 9 offline tests passed. Có resume ở fetch và checkpoint encoder; batch API lỗi đã được phục hồi.

91.970 documents vượt 384 tokens của encoder. **Toàn bộ title/abstract vẫn được lưu trong documents**; vector chỉ dùng đầu vào giới hạn 384 tokens. Đây là vector cấp document, chưa phải embedding toàn bộ chunks. Metadata language là ngôn ngữ publication do PubMed cung cấp, không phải kết quả language detection cho abstract.

Phạm vi thu thập là PubMed citation/abstract, chưa bao gồm full text PMC/publisher. Đây là release bổ sung; chưa hợp nhất vào BM25/dense/index hoặc chuyển runtime truy hồi chung.

## Đường dẫn bàn giao

Root: `/data_hdd_16t/trungnguyen12/medical-rag/medical-rag/data/vibiomir/pubmed/71be0e1239a19dfd7a717e2bc13e3bec21bd0334/run_20261013_2gpu`.

- `documents/part_*.parquet`: 48 shards, mỗi dòng có official ID/URL, PMID, trạng thái, title/text, language metadata, raw/text checksums.
- `raw/*.xml.gz` và sidecars: raw có thể parse lại offline.
- `metadata_only.parquet`: 126 records thiếu abstract.
- `missing_pmids.parquet`: 104 official ID–URL/PMID kèm lý do chưa có record.
- `vectors/pubmed_document_ids.npy`: IDs tăng dần.
- `vectors/pubmed_document_vectors.npy`: fp16, shape `(239770, 1024)`, normalized; cùng thứ tự hàng với IDs.
- `vectors/pubmed_document_token_counts.npy`: token counts trước truncation.
- `vectors/embedding_report.json`: vector checksums và provenance theo part.
- `release_manifest.json`: nguồn/queue, kết quả toàn bộ, checksums, versions và code files.
- `state.sqlite`, `book_title_repair.json`: trạng thái/attempts và lịch sử sửa từ raw.

Runbook và lệnh resume: `/data_hdd_16t/trungnguyen12/medical-rag/medical-rag/outputs/pubmed_240k_plan_2gpu_20261012/EXECUTION.md`. Runtime Python/CUDA được chuẩn bị riêng tại `outputs/pubmed_runtime_deps_20261012`.

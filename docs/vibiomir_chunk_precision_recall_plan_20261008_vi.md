# Kế hoạch cải thiện chunk precision và recall — 08/10/2026

## Mục tiêu và mốc đối chứng

Tìm thêm evidence đúng và giảm evidence sai; không giả định siết threshold sẽ
tăng cả precision và recall. Tách từng thay đổi, giữ kết quả đối chứng và đo
điểm chính thức trước khi đổi cấu hình pipeline chính.

Mốc đã được người dùng chấm: `vibiomir_threshold_neg2_20261008/submission.zip`.

| Chỉ số | Mốc threshold -2 |
|---|---:|
| FINAL_SCORE | 0.1681 |
| DOCS_F2MACRO | 0.2802 |
| CHUNKS_F2MACRO | 0.0560 |
| DOCS_PRECISION | 0.2665 |
| DOCS_RECALL | 0.3190 |
| CHUNKS_PRECISION | 0.0675 |
| CHUNKS_RECALL | 0.0618 |

Cấu hình mốc: max_docs=200, max_chunks=60, chunks_per_doc=1,
chunk_threshold=-2, document_mode=threshold, không dùng delta.
Trung bình 74.98 docs và 53.1525 chunks/query; 1/1200 query rỗng.

Cache xếp hạng:
`outputs/evidence/retrieval_crawl_rerank80_20261008/rankings.jsonl.gz` (~65 MiB).
Nó có 200 candidates đã chấm/query, tổng 240.000 cặp query–chunk.
Cache embedding đã có (~5.5 GiB):
`data/vibiomir/emb_chunks_crawl_20261007_fused_crawl_20261007_d1616e6a4b54c657.npz`.

Diagnostics của mốc: 81.777 candidates bị loại do cap một chunk/document,
26.560 do cap 60 chunks/query, 59.742 do threshold, 8.138 do trùng text,
63.783 được chọn. Đây là thống kê lọc, không phải nhãn relevance.

## Nguyên tắc thí nghiệm

- Mỗi so sánh chỉ đổi một yếu tố; tên ZIP và config phải chỉ rõ yếu tố đó.
- Với test tập trung vào chunk, đặt doc_threshold=-2 rõ ràng để thay
  chunk_threshold không tự thay document gate. Cùng cache và cùng document
  ordering thì tập relevant_docs phải giữ nguyên; kiểm tra trước khi chấm.
- Giữ corpus, fused candidates, phiên bản model và text nguồn trong các test
  selector. Không backfill chunk dưới ngưỡng.
- Ghi tên ZIP, SHA-256, config, provenance, số lượng đầu ra, bảy metric,
  phase/time chấm. Không suy số gold từ tỷ số macro precision/recall.
- Chọn kết quả tăng CHUNKS_F2MACRO và không giảm FINAL_SCORE so với đối chứng
  cùng giai đoạn. Nếu cả precision và recall cùng tăng, đó là bằng chứng mạnh
  hơn; nếu đánh đổi, ghi rõ thay vì gọi là cải thiện cả hai.
- Local chưa có scorer chính thức/qrels đầy đủ. Internal set metrics trong
  calibrate_evidence.py không thay thế luật matching chunk text chính thức.

## Giai đoạn 1 — Khai thác cache bằng CPU

Không chạy model, không đọc lại toàn bộ corpus, không cần GPU.

1. Lưu kết quả mốc -2 vào bảng experiment.
2. Nhận điểm của hai ZIP đã tạo: threshold -2.5 và threshold -2.5 + delta6.
   Các ZIP này giữ document gate mặc định theo chunk threshold, nên chỉ có
   thể so sánh delta giữa hai bản để giữ documents cố định; so với -2 thì
   cả document gate lẫn chunk gate thay đổi.
3. Tạo đối chứng CPU với doc_threshold=-2 explicit; xác nhận prediction khớp
   mốc -2 nếu mọi tham số khác giữ nguyên.
4. Tạo test chunks_per_doc=2, cap tổng vẫn 60, threshold chunk/doc vẫn -2.

| ID | Chunk threshold | Doc threshold | Chunk delta | Chunks/doc | Cap | So với |
|---|---:|---:|---:|---:|---:|---|
| S0 | -2 | -2 | null | 1 | 60 | Mốc đã chấm |
| S1 | -2 | -2 | null | 2 | 60 | S0: chỉ thay cap mỗi doc |
| S2, tùy chọn | -2.5 | -2 | null | 1 | 60 | S0: chỉ thay chunk gate |
| S3, tùy chọn | -2.5 | -2 | 6 | 1 | 60 | S2: chỉ thêm delta |

S1 là test mới ưu tiên. S2/S3 chỉ cần nếu muốn tách tác động chunk gate khỏi
document gate sau khi đã có điểm hai ZIP -2.5 hiện tại.

Kiểm tra đủ 1200 query, parents hợp lệ, không trùng identity, score thật,
caps đúng, ZIP giải nén được, giới hạn dung lượng theo validator hiện có.
Đo số second chunks được chọn và mức overlap text trong cùng document;
không gọi hai chunks gần như cùng nội dung là evidence đa dạng.

CPU replay trước đây mất khoảng 7 giây/ZIP trên server này; thời gian có thể
thay đổi theo tải, nhưng giai đoạn này thường ở mức giây đến phút.
Leaderboard scoring là bước riêng, chưa nằm trong thời gian tạo ZIP.

## Giai đoạn 2 — Thứ tự và độ đa dạng evidence, CPU

Code hiện tại ưu tiên document rồi mới tới chunks của document đó. Với
chunks_per_doc=2, chunk thứ hai của doc đầu có thể được chọn trước chunk
điểm cao hơn của doc sau.

1. Thêm lựa chọn chunk ordering: giữ document_order để chọn docs, nhưng khi
   chọn chunks thì có thể sort toàn cục theo raw reranker score giảm dần.
   Giữ stable tie-break theo thứ tự cache; scores null không qua gate.
2. Giữ behavior mặc định để tái lập baseline; ghi ordering vào provenance.
3. Test có ý nghĩa: second chunk yếu của doc A không chiếm slot của first
   chunk mạnh của doc B, parent/cap/gates vẫn được áp dụng, replay cũ khớp.
4. So S1 với S1-global: chỉ thay ordering; không đổi threshold/caps/dedup.
5. Nếu hai ZIP có cùng tập dự đoán thì không cần một lượt chấm mới để đánh
   giá set metrics; vẫn cần xác minh luật scorer trước khi giả định thứ tự
   không ảnh hưởng matching.
6. Sau khi có biến thể thắng, thử max_chunks=80; chỉ đổi cap tổng. Nếu recall
   tăng nhưng precision/CHUNKS_F2 giảm thì không nhận cấu hình đó.

Không cần GPU cho sửa selector, unit tests và replay. Không nới cap mỗi doc,
thêm near-duplicate filter và đổi threshold cùng một test. Near-duplicate
filter chỉ thử trong cùng doc khi audit xác nhận overlap gây lãng phí slot.
Khử trùng text xuyên doc cần kiểm chứng scorer vì doc_id khác nhau.

## Giai đoạn 3 — Mở rộng candidate chunk pool, dùng một GPU

Mục tiêu: cho reranker thấy thêm passage có thể đã bị dense shortlist top200
loại. Chưa đổi top400 documents của fusion hay rebuild index.

1. Giữ `chunks_crawl_20261007.parquet`, `fused_crawl_20261007.parquet` và encoder.
2. Kiểm provenance của embedding cache; code rank.py không đưa pool/rerank_top
   vào embedding cache key, nên có thể tái dùng embedding nếu inputs khớp.
3. Chạy pool=400 và rerank_top=400, giữ selector đối chứng của giai đoạn này.
4. Cache 400 scores/query vào đường dẫn mới. Từ cache mới replay CPU để tune
   selector; không rerank lại cho mỗi threshold.
5. Chạy thử 50 query để đo tốc độ và peak VRAM trước full 1200 query. Không
   suy điểm leaderboard từ mẫu 50 query không có nhãn.

Chi phí mặc định: tối đa 480.000 cặp, khoảng gấp đôi số cặp pool200.
Reranker hiện tại sẽ chấm lại cả 400; chưa hỗ trợ nối cache điểm tăng dần.
Nếu bổ sung cơ chế reuse đúng `(query, doc, chunk, text, title, model settings)`
thì chỉ cần chấm thêm tối đa 240.000 cặp mới. Đó là tối ưu cần triển khai,
không phải khả năng sẵn có của command hiện tại.

Một GPU là đủ cho kế hoạch; dùng batch theo VRAM đo được. Không cần nhiều GPU.
Ước lượng thời gian sau pilot: số pairs chưa chấm / throughput pairs/s,
cộng thời gian load cache và dense shortlist. Chưa có benchmark mới để hứa
thời gian hoàn tất. Không chạy crawler/extractor/full index build cho bước này.

Nếu mở rộng pool không cải thiện: audit lỗi trước khi tăng tiếp lên 800.
Test phân bổ 2–4 chunks/document hoặc hybrid chunk retrieval nhằm giữ đa dạng
documents có thể là bước tiếp theo, nhưng cần chấm scores mới bằng GPU.

## Giai đoạn 4 — Audit passage và cải thiện chunking

1. Review mẫu 50–100 query phân tầng theo VI/ZH, score cao/thấp, số chunk và
   dạng câu hỏi. Nhãn trong pool chỉ dùng để đánh giá recall trong pool.
   Muốn đo missed docs toàn corpus cần qrels hoặc mở rộng phạm vi annotation.
2. Phân lỗi: không có doc đúng; có doc đúng nhưng không có passage đúng trong
   shortlist; passage đúng nhưng score thấp; trùng passage; text extraction
   thiếu/boilerplate; evidence không khớp quy tắc chấm.
3. Kiểm scorer chính thức trước khi tối ưu theo các comment LCS/40% trong
   extract.py/expand.py: repo hiện ghi rõ các luật này chưa được xác minh.
4. Thử window ngữ cảnh quanh chunk đã chọn trong cùng doc: giữ doc/center/cap
   như đối chứng, chỉ đổi text. CPU đọc corpus và đóng ZIP, có thể tốn RAM/I/O
   hơn replay 65 MiB; không tự động coi score của chunk trung tâm là score
   của toàn đoạn mở rộng. Nếu chấm lại text mở rộng thì dùng GPU.
5. Nếu lỗi ranh giới/truncation rõ: chunk theo tokenizer và ranh giới câu/đoạn,
   đo độ dài thực tế title+query+passage so với giới hạn 512 của reranker.
   Không chọn target length chỉ theo word count xấp xỉ.

Extraction/chunking mới chạy CPU. Khi đổi text được dùng trong retrieval cần
embedding và rerank mới, nên dùng GPU. Bắt đầu trên mẫu có nhãn trước khi
rebuild toàn corpus. Corpus mới sẽ làm cache embedding cũ không còn hợp lệ.

## Giai đoạn 5 — Fine-tune nếu nhãn chỉ ra reranker là nút thắt

Chuẩn bị positive và hard negatives (cùng bệnh/chủ đề nhưng sai ý hỏi), chia
dev/holdout theo query và hạn chế trùng nguồn. Annotation/evaluation CPU;
fine-tune và inference reranker dùng GPU. Chỉ làm sau khi có nhãn đủ tin cậy
và các thử nghiệm selector/pool đã được đo, không ưu tiên ngay lúc này.

## Kết quả cần bàn giao và điểm dừng

Mỗi run: submission.zip, selection_config.json, diagnostics, summary,
provenance/SHA-256 và leaderboard_feedback khi người dùng gửi điểm.
Giữ baseline và outputs cũ; mọi run dùng output riêng. Không tự upload
leaderboard hoặc sửa pipeline mặc định trong bước lập kế hoạch.

Ưu tiên triển khai: S1 bằng CPU → global ordering bằng CPU → cap80 nếu có
cơ sở → pool400 bằng một GPU → audit/chunking → fine-tune nếu cần.
Không cần giữ GPU để làm hai giai đoạn đầu. Chỉ bật giai đoạn GPU sau khi
đã có kết quả CPU hoặc audit cho thấy thiếu evidence trong candidate pool.

Đây là kế hoạch; chưa triển khai thêm selector, chưa tạo các ZIP S1/S2/S3 và
chưa khởi chạy tác vụ GPU trong lượt này.

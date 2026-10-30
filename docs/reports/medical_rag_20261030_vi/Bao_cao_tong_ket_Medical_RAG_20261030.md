# BÁO CÁO TỔNG KẾT DỰ ÁN

MEDICAL-RAG
Truy xuất tài liệu và bằng chứng y khoa đa ngôn ngữ

> Lựa chọn mô hình • Kỹ thuật • Workflow • Kết quả
Khó khăn • Bài học • Hướng cải thiện

Ngày lập báo cáo: 30/10/2026
Mốc dữ liệu thực nghiệm mới nhất tìm thấy: 26/10/2026
Phạm vi: lịch sử benchmark ViMed và hệ thống thi đấu V4 / ViBioMIR

Báo cáo tổng hợp từ mã nguồn, cấu hình, nhật ký thực nghiệm, manifests và score receipts trong workspace. Nội dung tập trung giải thích những quyết định quan trọng, kết quả đã đạt, các giới hạn còn lại và căn cứ cho bước phát triển tiếp theo.

| Mốc hiện tại | Giá trị đã ghi nhận |
| --- | --- |
| Bản tốt nhất đã có receipt | Cap-10 evidence-window |
| FINAL / DOCS_F2 / CHUNKS_F2 | 0,2324 / 0,2788 / 0,1860 |
| Baseline đối chiếu | S2: FINAL 0,2318 |
| Mô hình trong bản tốt nhất | BAAI/bge-m3 + BAAI/bge-reranker-v2-m3 gốc |
| Quy mô submission | 1.200 queries • 140.095 documents • 72.000 chunks |

Nguồn điểm: kết quả BTC do người dùng cung cấp, được lưu thành receipt cục bộ; báo cáo này không truy vấn lại hệ thống BTC. SHA-256 của ZIP champion đã được kiểm tra khớp receipt khi lập báo cáo. [1]

---

# TÓM TẮT ĐIỀU HÀNH

Medical-RAG đã xây dựng được pipeline truy xuất nhiều tầng, từ chuẩn bị corpus, lập chỉ mục sparse/dense, hợp nhất kết quả, reranking đến chọn tài liệu và trích xuất evidence nguyên văn. Trọng tâm thực nghiệm hiện tại là trả về đúng document IDs và các đoạn bằng chứng theo hợp đồng cuộc thi. Chất lượng sinh câu trả lời bằng LLM chưa phải thành phần được chấm ở các kết quả trình bày trong báo cáo.

Lựa chọn chính là BGE-M3 cho dense retrieval, BM25 cho đối sánh từ khóa và BGE reranker cho đánh giá lại từng cặp query–passage. Benchmark ViMed trước đây cho thấy hybrid + reranker nâng Recall@5 lên 0,8864 so với 0,6919 của BM25; điều đó hỗ trợ hướng kiến trúc nhiều tầng. Tuy nhiên benchmark này khác corpus và nhãn của cuộc thi, nên không dùng điểm ViMed để dự báo điểm V4. [2]

Trên bài toán 1.200 query, FINAL đi từ A0 0,2032 lên S2 0,2318, sau đó lên cap-10 0,2324. Cải thiện lớn đến từ chấm bổ sung passages, mở ngữ cảnh và điều chỉnh số documents theo query. Cải thiện mới nhất chỉ tăng CHUNKS_F2 thêm 0,0012; DOCS_F2 giữ nguyên vì cap-10 không thay danh sách documents. [1, 3, 4, 5]

Điểm hiện tại còn thấp vì hệ thống vừa bỏ sót tài liệu/evidence đúng, vừa xuất nhiều kết quả chưa phù hợp. Những nguyên nhân có bằng chứng gồm cắt ứng viên quá sớm, chưa khảo sát đủ passages trong parent, khóa context trước khi so alternatives, cấp slot evidence phụ thuộc nhiều vào document ranking và thiếu judgments trực tiếp để học đúng relevance. Chưa có đủ gold/scorer để quy phần trăm tổn thất chính thức cho từng nguyên nhân. [6]

> Ưu tiên tiếp theo: củng cố judgments và đo mất mát theo stage; thử coverage và evidence selection với ngân sách khóa trước; chỉ thay champion sau khi một challenger vượt gate và có kết quả chấm tương ứng.

| Báo cáo trả lời | Vị trí |
| --- | --- |
| Mục tiêu, dữ liệu và lịch sử phát triển | Mục 1 |
| Chọn model gì, vì sao; kỹ thuật và workflow | Mục 2–4 |
| Kết quả và các vòng thử nghiệm | Mục 5–6 |
| Kiến thức, khó khăn, phần đã sửa và chưa sửa | Mục 7–9 |
| Vì sao điểm thấp và kế hoạch cải thiện | Mục 10–11 |
| Thông điệp trình bày và nguồn bằng chứng | Mục 12 và phụ lục |

---

# 1. BÀI TOÁN, DỮ LIỆU VÀ QUÁ TRÌNH PHÁT TRIỂN

## 1.1. Mục tiêu thực tế

Đầu vào là câu hỏi, chủ yếu theo hướng truy xuất y khoa tiếng Việt trên nguồn đa ngôn ngữ Việt/Anh/Trung. Đầu ra thi đấu gồm danh sách tài liệu liên quan và những đoạn text làm bằng chứng. Hai quyết định cần tối ưu riêng: tài liệu có thực sự trả lời nhu cầu thông tin hay không, và đoạn nào trong tài liệu chứa evidence đủ để được matching.

Parent là tài liệu nguồn; passage là đơn vị được lập chỉ mục/chấm; chunk/context là đoạn thực sự xuất trong submission. Một document đúng có thể có nhiều passages sai trọng tâm. Vì vậy tìm đúng parent mới là một điều kiện, chưa đảm bảo chunk đúng.

## 1.2. Hai giai đoạn dữ liệu cần phân biệt

| Giai đoạn | Quy mô và vai trò | Giới hạn cần nhớ |
| --- | --- | --- |
| ViMed, baseline 30/09 | 17.955 chunks; 2.210 validation queries. Kiểm chứng BM25 → dense → hybrid → reranker. | Qrels một-positive theo context nguồn; không phải toàn bộ relevance. |
| V4 / ViBioMIR, tháng 10 | 1.073.525 eligible sources; 2.951.224 passages; 1.200 query thi đấu; vectors 1.024 chiều. | Full eligible index không bao phủ mọi official URL. |
| Dữ liệu chẩn đoán | 2.190 historical matched outputs và 87 manual evidence. | Không phải complete gold; manual cohort đã được dùng qua nhiều vòng. |

Tổng official URLs theo review là 1.220.598; 147.067 chưa Phase1 PASS và 6 semantic exclusions không vào eligible release. Chưa biết lượng gold nằm trong phần thiếu. Không thể quy toàn bộ điểm thấp cho coverage corpus, cũng không thể gọi index là đủ mọi URL chính thức. [2, 6, 7]

## 1.3. Các mốc quan trọng

Tháng 9: kiểm chứng retrieval trên ViMed. Ngày 21/10: nghiệm thu full index với checksum, mapping, offsets và smoke. Ngày 23–24/10: hoàn tất A0, supplemental reranking và các bản refined/S1/S2. Ngày 25/10: review lại tính đúng của nhãn, offsets, split và gate. Ngày 26/10: cap-10 trở thành champion; audit 87 evidence và thử document scorer mới; end-to-end recovery được khóa policy nhưng chưa có full run. [2–12]

---

# 2. LỰA CHỌN MÔ HÌNH VÀ LÝ DO

## 2.1. BGE-M3 cho dense retrieval

BAAI/bge-m3 là encoder được dùng để biểu diễn query và passage thành vector. Model card mô tả hỗ trợ đa ngôn ngữ và nhiều cơ chế retrieval; pipeline champion hiện dùng nhánh dense của nó, kết hợp BM25 riêng. Không nên mô tả hệ thống đang dùng cả learned sparse và multi-vector của BGE-M3 khi config không triển khai các nhánh đó. [13]

Cơ sở chọn: phù hợp định hướng đa ngôn ngữ; passage vectors có thể tính trước và tái sử dụng; có revision được pin để tái lập; và có kết quả ViMed hỗ trợ giá trị của dense so với BM25. Đây là lựa chọn có căn cứ cho baseline hiện tại, chưa phải bằng chứng BGE-M3 tốt nhất trong tất cả model có thể dùng.

## 2.2. BGE reranker-v2-m3 cho reranking

BAAI/bge-reranker-v2-m3 đọc trực tiếp cặp query–passage và trả relevance score, thay vì chỉ so hai embeddings đã tính độc lập. Vai trò là phân biệt tốt hơn các ứng viên sau retrieval. Lý do giữ model này gồm khả năng đa ngôn ngữ, tích hợp đã ổn định, score cache lớn và benchmark ViMed cho thấy reranking cải thiện rõ top ranks. [2, 14]

Champion dùng reranker gốc, không dùng trọng số R3 fine-tuned. R3 có AUC/AP validation cao hơn nhưng các selector áp dụng thực tế làm mất manual positives; R5 còn có vấn đề chuyển nhãn từ text cũ sang text mới. Vì vậy chưa có bằng chứng đủ để đưa R3 vào champion. [8, 9]

| Thành phần | Cấu hình thực tế | Điểm cần giải thích |
| --- | --- | --- |
| Dense encoder | bge-m3; max_length 512; embedding 1.024 chiều. | 512 là lựa chọn triển khai hiện tại, không phải toàn bộ khả năng dài nhất của model. |
| Cross-encoder | bge-reranker-v2-m3 gốc; max_length 512. | Không thể chọn đúng evidence nếu evidence không nằm trong phần input được nhìn. |
| Lexical retrieval | Document BM25 + passage BM25. | Bổ sung tín hiệu từ khóa; không phải model sinh văn bản. |
| R3 / logistic document scorer | Thử nghiệm, chưa được promotion. | Tốt hơn trên proxy không đủ để thay model đang có điểm BTC. |

> Báo cáo phân biệt lý do kỹ thuật để chọn model và kết quả đã đo. Repository chưa cung cấp một so sánh có kiểm soát đầy đủ để khẳng định đây là bộ model tối ưu cho toàn corpus thi đấu.

---

# 3. CÁC KỸ THUẬT ĐÃ SỬ DỤNG

| Kỹ thuật | Cách dùng trong dự án | Mục đích / giới hạn |
| --- | --- | --- |
| Chuẩn hóa corpus, provenance | Gắn source identity, hash, passage mapping và original offsets; kiểm eligible release. | Biết text đến từ đâu và slice có khớp source hay không. |
| Phân cấp document–passage | Lập index theo tài liệu và passage; gom về parent khi chọn output. | Giảm không gian xử lý; có thể bỏ mất vùng evidence nếu cắt sớm. |
| Hybrid retrieval | Document BM25 400 + passage BM25 200 + dense passages 200 ở precursor. | Kết hợp từ khóa và ngữ nghĩa; các budgets không cộng thành số parents duy nhất. |
| Reciprocal Rank Fusion | RRF k = 60; cắt top-400 mixed items trước khi gom parent. | Hợp nhất theo thứ hạng; mixed-item cap có thể loại parent còn hữu ích. |
| Parent-first reranking | Precursor chấm 200 pairs/query; supplemental chấm các cached passages còn thiếu. | Mở rộng độ phủ scoring; không đồng nghĩa đã chấm mọi passage của mọi parent. |
| Context expansion | Lấy vùng lân cận, từ cap 4.096 đến refinement 5.500 ký tự. | Giữ evidence nguyên văn; phần export có thể dài hơn input scorer. |
| Adaptive document selection | Pre-gate -4; đếm scores ≥ -2, clip target 80–140; bảo vệ chunk parents. | Điều chỉnh lượng docs theo query; đây là policy, không phải xác suất relevance đã calibrated. |
| Evidence-window cap-10 | Tối đa ba windows head/middle/tail, body 192 tokens; max logit; tối đa 10 swaps/query. | Tách tín hiệu cấp slot khỏi parent prefix; vẫn giữ fixed context. |
| Fine-tuning, weak supervision | R3 supervised reranker; logistic multi-passage document features. | Đã thử, chưa trở thành champion; nhãn và representation là giới hạn chính. |
| Caching, parity, resume, validation | Khóa exact input identity; kiểm scores, cardinality, ZIP, parent membership, hashes. | Giảm chi phí lặp và ngăn sai artifact; không chứng minh relevance tăng. |

RRF có dạng tổng 1/(k + rank) trên các nhánh có ứng viên. Nó kết hợp thứ hạng mà không đòi raw BM25 score và dense similarity cùng thang đo. Sau reranking, quyết định chọn phải dựa trên score/policy tương ứng, tránh trộn score khác nghĩa. [7, 9]

Hai giới hạn khác nhau cần phân biệt: 512 là token budget của cặp scorer; 5.500 là character cap của context export. Tăng ký tự export không làm reranker nhìn thêm evidence ngoài input của nó. [6, 7]

---

# 4. WORKFLOW THỰC TẾ CỦA CHAMPION

Cap-10 kế thừa retrieval, refined candidates và document lists từ S2, sau đó thay cơ chế cấp chunk slots trong phạm vi documents đã chọn. Nó không phải một lần chạy lại default CLI/API. [1, 5, 7]

| Bước | Luồng xử lý | Đầu ra / điểm kiểm |
| --- | --- | --- |
| 01 • OFFLINE | Official sources → eligible release → registered passages → sparse/dense indexes. | Khóa corpus/model identity, mapping và source offsets. |
| ↓ 02 • RETRIEVE | Query → document BM25 + passage BM25 + BGE-M3 dense. | Giữ candidate identities và branch ranks khi có trace. |
| ↓ 03 • FUSE | RRF top-400 mixed document/passage items → group parents. | Đây là một cửa cắt có thể mất parent. |
| ↓ 04 • RERANK | 200 pairs/query ban đầu → bổ sung cached passages còn thiếu. | 393.401 pairs = 240.000 cũ + 153.401 bổ sung ở refinement. |
| ↓ 05 • BUILD S2 | Rank parents → chọn seed/context → expand → adaptive documents. | 140.095 documents; 60 chunks/query trước cap-10. |
| ↓ 06 • WINDOW SCORE | Trong S2 docs, giữ một fixed context/parent → tối đa ba windows → BGE reranker. | 418.751 window pairs toàn bộ package cap-10; pair input ≤ 512 tokens. |
| ↓ 07 • ALLOCATE | Max window logit/parent → so entrant/victim → tối đa 10 chunk-parent swaps/query. | Document lists giữ nguyên; export full fixed context, không export riêng window ngắn. |
| ↓ 08 • EXPORT | Kiểm source, IDs, 60 slots, một chunk/parent, parent membership và ZIP. | 1.200 queries; 72.000 chunks; ZIP 87.350.407 bytes. |
| ↓ 09 • SCORE | Gắn đúng ZIP hash với receipt BTC. | FINAL 0,2324; chỉ promotion đúng artifact đã được chấm. |

## Điểm khác giữa workflow hiện có và hướng đề xuất

Workflow mới dự kiến sẽ mở thêm passages trong parent, sinh nhiều windows có coverage đo được, rồi chọn document relevance và evidence quality theo hai mục tiêu riêng. Các thay đổi này chưa nằm trọn trong champion hiện tại. Một candidate depth 800 đã được khóa policy, nhưng chưa có retrieval/scoring đầy đủ hay submission. [6, 12]

---

# 5. KẾT QUẢ THỰC NGHIỆM VÀ CÁCH ĐỌC ĐIỂM

## 5.1. Benchmark ViMed: căn cứ ban đầu cho kiến trúc

| Stage | Recall@1 | Recall@5 | Recall@100 | MRR@100 |
| --- | --- | --- | --- | --- |
| BM25 | 0,4905 | 0,6919 | 0,8837 | 0,5839 |
| Dense BGE-M3 | 0,5566 | 0,7661 | 0,9222 | 0,6504 |
| Hybrid RRF | 0,5480 | 0,7719 | 0,9249 | 0,6498 |
| Hybrid + reranker | 0,7493 | 0,8864 | 0,9249 | 0,8091 |

Trên benchmark này, reranker cải thiện vị trí của positives đã vào pool nhưng không tăng Recall@100: những positives đã bị retrieval/fusion bỏ vẫn không được phục hồi. Đây là bài học quan trọng khi chuyển sang full corpus. Kết quả dùng 2.210 validation queries và qrels theo source-context, khác protocol thi đấu. [2]

## 5.2. Các mốc BTC được ghi nhận trong dự án

| Phiên bản | FINAL | DOCS_F2 | CHUNKS_F2 | Thay đổi chính |
| --- | --- | --- | --- | --- |
| A0 | 0,2032 | 0,2289 | 0,1775 | Mốc trước quality refinement |
| Rerank refined | 0,2274 | 0,2699 | 0,1848 | Supplemental rerank + gate/context |
| S1 fixed-120 | 0,2291 | 0,2735 | 0,1848 | Giảm document cardinality |
| S2 adaptive | 0,2318 | 0,2788 | 0,1848 | Target số docs theo score-count |
| R4 high-confidence | 0,2316 | — | — | Không vượt S2 |
| Cap-10 • champion | 0,2324 | 0,2788 | 0,1860 | Thay chunk slots; giữ docs S2 |

Các điểm BTC trên được dẫn từ receipts/báo cáo lưu kết quả người dùng cung cấp. Dấu “—” nghĩa là không đưa số thành phần vào bảng này vì chưa đối chiếu riêng, không phải bằng 0. [1, 3, 4, 5, 7, 8]

## 5.3. Ý nghĩa của cải thiện

Từ A0 đến cap-10, FINAL tăng tuyệt đối 0,0292. Riêng S2 → cap-10 tăng 0,0006, trong khi đổi 11.445/72.000 chunk slots trên 1.181 query. Mức thay đổi output lớn nhưng gain nhỏ cho thấy tín hiệu chọn evidence hiện tại chưa phân biệt đủ tốt. Không có kiểm định độ bất định trong artifacts đã đọc để khẳng định ý nghĩa thống kê của mức tăng này.

Theo cách tổng hợp ở các receipts, FINAL = (DOCS_F2MACRO + CHUNKS_F2MACRO)/2. F2 của mỗi đơn vị đánh giá có dạng 5PR/(4P + R), nhưng không được lấy precision/recall trung bình rồi thế vào để suy ra macro F2. Macro phải được tổng hợp theo đúng đơn vị/scorer của BTC. [1, 6]

---

# 6. NHỮNG VÒNG THỬ NGHIỆM ĐÃ DẠY ĐIỀU GÌ?

| Vòng / trạng thái cuối | Kết quả quan trọng | Bài học để giữ lại |
| --- | --- | --- |
| R3 • chưa promotion | AUC 0,5259 → 0,7079; AP 0,0763 → 0,1698 trên validation lịch sử. Selector blend vẫn làm mất manual positives. | Ranking metric tốt hơn không tự bảo đảm output tốt hơn. Phải kiểm nhãn, input và selection policy. |
| R4 • rejected | Thêm 180.000 passages; arm high-confidence đã submit nhưng FINAL 0,2316, thấp hơn S2. | Candidate expansion + proxy gain nhỏ chưa đủ để suy BTC gain. |
| R5 • NO_GO / audit | Same-input comparison từng dùng nhãn old chunk chuyển theo parent sang context mới. | So hai model trên cùng input vẫn sai mục tiêu nếu nhãn không thuộc input đó. |
| R6 • diagnostic | Cho thấy cơ hội ở coverage, context và allocation; review phát hiện sai mapping source offsets. | Oracle/preservation phải đúng ràng buộc và không được gọi là gold recall. |
| R8 • NO_GO | Lexical extractor/allocator gần như không thay output; development policies không đủ điều kiện. | Một treatment lexical thất bại chưa bác bỏ evidence selection nói chung. |
| R7 targeted • NO_GO | Thu hồi được một số parents vào docs nhưng vẫn không nhận chunk slot; downstream control doc loss. | Sửa upstream phải kiểm downstream; thu hồi parent chưa giải quyết evidence. |
| Cap-10 • CHAMPION | 418.751 window pairs; +0,0012 CHUNKS_F2; documents không đổi. | Evidence allocation có tác dụng đo được, nhưng gain còn nhỏ. |
| Manual87 recovery • pool only | Pool có 87/87 manual parents và full spans; 259.327 passages. | Khôi phục pool là bước đầu, không phải đã có final recall 100%. |
| R11 • STOP_BUDGET | Canary dự phóng raw-only 22.792,9 CPU-s cho 94 queries, vượt 14.400 CPU-s. | Dừng do chi phí; coverage gate chưa đo, không kết luận quality fail. |
| Document multi-passage • NO_GO | 2.788 document swaps trên 1.066 queries; validation +1 gain / 1 loss. | Thêm aggregation features chưa tạo tín hiệu relevance đủ tốt. |
| End-to-end recovery • đang chờ | Policy depth 800 đã khóa; status WAITING_FOR_CUDA_HOST; chưa có candidate ZIP. | Technical parity PASS không tương đương đã chạy hay đã tăng BTC score. |

Bảng ghi trạng thái mới nhất được đối chiếu đến 26/10; các báo cáo cũ có thể mô tả thời điểm “chờ chấm” trước khi có receipt mới. Không gộp các trạng thái theo thời điểm khác nhau thành một kết quả duy nhất. [5–12]

---

# 7. KIẾN THỨC CẦN NẮM VỮNG SAU DỰ ÁN

## 7.1. Retrieval, ranking và selection là ba bài toán khác nhau

Retrieval đưa evidence vào candidate pool; ranking đánh thứ tự những gì đã vào pool; selection quyết định xuất bao nhiêu và cấp slot cho ai. Reranker mạnh không cứu được candidate bị cắt trước đó. Pool đầy đủ cũng chưa đủ nếu ranker chọn nhầm hoặc slot allocator bỏ evidence đúng. Vì vậy cần coverage và loss attribution sau từng bước.

## 7.2. Relevance ở document khác relevance ở evidence

Một positive span có thể chứng minh parent chứa nội dung trả lời. Một negative passage không chứng minh toàn parent irrelevant vì vùng khác có thể đúng. Tương tự, document nói cùng chủ đề chưa chắc trả lời đúng intent, vị trí, tần suất hay đối tượng trong câu hỏi. Bộ nhãn cần có hard negatives cùng chủ đề nhưng sai nhu cầu thông tin.

## 7.3. Nhãn phải gắn với đúng text được đánh giá

Identity tối thiểu gồm query, document và exact text hash; scoring cache còn cần passage/window identity và đúng input hash. Nếu context được mở rộng hoặc đổi seed, nhãn text cũ không tự chuyển sang text mới. Unjudged là chưa biết, không mặc nhiên negative. Re-audit cho thấy chỉ 4/72.000 S2 chunks khớp nguyên text historical judgments. [9, 10]

## 7.4. Proxy, oracle và official metric có vai trò riêng

Historical anchor preservation đo giữ lại output từng match; manual containment đo một cohort evidence cụ thể; oracle cho biết trần khả thi của pool dưới ràng buộc giả định. Chúng hỗ trợ chẩn đoán nhưng không thay DOCS_F2/CHUNKS_F2 chính thức. Nếu oracle chọn một context/parent thì phải thực thi đúng ràng buộc đó, tránh cộng evidence thuộc nhiều contexts không thể xuất đồng thời.

## 7.5. Token, ký tự và offsets không thể thay thế nhau

Tokenizer quyết định model được nhìn gì; character cap quyết định text được export. NFKC và HTML-unescape có thể làm đổi số ký tự nên offset trên chuỗi normalized không dùng trực tiếp để slice source gốc. Cần mapping về original spans và round-trip assertion. Lỗi audit offsets lịch sử không tự chứng minh source slices của champion sai. [9, 10]

## 7.6. Split và gate là một phần của phương pháp thực nghiệm

Phải tách query theo development/validation/held-out xuyên suốt train và policy selection. Khóa policy trước validation; ghi gross gains và gross losses riêng để net gain không che lỗi. Khi không arm nào đủ điều kiện, kết quả hợp lệ là không chọn arm. Registry mới gồm 789/210/201 queries; historical labels đã được xem nên held-out này không phải benchmark mới hoàn toàn chưa đụng tới. [10]

---

# 8. KHÓ KHĂN ĐÃ GẶP VÀ NHỮNG PHẦN ĐÃ KHẮC PHỤC

| Khó khăn | Cách đã xử lý | Bằng chứng / giới hạn |
| --- | --- | --- |
| Corpus lớn, dễ lệch vector–passage–source. | Nghiệm thu full eligible index; rehash vectors, mapping, metadata và grounding. | 2.951.224 rows / 1.073.525 parents PASS theo acceptance 21/10; không encode lại trong báo cáo này. |
| Rerank budget ban đầu bỏ qua cached passages. | Chấm bổ sung 153.401 pairs, merge bằng exact identity và kiểm coverage. | Tổng 393.401 pairs ở refinement; chưa mở mọi passage ngoài cached pool. |
| Xuất gần 200 docs/query tạo nhiều false positives. | Thử gate, S1 cap-120, rồi S2 adaptive 80–140 với chunk-parent protection. | DOCS_F2 từ 0,2289 ở A0 lên 0,2788 ở S2; đây là tác động gộp các thay đổi. |
| Chỉ cấp chunk theo document prefix. | Cap-10 dùng window score để hoán đổi tối đa 10 slots trong docs S2. | CHUNKS_F2 0,1848 → 0,1860; mức giải quyết còn hạn chế. |
| Chuyển nhãn old text sang context mới. | Khóa exact-identity labels; text thay đổi để unjudged nếu chưa chấm trực tiếp. | F01 đã sửa; phần lớn S2 chunks vẫn thiếu exact historical judgments. |
| Normalization khiến audit offsets sai. | Giữ mapping về source gốc, round-trip và re-audit. | 320 slices sai → 0/1.933 slices được audit; token visibility cũ chưa tính lại. |
| Development không có policy đạt mà vẫn chọn. | Fail-closed, selected policy null; tách gross gain/loss và runtime checks. | F06 đã sửa; không biến structural PASS thành quality PASS. |
| Split thay đổi giữa các vòng. | Lập registry chung và kiểm train/eval membership. | Audit mới: 335 R3 train queries / 0 overlap với evaluation mới. |
| Không biết manual evidence thiếu do index hay downstream. | Audit 87 spans; mở passages và retrieval diagnostic cho 11 query thiếu parent. | 87/87 vào recovery pool; chưa chấm/chọn toàn bộ pool. |
| Khó gắn điểm với đúng output. | Lưu receipt, ZIP SHA-256, payload hash và provenance. | Khi lập báo cáo đã kiểm ZIP cap-10 khớp receipt; baseline S2 được giữ riêng. |

Các mốc kiểm thử lịch sử hỗ trợ độ tin cậy phần triển khai: đợt evaluation fixes ghi 63 targeted tests PASS và 293 tests PASS khi bỏ API-tail suite đang treo. Báo cáo này không chạy lại inference/training hoặc toàn bộ tests, nên không coi những số đó là kết quả test mới ngày 30/10. [3, 5, 7, 9, 10, 11]

---

# 9. NHỮNG VẤN ĐỀ CHƯA KHẮC PHỤC HOÀN TOÀN

## 9.1. Coverage và quyết định chọn evidence

Mixed document/passage top-400 vẫn có thể cắt parent trước khi gom. Parent đã vào pool cũng chưa bảo đảm passage đúng đã được khảo sát. Cap-10 chỉ nhìn tối đa ba windows trong một fixed context; không tự mở mọi vùng của document và không sửa được context đã khóa sai. Số 87/87 recovery pool chưa được chuyển thành 87/87 output. [6, 11]

## 9.2. Chưa có relevance signal đủ mạnh cho hai tầng

Document score vẫn chủ yếu dựa trên max passage logit; evidence scorer vẫn là generic BGE reranker. Logistic multi-passage scorer thêm 12 features nhưng validation net gain bằng 0. Nó còn dùng 207.477 unlabeled rows làm target 0 với weight 0,02 trong training; tổng weight khoảng 4.149,54 so với 1.254 positives. Đây là weak supervision có ảnh hưởng đáng kể, chưa phải xác suất relevance được hiệu chuẩn. [6, 8]

## 9.3. Thiếu judgments trực tiếp và đánh giá phù hợp

Packet document review có 1.554 rows / 300 queries đang chờ judgments trực tiếp. Historical positives không liệt kê mọi relevant document; negative một chunk không phải negative cả document. Cần đánh giá exact context/span mới, giữ unjudged đúng nghĩa và thêm mẫu nền ngoài entrant/victim swaps. Bộ pilot 13.546 contexts cũ đã dừng trước annotation; không giao lại bộ đó như công việc đang cần làm. [6, 8, 10]

## 9.4. Vận hành, tái lập và cấu hình

Candidate end-to-end đang chờ CUDA host theo status lưu ngày 26/10; chưa có full retrieval, reranking hay ZIP. R11 đã dừng tại cost gate. Chưa có benchmark full GPU path để hứa thời gian chạy. Effective config champion vẫn trải qua YAML, Python và manifests; default demo CLI/API không tái tạo champion. Precursor YAML ghi ZIP cap 50 MB còn exporter sau refinement dùng 100 MB theo xác nhận đã lưu. [7, 12]

## 9.5. Những điều chưa thể kết luận

Chưa biết bao nhiêu gold nằm ngoài eligible corpus. Chưa biết đầy đủ RAW_PARENT_MISS so với FUSION_DROP khi thiếu raw traces. Token visibility theo corrected offsets chưa được tính lại. Hai API-tail tests được ghi nhận treo ở vòng kiểm tra trước, chưa có bằng chứng đã xử lý. Các giới hạn này cần giữ rõ để tránh dùng kết quả chẩn đoán vượt quá phạm vi. [6, 9, 10]

> “Đã viết runner”, “technical parity PASS”, “có candidate pool” và “đã có điểm BTC tốt hơn” là bốn mức tiến độ khác nhau. Hiện chỉ cap-10 có receipt làm căn cứ champion mới nhất.

---

# 10. VÌ SAO ĐIỂM HIỆN TẠI VẪN THẤP?

## 10.1. Cả precision và recall đều còn hạn chế

| Tầng đánh giá | Precision | Recall | F2 macro |
| --- | --- | --- | --- |
| Documents | 0,1946 | 0,3695 | 0,2788 |
| Chunks | 0,2084 | 0,2117 | 0,1860 |

Các chỉ số cho thấy còn nhiều kết quả sai và nhiều gold chưa được tìm đúng. Chúng không chỉ ra chính xác phần trăm loss do model, fusion hay context. Tăng lượng output có thể tăng recall nhưng cũng thêm false positives; giảm output có thể tăng precision nhưng mất positives. Vì vậy cần lựa chọn cardinality theo relevance đã được đánh giá. [1, 6]

## 10.2. Evidence tồn tại nhưng không đi hết pipeline

| Kiểm tra trong cohort manual | Có mặt / 87 |
| --- | --- |
| Source hiện hành / passage index chứa trọn evidence | 87 / 87 |
| Parent / passage chứa trọn evidence trong refined pool | 76 / 71 |
| Parent có trong documents / được cấp chunk slot | 76 / 67 |
| Fixed context chứa evidence / final chunk chứa đủ evidence | 73 / 65 |
| Recovery pool sau mở rộng có full span | 87 |

22 final misses được chia không trùng: 11 thiếu parent trong pool, 9 có document nhưng thiếu slot và 2 có slot nhưng sai context. Các hàng bảng không tạo thành chuỗi tập con thuần túy vì context expansion có thể lấy evidence ngoài seed. Cohort đã dùng chẩn đoán, nên tỷ lệ 65/87 không phải official recall hoặc ước lượng đại diện cho toàn bộ queries. [6]

## 10.3. Những cải tiến gần đây thường tối ưu proxy hoặc một tầng

S1/S2 chỉ đổi documents nên không tăng CHUNKS_F2. Cap-10 chỉ đổi chunk allocation trong docs S2 nên DOCS_F2 đứng yên; fixed context và generic scorer vẫn giữ giới hạn cũ. R3 AUC/AP tăng nhưng selector mất positives. R4 proxy tăng nhỏ nhưng BTC giảm. Bởi vậy chưa thể lấy tăng local metric làm bằng chứng final quality tăng. [4, 5, 8, 9]

## 10.4. Mức cải thiện cần thiết lớn hơn một lần chỉnh cap

Để FINAL đạt 0,4 theo cách tổng hợp hiện tại, tổng DOCS_F2 + CHUNKS_F2 cần 0,8; hiện là 0,4648, còn chênh 0,3352. Nếu chỉ tăng chunks và giữ docs 0,2788 thì chunks phải đạt 0,5212; nếu chỉ tăng docs và giữ chunks 0,1860 thì docs phải đạt 0,6140. Đây là phép tính mục tiêu, không phải dự báo khả năng đạt. Chưa có căn cứ cam kết một thay đổi đơn lẻ đưa hệ thống tới 0,4.

---

# 11. LỘ TRÌNH CẢI THIỆN CÓ CĂN CỨ

Lộ trình dưới đây là đề xuất cho giai đoạn tiếp theo, chưa phải công việc đã thực thi. Thứ tự dựa vào các nút thắt đo được, tránh tiếp tục đổi nhiều biến mà không biết yếu tố nào tạo gain.

| Ưu tiên | Việc cần làm | Điều kiện hoàn tất |
| --- | --- | --- |
| P0 • baseline và đo lường | Đóng gói effective config cap-10; manifest code/model/data; trace source → raw branches → fusion → parent → passage → window → slot. | Replay khớp champion; stage thiếu trace ghi UNKNOWN; labels theo exact identity; policy khóa trước validation. |
| P0 • direct judgments | Chấm packet document 1.554 rows/300 queries, thêm mẫu nền; chấm exact windows/contexts cho evidence. Có hard negatives sai intent/điều kiện. | Judgments có provenance và review ca mơ hồ; unjudged không dùng như negative gold; báo riêng document/evidence metrics. |
| P1 • coverage trước scorer | So arm mở passages trong hierarchy và arm bảo lưu passages mạnh; parent budget riêng; cheap shortlist rồi cross-encoder. | Đo coverage sau mỗi prune, với cùng split/corpus/pair budget. Không dùng manual IDs hoặc evidence text để chọn inference candidates. |
| P1 • document/evidence relevance | Huấn luyện hoặc calibration hai mục tiêu riêng; có thể chia sẻ backbone. So context alternatives trước khi khóa export. | Gain trên validation có direct judgments; gross losses trong budget; document precision và evidence quality được kiểm riêng. |
| P1 • GPU candidate đã khóa | Nếu chạy depth-800 end-to-end thì giữ policy hiện có, mount đủ index và dùng runtime CUDA phù hợp. | Có actual retrieval/scoring traces, source-grounded export, ZIP validation và receipt; không đổi policy theo manual87 rồi gọi là cùng experiment. |
| P2 • hiệu năng và mở rộng | Benchmark sparse aggregate/batched dense, caching và shortlist; cân nhắc query variants, thuật ngữ đa ngôn ngữ sau khi có cohort lỗi. | Ghi CPU/GPU time, pair count, memory và quality/cost; tối ưu nhanh phải kiểm parity hoặc đo trade-off. |

## Một vòng thử nghiệm nên trả lời một câu hỏi

Ví dụ: “Mở rộng passages trong các parents đã có, dưới cùng ngân sách rerank, có tăng evidence coverage và final relevance không?” Giữ các biến khác cố định, đo candidate ceiling trước, khóa scorer/allocator rồi mới mở validation. Nếu pool đã có evidence nhưng scorer chọn sai, ưu tiên nhãn và relevance model; nếu evidence chưa vào pool, ưu tiên coverage. [6]

Mỗi challenger cần lưu treatment, split, input hashes, model revision, score representation, gross gain/loss, chi phí và quyết định GO/NO_GO. Chỉ chạy full submission khi kết quả pilot đủ căn cứ; chỉ promotion đúng artifact có kết quả chấm phù hợp.

---

# 12. KẾT LUẬN VÀ GỢI Ý TRÌNH BÀY

## 12.1. Kết luận dự án đến thời điểm hiện tại

Dự án đã hoàn thành được hạ tầng retrieval có khả năng xử lý corpus lớn, kết hợp lexical và semantic search, reranking, source-grounded context export và submission validation. Điểm BTC được ghi nhận đã tăng từ A0 0,2032 lên cap-10 0,2324. Cùng với tăng điểm, dự án đã sửa các lỗi đánh giá quan trọng về text identity, normalization offsets và development gate.

Phần chưa giải quyết đầy đủ là chất lượng relevance và khả năng đưa evidence qua toàn bộ pipeline. Chọn đúng model là nền tảng, nhưng chưa thay thế việc bảo toàn candidate coverage, khảo sát đủ nội dung trong parent, chọn đúng context và phân bổ 60 slots. Thiếu judgments đúng đối tượng khiến nhiều vòng tối ưu vẫn dựa vào proxy, nên mức đầu tư compute chưa chuyển thành gain tương ứng.

Hướng tiếp theo có căn cứ là giữ cap-10 làm baseline, hoàn thiện direct judgments và stage traces, so hai cơ chế giữ evidence với cùng ngân sách, rồi học/calibrate document và evidence relevance riêng. Candidate end-to-end đã chuẩn bị cần được chạy và đánh giá như một treatment khóa trước; hiện chưa có kết quả để tuyên bố nó tốt hơn champion.

## 12.2. Mạch trình bày khi báo cáo với giảng viên / nhóm

| Phần trình bày | Thông điệp chính |
| --- | --- |
| Bài toán và dữ liệu | Cần đúng tài liệu lẫn đoạn bằng chứng; full eligible corpus lớn hơn rất nhiều benchmark ViMed. |
| Model và phương pháp | BM25 + BGE-M3 + RRF + BGE reranker; lựa chọn dựa trên nhu cầu đa ngôn ngữ, tái sử dụng index và benchmark nội bộ. |
| Kết quả | Nêu A0 → refined → S2 → cap-10; giải thích phần gain đến từ documents hay chunks. |
| Khó khăn và điều đã sửa | Trình bày một lỗi nhãn, một lỗi offset, một lỗi gate; nêu phạm vi đã xác minh. |
| Vì sao còn thấp | Dùng 87 manual evidence minh họa ba cửa mất: candidate pool, slot và context. |
| Bước tiếp theo | Direct judgments + coverage traces + relevance objectives riêng; chốt điều kiện để promotion. |

> Câu chốt có thể dùng: “Sau dự án, vấn đề quan trọng nhất tôi nhận ra là phải đo đúng evidence mất ở đâu. Tăng score hay thêm model chỉ có ý nghĩa khi đoạn trả lời đúng thực sự đến được đầu ra và được xác nhận bằng quy trình đánh giá phù hợp.”

---

# PHỤ LỤC A. NGUỒN TÀI LIỆU VÀ PHẠM VI KIỂM CHỨNG

Đường dẫn dưới đây tính từ repository medical-rag/medical-rag. Nguồn local là bằng chứng thực nghiệm của dự án; model cards chỉ dùng giải thích đặc tính và vai trò mô hình, không thay bằng chứng chất lượng trên corpus cuộc thi.

## [1] Champion cap-10: receipt và execution

outputs/v4/evidence_window_cap10_full_20261025/scoring_receipt_user_20261026.json

docs/v4_evidence_window_cap10_full_execution_20261026_vi.md

## [2] ViMed: baseline tham chiếu và review trước V4

archive/docs_pre_v4/baseline_reference_vi.md

archive/docs_pre_v4/tong-quat.md

## [3] A0: quality fixes và supplemental reranking

docs/v4_a0_quality_fix_20261024_vi.md

## [4] S1/S2: document selection

docs/v4_s1_s2_document_selection_20261024_vi.md

## [5] Cap-10: validation / held-out và full execution

docs/v4_evidence_window_cap10_validation_execution_20261025_vi.md

docs/v4_evidence_window_cap10_full_execution_20261026_vi.md

## [6] Chẩn đoán DOCS_F2 và CHUNKS_F2 mới nhất

docs/v4_doc_chunk_f2_diagnosis_20261026_vi.md

## [7] Full index, workspace guide và competition config

docs/v4_task4_full_index_execution_result_20261021_vi.md

docs/V4_WORKSPACE_GUIDE.md

configs/v4_submission_p2_a0.yaml

Báo cáo ưu tiên receipt và artifacts mới hơn khi tài liệu mâu thuẫn. README vẫn ghi S2 0,2318 ở thời điểm đọc; workspace guide và receipt ngày 26/10 đã xác nhận cap-10 0,2324. Run manifest cap-10 trước chấm còn ghi UNMEASURED; receipt mới hơn là căn cứ cho trạng thái champion.

---

# PHỤ LỤC A. NGUỒN TÀI LIỆU (TIẾP)

## [8] R3/R4 và document multi-passage challenger

docs/v4_r3_r4_execution_20261024_vi.md

docs/v4_doc_f2_multipassage_execution_20261026_vi.md

outputs/v4/doc_f2_multipassage_20261026/manifest.json

## [9] Review dự án: findings F01–F10 và phạm vi kết luận

docs/v4_project_review_20261025_vi.md

## [10] Evaluation fixes: exact identity, offsets, split và gates

docs/v4_evidence_evaluation_allocation_20261025_vi.md

## [11] Manual87 pool recovery, R7 và R11

docs/v4_manual87_recovery_execution_20261026_vi.md

docs/v4_r7_score_gate_execution_20261025_vi.md

docs/v4_r11_high_recall_coverage_execution_20261026_vi.md

## [12] End-to-end recovery: execution/status, chưa có ZIP

docs/v4_end_to_end_recovery_execution_20261026_vi.md

outputs/v4/end_to_end_recovery_20261026/status.json

## [13] BAAI BGE-M3: model card chính thức, truy cập 30/10/2026

https://huggingface.co/BAAI/bge-m3

## [14] BAAI BGE reranker-v2-m3: model card chính thức, truy cập 30/10/2026

https://huggingface.co/BAAI/bge-reranker-v2-m3

---

# PHỤ LỤC B. NHẬT KÝ ĐỐI CHIẾU KHI LẬP BÁO CÁO

Đã đọc các reports, config, receipts và manifests nêu ở phụ lục A; đọc phần tạo windows và allocation trong scripts/v4/run_evidence_window_allocation.py và full cap-10 runner. Đã kiểm tra status end-to-end để không nhầm “đã khóa policy” với “đã thực thi”.

Ngày 30/10/2026 đã băm trực tiếp ZIP champion, đối chiếu receipt và đếm lại các cờ của all87.csv/missing22.csv. Các kiểm tra này xác nhận liên kết artifact–receipt và số liệu cohort trong báo cáo; không phải một lần chấm BTC hoặc inference mới.

| Đối chiếu | Kết quả |
| --- | --- |
| ZIP champion | outputs/v4/evidence_window_cap10_full_20261025/export/submission.zip |
| SHA-256, dòng 1 | e422b9076bf59f9a93d4819776026e789 |
| SHA-256, dòng 2 | e7cf3f407b3befee8738fc5abb270f8 |
| Hash so receipt | Khớp; nối liền dòng 1 và dòng 2 là SHA-256 đầy đủ. |
| Manual audit | 87 rows; 22 misses; 11 pool / 9 slot / 2 context. |
| Recovery pool | 87/87 parent pairs và 87/87 spans trong CSV coverage. |
| Code HEAD khi đọc | ee7e3434ad0e599fe255c23205d4d495e1ba9531 |
| Workspace | Có thay đổi chưa commit từ trước; HEAD không đại diện toàn bộ working tree. |

Không thực hiện trong lượt lập báo cáo: training, GPU inference, retrieval mới, thay champion, gửi BTC hoặc chạy lại test suite. Các số test/latency được trích từ nhật ký lịch sử và được ghi đúng phạm vi. Không sửa những thay đổi code/docs đã tồn tại trong workspace.

## Từ điển ngắn

| Thuật ngữ | Nghĩa trong báo cáo |
| --- | --- |
| Parent / passage / context | Tài liệu nguồn / đơn vị được index-score / đoạn ngữ cảnh dự kiến xuất. |
| Evidence / anchor | Bằng chứng trả lời / đoạn output lịch sử dùng theo dõi preservation. |
| Champion / challenger | Bản tốt nhất có bằng chứng chấm / bản ứng viên đang thử để so sánh. |
| GO / NO_GO / STOP_BUDGET | Cho phép sang bước tiếp / không vượt quality gate / dừng do trần chi phí. |
| Unjudged / held-out | Chưa được chấm nhãn / phần tách để đánh giá theo protocol đã khóa. |

---

# Hướng cải thiện ViBioMIR — 07/10/2026

## Kết luận từ điểm và artifacts hiện có

Người dùng cho biết ảnh điểm là của `submission.zip`; taxonomy cho kết quả
gần tương đương. Chưa có checksum của tệp đã upload để xác nhận tuyệt đối nó
trùng `outputs/evidence/test_20261006/selected/submission.zip` trong workspace.
Ảnh cho thấy:

| Chỉ số | Điểm |
|---|---:|
| FINAL_SCORE | 0,1443 |
| DOCS_F2MACRO | 0,2347 |
| CHUNKS_F2MACRO | 0,0538 |
| DOCS_PRECISION / DOCS_RECALL | 0,1246 / 0,3866 |
| CHUNKS_PRECISION / CHUNKS_RECALL | 0,0430 / 0,0722 |

Điểm cuối khớp, sau làm tròn, trung bình hai F2. Đây là suy luận từ ảnh,
chưa phải xác minh chương trình chấm chính thức. Cả precision và recall của
chunk đều thấp: chỉ tăng số lượng hoặc chỉ siết threshold đều chưa đủ.
Coverage crawl 80,70% cũng không phải recall retrieval; URL đã tải không đồng
nghĩa tài liệu đúng đã vào candidate pool và evidence đúng đã được chọn.

Đối chiếu hai ZIP baseline và taxonomy trên đủ 1.200 câu hỏi: **0 câu thay đổi
tập document, 0 câu thay đổi tập chunk**; 692 câu thay đổi thứ tự document.
Đổi thứ tự không thể cải thiện document F2 dựa trên tập dự đoán khi cutoff
không đổi. Chưa xác minh scorer có phụ thuộc thứ tự khi ghép chunk hay không.

Luồng ZIP baseline (`r2ai/pack_cached_submission.py`) chọn một chunk theo dense
similarity cho mỗi document. Luồng rerank evidence chỉ chấm lại những chunk đó,
không tìm chunk khác cùng document; `document-mode=baseline` giữ 200 document
gốc. Vì vậy reranker chưa sửa được document selection và chưa sửa được việc
bỏ sót evidence ngoài pool này. Cache thật có 144.000 cặp, tức 120 document có
evidence mỗi câu; 80 document còn lại của baseline chưa có điểm reranker.

Threshold -6 khá rộng: trước dedup/cap, trung bình 109,66/120 chunk vượt ngưỡng.
Các ngưỡng -4, -2, 0 giữ trung bình 95,55; 71,95; 42,03 chunk. Ở ngưỡng 0,
24 câu không còn evidence. Đây là phân bố score, không phải bằng chứng ngưỡng
nào tăng F2. Score là raw logit, không phải xác suất relevance.

## Hai bản test mới

Script tái lập: `outputs/build_vibiomir_probes_20261007.py`.
Artifacts và checksum: `outputs/submissions/vibiomir_probes_20261007/`.

- `vibiomir_rerank80_exact_20261007.zip`: chọn tối đa 80 document theo điểm
  chunk reranker cao nhất của document, tối đa 60 chunk có score >= -6, một
  chunk/document, giữ text nguồn. Đây là bản đối chứng về độ dài.
- `vibiomir_rerank80_window1_20261007.zip`: cùng document, cùng chunk trung tâm
  và cùng số chunk; nối thêm chunk liền trước/sau trong cùng document nếu có.
  Không sinh hoặc dịch evidence. Đây là giả thuyết tăng độ phủ nội dung cần
  được kiểm chứng bằng điểm CHUNKS_F2MACRO và CHUNKS_RECALL thực tế.
- Giữ evidence của hai document riêng biệt dù text giống nhau; không khử trùng
  text trên toàn bộ các document. Chỉ số document có phân biệt doc_id.
- JSON rời được giữ bên cạnh ZIP. Mỗi ZIP chứa đúng một `predictions.json` ở
  root, đầy đủ query IDs và document IDs thuộc corpus, đúng quan hệ cha.

Đây là thí nghiệm, không phải cấu hình đã hiệu chỉnh. So với submission cũ,
các bản mới đổi document cutoff/selection và chunk cutoff. **Giữa exact và
window1 chỉ đổi text mở rộng**. Chấm exact trước để đo selection, chấm window1
sau để đo ảnh hưởng của ngữ cảnh. Document F2 của hai bản phải bằng nhau với
cùng scorer/tập test. Nếu không bằng, kiểm tra lại tệp đã upload và phase chấm.

Validator hiện có và format của các ZIP đã được chấm là cơ sở đóng gói. Chưa
đọc được độc lập submission instructions và scorer chính thức qua web/API.
Không khẳng định luật LCS, tokenizer hay ngưỡng 40% trong comment code cũ đã
được xác minh. Cần đối chiếu scorer trước khi tối ưu theo các giả định này.

## Thứ tự ưu tiên cho dự án

1. **Kiểm chứng luật chấm và chất lượng evidence.** Lấy scorer/submission
   instructions từ trang cuộc thi; kiểm tokenization, normalization, matching
   theo doc_id, ghép một-một/nhiều-một và cách tính macro. Dùng mẫu từ 100–200
   câu ViBioMIR, chia dev/holdout theo query ID, gán nhãn document và passage
   đúng/sai, có cả kết quả tiếng Trung và tiếng Việt. Nhãn trong candidate pool
   chỉ đo recall trong pool, không đo recall toàn corpus. Không dùng điểm
   ViMedQA làm điểm validation ViBioMIR.

2. **Rerank nhiều evidence trong mỗi document và chọn lại document.** Từ
   shortlist BM25 + dense, lấy 2–4 chunk tốt nhất/document trước cross-encoder;
   bảo đảm document diversity trong pool thay vì để vài document chiếm hết
   global top-chunk pool. Chấm cả document ngoài 120 evidence của ZIP cũ.
   Tổng hợp max/top-2 chunk score để rank document; giữ pool/cache đầy đủ để
   thay cutoff không phải chạy lại model. Đánh giá recall@100/400/1000 của
   stage 1 riêng với quality của stage 2.

3. **Cải thiện chunking/extraction.** Với document liên quan nhưng chunk sai,
   kiểm tra text bài chính, đoạn trả lời, lỗi boilerplate/encoding và ranh giới
   chunk. Thử chunk theo tokenizer thật, ranh giới đoạn/câu và overlap; so
   sánh nhiều độ dài trên scorer đúng. Window1 là probe ban đầu, không thay
   thế thiết kế chunking. Sau đó cho phép nhiều chunk/document khi câu hỏi
   cần nhiều ý, thay vì áp cứng một chunk/document. Audit nhóm khoảng 300 nghìn
   document crawl thành công nhưng chưa có chunks trong báo cáo 06/10;
   phân biệt extraction lỗi với trang không có nội dung hữu ích.

4. **Tăng recall candidate và xử lý đa ngôn ngữ.** Dense stage 1 hiện biểu diễn
   document bằng title + vài chunk đầu, nên evidence ở giữa/cuối có thể bị bỏ
   sót. Thử nhiều cửa sổ hoặc chunk retrieval rồi gom theo document, đo chi
   phí/recall trên dev trước khi rebuild toàn corpus. BM25 dùng query Việt
   không tự truy hồi alias tiếng Trung; thử query expansion song ngữ có kiểm
   soát, giữ query gốc và so BM25/dense/fusion theo ngôn ngữ. Taxonomy chỉ hữu
   ích khi thay candidate pool, selection hoặc features có hiệu chỉnh.

5. **Hiệu chỉnh số lượng theo dữ liệu thật rồi mới fine-tune.** Replay các
   cutoff document/chunk trên cùng cache và dev/holdout; tối ưu macro F2,
   kiểm recall giảm và câu trả rỗng. Không suy chính xác số gold hoặc K tối ưu
   từ tỷ số macro precision/recall: trung bình tỷ số không bằng tỷ số trung
   bình. Fine-tune reranker bằng positive/hard negative cùng bệnh nhưng sai
   ý định, cơ quan, xét nghiệm hoặc con số sau khi có nhãn đủ tin cậy.

## Ghi kết quả

Ghi cả bảy chỉ số, tên ZIP, SHA-256, phase và thời điểm cho mỗi lần chấm.
Không chỉ nhìn FINAL_SCORE. Nếu exact cải thiện DOCS_F2 nhưng window1 không
cải thiện CHUNKS_F2, ưu tiên tìm đúng passage/chỉnh extractor trước khi tăng
độ dài tiếp. Nếu recall document giảm mạnh khi dùng 80 document, mở rộng
scored candidate pool và thử cutoff lớn hơn; không mặc định 80 là tối ưu.
Nếu hai bản đều không tăng, giữ baseline và dùng phân tích có nhãn để quyết
định thay retrieval, reranker hay chunk matching.

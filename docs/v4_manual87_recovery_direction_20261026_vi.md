# Hướng phục hồi 87 manual evidence — 26/10/2026

## Baseline đã đối chiếu

ZIP cap-10 có SHA-256
`e422b9076bf59f9a93d4819776026e789e7cf3f407b3befee8738fc5abb270f8`.
Payload trong ZIP bằng đúng `export/predictions.json`.

Đối chiếu 87 registry IDs từ R11 cohort với original `evidence_text`, kiểm hash
nhãn, hash source và corrected source-offset round-trip: **65/87 đã được chứa
đầy đủ trong output, 22/87 còn thiếu**, theo normalized exact containment.
Đây không phải official CHUNKS_F2, cũng không đo precision của toàn bộ context.

| Phép kiểm tra độc lập | Có / 87 |
|---|---:|
| Source hiện hành chứa evidence | 87 |
| Có ít nhất một passage đã index chứa toàn evidence | 87 |
| Parent có trong refined candidate pool | 76 |
| Passage trong refined candidate pool chứa evidence | 71 |
| Parent có trong cap-10 documents | 76 |
| Một context trong R5 candidate set chứa evidence | 75 |
| Context cố định để cap-10 chấm chứa evidence | 73 |
| Parent có chunk slot | 67 |
| Chunk xuất ra chứa evidence | 65 |

Các hàng không phải một chuỗi tập con: mở rộng passage thành context có thể
phục hồi evidence không có đầy đủ trong seed passage.

Phân nhóm 22 output misses không trùng nhau:

- 11 parent không có trong refined pool: queries 1, 23, 245, 349, 396, 610,
  918, 991, 1013, 1035, 1159.
- 9 parent có document nhưng không có slot: 65, 71, 130, 353, 566, 579,
  594, 910, 1150.
- 2 parent có slot nhưng sai context: 672, 893. Cả hai đã có context đúng
  trong R5 candidate set.

Một ca có thể có nhiều lỗi: q566 vừa thiếu slot vừa thiếu context đúng trong
candidate set. Vì vậy 9 slot misses không có nghĩa chỉ sửa allocator là đủ.
Không suy raw retrieval miss từ việc thiếu parent trong refined pool.

## Bước 1: phục hồi candidate coverage trước khi đổi slot

Mục tiêu là candidate pool có ít nhất một đoạn chứa evidence cho mỗi cặp
query–parent đã gán nhãn. Mục tiêu 87/87 là tiêu chí kiểm thử, chưa phải kết quả
đã đạt được bằng retrieval. Không chèn doc IDs hoặc evidence từ gold vào pool
inference để đạt mục tiêu này.

Hướng thực hiện: **truy hồi parent với độ bao phủ cao, sau đó mở rộng passages
trong từng parent trước khi chọn context**.

1. Giữ candidate pool hiện hành làm một nhánh union. Dùng query gốc truy hồi
   document BM25, passage BM25 và dense; gom parent trước khi áp parent cap.
   Tái sử dụng raw traces đã có; phân biệt raw miss, fusion drop và chưa đủ trace.
   Các parent mới cần đi tiếp tới evidence pool ngay cả khi chưa được chọn vào
   danh sách documents của champion.
2. Với mọi parent trong pool, liệt kê passages đã index của parent, thay vì chỉ
   giữ passages thắng global retrieval hoặc một seed top-1. Tách parent retrieval
   budget khỏi passage coverage. Deduplicate bằng identity và lưu source offsets.
   Có thể lưu manifest passage IDs thay vì nhân bản toàn text theo query.
3. Đo coverage của pool mở rộng **trước scoring/pruning**. Cả 87 evidence đã có
   trong indexed passages, nên chưa có bằng chứng cần re-chunk/rebuild index cho
   cohort này. Với 76 parent hiện có, mở rộng toàn bộ indexed passages có cơ sở
   đạt 76/87 candidate coverage; phần còn lại phụ thuộc thu hồi 11 parent.
4. Nếu parent vẫn không xuất hiện trong raw union, thử tăng retrieval depth theo
   các mức đã khóa và query normalization trên development. Dùng controls để
   kiểm mất parent; không mặc định tăng depth sẽ đạt 87/87. Nếu thiếu trace thì
   báo UNKNOWN, không đoán nguyên nhân. R11 hiện STOP_BUDGET, cần một retrieval
   path hiệu quả hơn và đo chi phí trước khi tiếp tục cohort.
5. Sau khi có coverage report, mới áp cheap passage filter rồi evidence scorer.
   Báo riêng coverage sau mỗi lần cắt để không giấu việc đánh mất đoạn vừa phục
   hồi. Không cross-encode toàn bộ expanded pool một cách không giới hạn.

Điều kiện hoàn tất bước 1: báo cáo đủ 87 cặp, candidate spans round-trip đúng
source, ghi rõ số ca đã phục hồi/chưa phục hồi, không mất evidence đang có trong
pool baseline, và đo chi phí. Chỉ khi đạt đủ mới ghi `87/87 candidate coverage`.
Đây chưa phải `87/87 final output`.

## Các bước sau, thực hiện riêng

1. Context selection: chấm nhiều passages/spans trong cùng parent, chọn evidence
   trước rồi dựng context quanh nó. Không cố định context trước khi evidence
   scorer nhìn thấy alternatives. Giữ baseline context làm candidate để đo loss.
2. Allocation: dùng điểm evidence để cấp 60 slots; document relevance và evidence
   relevance có mục tiêu riêng. Không dùng lại chính document logit làm tín hiệu
   mới, không chỉ tăng max-swaps.
3. Đo final manual containment cùng gross gains/losses, kiểm source/parent/caps,
   và đánh giá relevance trực tiếp. Đạt 87/87 manual containment không tự chứng
   minh official F2 tăng hoặc tổng quát tốt trên query mới.

Tập 87 đã được xem để chẩn đoán. Giữ split registry hiện hành; không dùng các ca
validation/held-out vừa thấy để lựa chọn rules riêng theo query. Không gọi cohort
này là một independent holdout mới.

## Artifacts của lần audit này

- `outputs/v4/manual87_recovery_audit_20261026/audit.json`: counts, nguồn/hashes,
  original-registry recheck và giới hạn phép đo.
- `outputs/v4/manual87_recovery_audit_20261026/all87.csv`: từng cặp và từng stage.
- `outputs/v4/manual87_recovery_audit_20261026/missing22.csv`: danh sách cần phục hồi.

Đã thực hiện audit và lập hướng xử lý; chưa đổi runtime, retrieval, slot allocation
hay champion submission.

# Kiểm tra định dạng submission

`r2ai.validate.validate` trả 0 lỗi. ZIP preview chứa 1.200 query, 1.150 query ngoài pilot cố ý rỗng; không dùng để nộp hoặc đánh giá leaderboard.

```text
records            : 1200
unique query ids   : 1200 / 1200
docs per query     : avg 0.17  (empty: 1150)
chunks per query   : avg 0.17  (empty: 1150)
zip size           : 0.0 MB
WARN  1150 queries have no predicted docs -> contribute 0 to Doc F2
```

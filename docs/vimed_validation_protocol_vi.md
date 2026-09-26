## T?i nguy?n v? th?i gian

RTX 4050 Laptop 6 GB; RAM 16 GB, c?n kho?ng 5,5 GB tr??c l??t reranker; disk c?n kho?ng 178 GB. Dense d?ng peak PyTorch allocated kho?ng 1,07 GiB; reranker 1,15 GiB (reserved 1,23 GiB). ??y l? b? nh? PyTorch, kh?ng g?m desktop v? driver. Reranker ch?y 2.210 c?u trong 1.729 gi?y (~28,8 ph?t), sau warmup. Th?i gian t?i model kh?ng t?nh v?o benchmark.

207 c?u ch?a t?m ???c context ngu?n trong top10: 166 c?u kh?ng c? context ngu?n trong candidates top100, 41 c?u c? nh?ng b? x?p sau top10. B??c c?i thi?n ti?p theo n?n th? t?ng pool candidates v? ki?m tra l?i topic 2 (Recall@10 82,55%) tr?n validation.

## Ch?y l?i

D?ng m?i tr??ng `.venv` c? CUDA; ??t `HF_HOME` tr? t?i `.cache/huggingface`. Model ?? ???c cache v? kh?a revision; c? th? ??t `HF_HUB_OFFLINE=1`.

```powershell
.venv/Scripts/python.exe -m scripts.prepare_vimed_validation
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage bm25 --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage dense --timeout-seconds 1200
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage hybrid --timeout-seconds 600
.venv/Scripts/python.exe -m scripts.benchmark_vimed_validation --stage rerank --timeout-seconds 7200
.venv/Scripts/python.exe -m scripts.report_vimed_validation
.venv/Scripts/python.exe -m pytest -q
```

L??t t??ng th?ch ?? ho?n th?nh s? ???c d?ng l?i. N?u ??i config ho?c d? li?u, d?ng th? m?c `--output` m?i; ch??ng tr?nh t? ch?i tr?n cache kh?c fingerprint. `--limit` ch? d?nh cho probe v? b?t bu?c c? th? m?c output ri?ng. Windows Application Control ch?n DLL sklearn trong sandbox c?a Codex; l??t model th?t ?? ch?y ngo?i sandbox v?i quy?n ???c c?p.

So v?i BM25 ? top10: kh?i ph?c 349 c?u, gi?m ch?t l??ng 9 c?u, c?ng b? s?t 198 c?u.

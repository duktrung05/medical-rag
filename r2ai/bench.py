"""Measure embed and rerank throughput on this box, to size the full-corpus run.

Estimates for a 4.4M-document corpus are only as good as the per-item rate, and
that rate depends on the card, the dtype and the sequence length actually used.
So measure it here rather than quoting datasheet numbers.
"""
from __future__ import annotations

import argparse
import time

import torch
from sentence_transformers import CrossEncoder, SentenceTransformer

EMBED_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

# Representative of the corpus: a Vietnamese query against Chinese/Vietnamese body
# text, padded to the sequence length the pipeline actually uses.
VI = "Cần làm gì đối với tình trạng tắc nghẽn đường tiết niệu do sỏi thận? "
ZH = "弱视是目前发病率相当高的一种眼病了，也是儿童时期较为多见的一种，视力开始下降。"


def make_docs(n: int, seq: int) -> list[str]:
    """Build n documents whose tokenised length is around `seq`."""
    unit = (VI + ZH) * max(1, seq // 40)
    return [f"{i} {unit}" for i in range(n)]


def bench_embed(device: str, n: int, batch: int, seq: int, dtype: torch.dtype) -> float:
    model = SentenceTransformer(EMBED_MODEL, device=device, model_kwargs={"torch_dtype": dtype})
    model.max_seq_length = seq
    docs = make_docs(n, seq)
    model.encode(docs[: min(64, n)], batch_size=batch, show_progress_bar=False)  # warm up
    torch.cuda.synchronize(device)
    start = time.time()
    model.encode(docs, batch_size=batch, normalize_embeddings=True,
                 convert_to_numpy=True, show_progress_bar=False)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    del model
    torch.cuda.empty_cache()
    return n / elapsed


def bench_rerank(device: str, n: int, batch: int, seq: int, dtype: torch.dtype) -> float:
    model = CrossEncoder(RERANK_MODEL, device=device, max_length=seq,
                         model_kwargs={"torch_dtype": dtype})
    pairs = [[VI, d] for d in make_docs(n, seq)]
    model.predict(pairs[: min(64, n)], batch_size=batch, show_progress_bar=False)  # warm up
    torch.cuda.synchronize(device)
    start = time.time()
    model.predict(pairs, batch_size=batch, show_progress_bar=False)
    torch.cuda.synchronize(device)
    elapsed = time.time() - start
    del model
    torch.cuda.empty_cache()
    return n / elapsed


def hours(count: float, rate: float) -> str:
    return f"{count / rate / 3600:6.2f} h" if rate > 0 else "n/a"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--n", type=int, default=3072)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    print(f"device={args.device}  {torch.cuda.get_device_name(args.device)}\n")
    results: dict[tuple[str, int, str], float] = {}
    for dtype, label in ((torch.float16, "fp16"), (torch.float32, "fp32")):
        for seq in (256, 512):
            rate = bench_embed(args.device, args.n, args.batch, seq, dtype)
            results[("embed", seq, label)] = rate
            print(f"embed   seq={seq:<4} {label}  {rate:8.1f} docs/s")
    for seq in (256, 512):
        rate = bench_rerank(args.device, args.n // 2, args.batch, seq, torch.float16)
        results[("rerank", seq, "fp16")] = rate
        print(f"rerank  seq={seq:<4} fp16  {rate:8.1f} pairs/s")

    print("\n--- projected wall time, single card ---")
    corpus = 4_394_718
    for seq in (256, 512):
        r = results[("embed", seq, "fp16")]
        print(f"embed 4.39M docs @ 1 vector, seq={seq}: {hours(corpus, r)}")
    r512 = results[("rerank", 512, "fp16")]
    for pool in (100, 200):
        print(f"rerank 1200 queries x {pool} chunks:        {hours(1200 * pool, r512)}")

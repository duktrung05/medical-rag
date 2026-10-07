"""Extract main article text from crawled pages and cut it into chunks.

Chunk size is the single most score-sensitive knob in this competition. The
official chunk metric counts a predicted chunk as relevant when its longest
common *subsequence* with some reference chunk covers >=40% of that reference
chunk's BGE-M3 tokens, and it only ever compares against reference chunks inside
the same document. Because LCS is a subsequence (not a substring) measure, a
longer chunk can only ever match more reference text, while precision counts
chunks, not tokens. Target size is therefore exposed as a parameter so it can be
A/B tested against the public leaderboard rather than guessed.
"""
from __future__ import annotations

import argparse
import gzip
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import lxml.html as LH
import trafilatura

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/vibiomir"
CRAWL = OUT / "crawl"
PAGES = CRAWL / "pages"

_WS = re.compile(r"[ \t ]+")
_NL = re.compile(r"\n{3,}")
CHUNK_SCHEMA = pa.schema([
    ("doc_id", pa.int64()), ("chunk_index", pa.int32()),
    ("title", pa.string()), ("text", pa.string()), ("n_words", pa.int32()),
])


def shard_path(doc_id: int) -> Path:
    return PAGES / f"{doc_id % 1000:03d}" / f"{doc_id}.gz"


def clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = _WS.sub(" ", text)
    return _NL.sub("\n\n", text).strip()


def extract_page(doc_id: int) -> tuple[str, str] | None:
    """Return (title, body) for a crawled doc, or None when unusable."""
    path = shard_path(doc_id)
    if not path.exists():
        return None
    try:
        raw = gzip.decompress(path.read_bytes())
    except Exception:  # noqa: BLE001
        return None
    html = None
    for enc in ("utf-8", "gb18030", "latin-1"):
        try:
            html = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if html is None:
        return None

    body = trafilatura.extract(
        html, include_comments=False, include_tables=True,
        favor_recall=True, no_fallback=False,
    )
    title, fallback = lxml_text(html)
    if not body:
        body = fallback
    if not body:
        return None
    return title[:300], clean(body)


# selectolax's parser silently yields an empty tree on some corpus pages, so lxml
# is used for the fallback path and for titles.
_BOILERPLATE = ("script", "style", "noscript", "nav", "footer", "header",
                "aside", "form", "iframe", "svg", "button")


def lxml_text(html: str) -> tuple[str, str]:
    """Return (title, body-text) using lxml, stripping boilerplate elements."""
    try:
        doc = LH.fromstring(html)
    except Exception:  # noqa: BLE001
        return "", ""
    title = ""
    for xp in ("//h1", "//title"):
        found = doc.xpath(xp)
        if found and found[0].text_content().strip():
            title = clean(found[0].text_content())
            break
    for element in doc.xpath("|".join(f"//{t}" for t in _BOILERPLATE)):
        parent = element.getparent()
        if parent is not None:
            parent.remove(element)
    # Prefer the densest article-like container; fall back to the whole document.
    best, best_len = None, 0
    for node in doc.xpath("//article|//main|//*[contains(@class,'content')]|//*[contains(@class,'detail')]"):
        length = len(node.text_content())
        if length > best_len:
            best, best_len = node, length
    root = best if best is not None and best_len > 400 else doc
    blocks = []
    for node in root.xpath(".//p|.//li|.//h2|.//h3|.//h4|.//td|.//div[not(*)]"):
        text = node.text_content().strip()
        if len(text) > 1:
            blocks.append(text)
    body = "\n".join(dict.fromkeys(blocks)) if blocks else root.text_content()
    return title, body


# CJK text carries no spaces, so whitespace counting undercounts it by ~20x and a
# word-based chunk target would never split a long Chinese article. Counting each
# CJK character as one unit tracks BGE-M3 tokenisation (the scorer's tokenizer)
# closely enough for sizing.
_CJK = re.compile(r"[㐀-鿿豈-﫿぀-ヿ가-힯]")


def token_estimate(text: str) -> int:
    """Approximate BGE-M3 token count across Vietnamese, English and Chinese."""
    cjk = len(_CJK.findall(text))
    return cjk + len(_CJK.sub(" ", text).split())


def _tail_by_tokens(text: str, budget: int) -> str:
    """Trailing whitespace-delimited pieces totalling about `budget` tokens.

    Slicing on whitespace keeps the text verbatim up to whitespace normalisation,
    which the official scorer applies anyway.
    """
    if budget <= 0:
        return ""
    pieces = text.split()
    kept: list[str] = []
    total = 0
    for piece in reversed(pieces):
        size = token_estimate(piece)
        if kept and total + size > budget:
            break
        kept.append(piece)
        total += size
    return " ".join(reversed(kept))


def split_paragraphs(text: str) -> list[str]:
    parts = [p.strip() for p in text.split("\n") if p.strip()]
    return [p for p in parts if len(p) > 1]


def pack(paragraphs: list[str], target_words: int, overlap: int,
         min_ratio: float = 0.45) -> list[str]:
    """Greedily pack paragraphs up to target_words, with word-level overlap.

    A chunk is only closed once it holds min_ratio of the target. Without that
    guard a short heading followed by a long paragraph is emitted as its own
    tiny chunk, which can never cover 40% of a reference chunk and so only costs
    precision.
    """
    min_words = max(1, int(target_words * min_ratio))
    chunks: list[str] = []
    buf: list[str] = []
    count = 0
    for para in paragraphs:
        size = token_estimate(para)
        if count >= min_words and count + size > target_words:
            chunks.append(" ".join(buf))
            tail = _tail_by_tokens(chunks[-1], overlap)
            buf, count = ([tail] if tail else []), token_estimate(tail)
        buf.append(para)
        count += size
    if buf:
        chunks.append(" ".join(buf))
    chunks = [c for c in chunks if c.strip()]
    # Fold a stub tail back into its predecessor rather than dropping the content.
    if len(chunks) > 1 and token_estimate(chunks[-1]) < min_words:
        chunks[-2] = f"{chunks[-2]} {chunks[-1]}"
        chunks.pop()
    return chunks


def process(args: tuple[int, int, int, int]) -> list[dict]:
    doc_id, target_words, overlap, max_chunks = args
    got = extract_page(doc_id)
    if not got:
        return []
    title, body = got
    paragraphs = split_paragraphs(body)
    if not paragraphs:
        return []
    chunks = pack(paragraphs, target_words, overlap)[:max_chunks]
    return [
        {"doc_id": doc_id, "chunk_index": i, "title": title, "text": c,
         "n_words": token_estimate(c)}
        for i, c in enumerate(chunks)
        if token_estimate(c) >= 16
    ]


def main(target_words: int, overlap: int, max_chunks: int, workers: int, out_name: str) -> None:
    doc_ids: list[int] = []
    for part in sorted(CRAWL.glob("manifest_*.parquet")):
        table = pq.read_table(part, columns=["doc_id", "status"]).to_pydict()
        doc_ids.extend(d for d, s in zip(table["doc_id"], table["status"]) if s == "ok")
    doc_ids = list(dict.fromkeys(doc_ids))
    print(f"docs to extract: {len(doc_ids):,}", flush=True)

    # Stream to parquet in row-group batches. Accumulating a million documents'
    # worth of chunk dicts before the single write costs many GB of RAM and loses
    # the whole run if it fails at the end.
    out = OUT / out_name
    rows: list[dict] = []
    total = docs_with_chunks = 0
    tasks = [(d, target_words, overlap, max_chunks) for d in doc_ids]
    writer: pq.ParquetWriter | None = None
    try:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, result in enumerate(pool.map(process, tasks, chunksize=64), 1):
                if result:
                    docs_with_chunks += 1
                rows.extend(result)
                if len(rows) >= 200_000:
                    writer = writer or pq.ParquetWriter(out, CHUNK_SCHEMA, compression="zstd")
                    writer.write_table(pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA))
                    total += len(rows)
                    rows = []
                if n % 50_000 == 0:
                    print(f"  {n:,}/{len(doc_ids):,} docs -> {total + len(rows):,} chunks",
                          flush=True)
        if rows or writer is None:
            writer = writer or pq.ParquetWriter(out, CHUNK_SCHEMA, compression="zstd")
            writer.write_table(pa.Table.from_pylist(rows, schema=CHUNK_SCHEMA))
            total += len(rows)
    finally:
        if writer is not None:
            writer.close()
    print(f"wrote {out}: {total:,} chunks over {docs_with_chunks:,} docs", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-words", type=int, default=220)
    ap.add_argument("--overlap", type=int, default=40)
    ap.add_argument("--max-chunks", type=int, default=60)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--out", default="chunks.parquet")
    a = ap.parse_args()
    main(a.target_words, a.overlap, a.max_chunks, a.workers, a.out)

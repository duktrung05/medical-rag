"""Language grouping for cross-lingual score correction.

BGE-M3 shares one vector space across languages, but similarity between a query
and a document in the *same* language still runs systematically higher than
across languages. Measured on this corpus: the stage-1 index was 47.1% Chinese
yet only 6.3% of retrieved documents were, so Chinese evidence was being filtered
out by an artefact of the embedding space rather than by relevance.

MEASURED 2026-10-04 — the same-language-bias premise above is WRONG, and the
z-score correction this module implements made things worse. Do not enable
`--lang-balance` without re-measuring.

Over 60 queries x 250k documents of the v3 index:

    vi  115,771 docs   mean 0.3677   std 0.0683
    zh  121,943 docs   mean 0.3893   std 0.0648   <- higher, not lower

Chinese documents score *above* Vietnamese on average, so there is no offset to
remove. Standardising within language subtracts more from Chinese (higher mean,
tighter spread) and pushed Chinese down in the top-300 from 29.6% to 26.5%.

What is real is that Chinese is filtered out progressively:

    index 48.6%  ->  stage-1 top-300 21.4%  ->  after rerank 10.3%

The cause is the tail, not the mean: relevant Vietnamese documents reach much
higher similarity than any Chinese one does, so Chinese loses on the top-k cut
rather than on average score. Whether boosting Chinese *helps* is unknown
without labels and can only be settled by a controlled leaderboard run, so the
useful part of this module is the measurement harness, not the correction.
"""
from __future__ import annotations

from urllib.parse import urlsplit

import numpy as np

VI, ZH, OTHER = 0, 1, 2
LANG_NAMES = {VI: "vi", ZH: "zh", OTHER: "other"}

# Hosts whose language is not implied by the TLD.
_ZH_HOSTS = {
    "www.cnkang.com", "www.120ask.com", "zysjonline.com", "www.a-hospital.com",
    "zhongyibaodian.net", "www.zydcd.com", "www.wujue.com", "qihuangzhishu.com",
    "www.youlai.cn", "bingli.iiyi.com", "iiyi.com", "test.pmphai.com",
}


def host_language(netloc: str) -> int:
    host = netloc.lower()
    if host.endswith(".vn"):
        return VI
    if host.endswith(".cn") or host.endswith(".39.net") or host in _ZH_HOSTS:
        return ZH
    return OTHER


def doc_languages(doc_ids: np.ndarray, url_by_id: dict[int, str]) -> np.ndarray:
    """Language code per document, aligned with doc_ids."""
    return np.fromiter(
        (host_language(urlsplit(url_by_id.get(int(d), "")).netloc) for d in doc_ids),
        dtype=np.int8, count=len(doc_ids),
    )


class LanguageStats:
    """Per-query, per-language mean and standard deviation of similarity.

    Accumulated over blocks so the whole score matrix never has to be held.
    """

    def __init__(self, n_queries: int, n_langs: int = 3) -> None:
        self.count = np.zeros(n_langs, dtype=np.float64)
        self.total = np.zeros((n_queries, n_langs), dtype=np.float64)
        self.total_sq = np.zeros((n_queries, n_langs), dtype=np.float64)

    def update(self, sims: np.ndarray, langs: np.ndarray) -> None:
        """sims: (n_queries, n_docs_in_block); langs: (n_docs_in_block,)."""
        for lang in range(self.total.shape[1]):
            mask = langs == lang
            if not mask.any():
                continue
            block = sims[:, mask].astype(np.float64)
            self.count[lang] += block.shape[1]
            self.total[:, lang] += block.sum(axis=1)
            self.total_sq[:, lang] += (block ** 2).sum(axis=1)

    def mean_std(self) -> tuple[np.ndarray, np.ndarray]:
        count = np.maximum(self.count, 1)[None, :]
        mean = self.total / count
        var = np.maximum(self.total_sq / count - mean ** 2, 1e-12)
        return mean.astype(np.float32), np.sqrt(var).astype(np.float32)

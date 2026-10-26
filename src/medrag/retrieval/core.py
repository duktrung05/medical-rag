from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Protocol

from src.medrag.core.contracts import Fused, Hit, Passage, Source

WORD = re.compile(r"[a-z0-9À-ỹ]+|[\u4e00-\u9fff]+", re.IGNORECASE)


def tokens(text: str) -> list[str]:
    result = []
    for match in WORD.finditer(unicodedata.normalize("NFC", text).lower()):
        term = match.group()
        if "\u4e00" <= term[0] <= "\u9fff":
            result.extend(term[i:i + 2] for i in range(max(1, len(term) - 1)))
        else:
            result.append(term)
    return result


class BM25:
    def __init__(self, records: list[tuple[int, str | None, str]], channel: str):
        self.records, self.channel = records, channel
        self.counts = [Counter(tokens(text)) for _, _, text in records]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.avgdl = sum(self.lengths) / max(1, len(records))
        self.df = Counter(term for count in self.counts for term in count)
        self.indices_by_parent: dict[int, list[int]] = defaultdict(list)
        for idx, (parent_id, _, _) in enumerate(records):
            self.indices_by_parent[parent_id].append(idx)

    def _search_indices(self, query: str, indices, limit: int) -> list[Hit]:
        n = len(self.records)
        terms = set(tokens(query))
        scores = []
        for idx in indices:
            count = self.counts[idx]
            score = 0.0
            for term in terms & count.keys():
                tf = count[term]
                idf = math.log(1 + (n - self.df[term] + .5) / (self.df[term] + .5))
                score += idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * self.lengths[idx] / max(self.avgdl, 1)))
            if score:
                scores.append((score, idx))
        scores.sort(key=lambda item: (-item[0], self.records[item[1]][0], str(self.records[item[1]][1])))
        return [Hit(self.records[idx][0], self.records[idx][1], score, self.channel, rank)
                for rank, (score, idx) in enumerate(scores[:limit], 1)]

    def search(self, query: str, limit: int = 20) -> list[Hit]:
        return self._search_indices(query, range(len(self.records)), limit)

    def search_parent(self, query: str, parent_id: int, limit: int = 1) -> list[Hit]:
        """Use global corpus BM25 statistics while scanning only one parent."""
        return self._search_indices(query, self.indices_by_parent.get(parent_id, ()), limit)


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class Dense:
    def __init__(self, passages: list[Passage], encoder: Encoder,
                 titles_by_doc: dict[int, str] | None = None):
        self.passages = passages
        self.encoder = encoder
        titles_by_doc = titles_by_doc or {}
        payload = [f"{titles_by_doc.get(p.official_id, '')}\n{p.text}"
                   if titles_by_doc.get(p.official_id) else p.text for p in passages]
        self.vectors = encoder.encode(payload)

    def search(self, query: str, limit: int = 20) -> list[Hit]:
        encode_query = getattr(self.encoder, "encode_query", self.encoder.encode)
        vector = encode_query([query])[0]
        def cosine(a, b):
            numerator = sum(x*y for x, y in zip(a, b, strict=True))
            denominator = math.sqrt(sum(x*x for x in a) * sum(y*y for y in b))
            return numerator / denominator if denominator else 0.0
        ranked = sorted(((cosine(vector, v), i) for i, v in enumerate(self.vectors)),
                        key=lambda item: (-item[0], self.passages[item[1]].official_id,
                                          self.passages[item[1]].passage_id))
        return [Hit(self.passages[i].official_id, self.passages[i].passage_id, score, "dense", rank)
                for rank, (score, i) in enumerate(ranked[:limit], 1)]


def fuse(rankings: list[list[Hit]], k: int = 60, limit: int = 100) -> list[Fused]:
    scores = defaultdict(float)
    ranks = defaultdict(dict)
    for ranking in rankings:
        for hit in ranking:
            key = (hit.official_id, hit.passage_id)
            if hit.channel in ranks[key]:
                continue
            ranks[key][hit.channel] = hit.rank
            scores[key] += 1 / (k + hit.rank)
    return [Fused(doc, passage, score, ranks[(doc, passage)]) for (doc, passage), score in
            sorted(scores.items(), key=lambda item: (-item[1], item[0][0], str(item[0][1])))[:limit]]


def fuse_with_parent_floor(rankings: list[list[Hit]], *, k: int = 60,
                           item_limit: int = 100, parent_limit: int = 100) -> list[Fused]:
    """Keep the frozen mixed-item prefix, then add one item for unseen parents.

    RRF ranks passages and document hits as separate items. Cutting that list
    before grouping can leave far fewer parent candidates than the configured
    item limit. This opt-in policy preserves every original item and its order,
    then supplements the pool until it contains ``parent_limit`` unique parents.
    """
    if min(k, item_limit, parent_limit) <= 0:
        raise ValueError("Fusion limits and k must be positive")
    available = sum(len(ranking) for ranking in rankings)
    full = fuse(rankings, k=k, limit=max(item_limit, available))
    output = list(full[:item_limit])
    parents = {item.official_id for item in output}
    if len(parents) >= parent_limit:
        return output
    for item in full[item_limit:]:
        if item.official_id in parents:
            continue
        output.append(item)
        parents.add(item.official_id)
        if len(parents) >= parent_limit:
            break
    return output


def build_lexical(sources: list[Source], passages: list[Passage]) -> tuple[BM25, BM25]:
    return (BM25([(s.official_id, None, f"{s.title} {s.text}") for s in sources], "document_bm25"),
            BM25([(p.official_id, p.passage_id, p.text) for p in passages], "passage_bm25"))


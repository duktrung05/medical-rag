"""Versioned lexical taxonomy and metadata annotation for ViBioMIR.

The first version intentionally uses auditable exact aliases and phrase rules.
It is a high-precision seed for experiments, not a complete medical ontology.
"""
from __future__ import annotations

import argparse
from functools import cached_property, lru_cache
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/vibiomir/taxonomy_v1.json"
DEFAULT_CHUNKS = ROOT / "data/vibiomir/chunks_v4.parquet"
DEFAULT_CORPUS = ROOT / "data/raw/vibiomir/links_corpus.parquet"


def normalize(text: str) -> str:
    """Normalize Unicode and case while preserving Vietnamese tone distinctions."""
    return unicodedata.normalize("NFKC", text).casefold().replace("đ", "d")


def fold_diacritics(text: str) -> str:
    value = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in value if unicodedata.category(ch) != "Mn")


def _contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    if any("\u3400" <= c <= "\u9fff" for c in phrase):
        return phrase in text
    return _literal_pattern(phrase).search(text) is not None


@lru_cache(maxsize=4096)
def _literal_pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w]){re.escape(phrase)}(?![\w])", flags=re.UNICODE)


def _find_alias(text: str, alias: str) -> int:
    """Exact accented match first; use accent folding only for multiword aliases."""
    if _contains_phrase(text, alias):
        if any("\u3400" <= c <= "\u9fff" for c in alias):
            return text.find(alias)
        match = _literal_pattern(alias).search(text)
        return match.start() if match else -1
    if any("\u3400" <= c <= "\u9fff" for c in alias) or len(alias.split()) < 2:
        return -1
    folded_text, folded_alias = fold_diacritics(text), fold_diacritics(alias)
    if not _contains_phrase(folded_text, folded_alias):
        return -1
    if any("\u3400" <= c <= "\u9fff" for c in folded_alias):
        return folded_text.find(folded_alias)
    match = _literal_pattern(folded_alias).search(folded_text)
    return match.start() if match else -1


def _compile_alias_matcher(
    concepts: tuple[dict[str, Any], ...], *, folded: bool,
) -> tuple[re.Pattern[str], dict[str, tuple[dict[str, Any], str]]]:
    alternatives: list[tuple[int, str, str, dict[str, Any], str]] = []
    serial = 0
    for concept in concepts:
        for alias in concept["aliases"]:
            alias_text = normalize(str(alias))
            if folded:
                if len(alias_text.split()) < 2 or any("\u3400" <= c <= "\u9fff" for c in alias_text):
                    continue
                alias_text = fold_diacritics(alias_text)
            cjk = any("\u3400" <= c <= "\u9fff" for c in alias_text)
            body = re.escape(alias_text)
            expression = body if cjk else rf"(?<![\w]){body}(?![\w])"
            group = f"m{serial}"
            serial += 1
            alternatives.append((len(alias_text), expression, group, concept, str(alias)))
    alternatives.sort(key=lambda item: item[0], reverse=True)
    group_map: dict[str, tuple[dict[str, Any], str]] = {}
    expressions = []
    for _, expression, group, concept, alias in alternatives:
        expressions.append(f"(?P<{group}>{expression})")
        group_map[group] = (concept, alias)
    if not expressions:
        raise ValueError("Taxonomy has no usable aliases")
    return re.compile("|".join(expressions)), group_map


_TOKEN_RE = re.compile(r"[\u3400-\u9fff]|[^\W_]+", flags=re.UNICODE)


def _alias_trie(concepts: tuple[dict[str, Any], ...], *, folded: bool = False) -> dict[str, Any]:
    root: dict[str, Any] = {}
    for concept in concepts:
        for raw_alias in concept["aliases"]:
            alias = normalize(str(raw_alias))
            if folded:
                if len(alias.split()) < 2 or any("\u3400" <= c <= "\u9fff" for c in alias):
                    continue
                alias = fold_diacritics(alias)
            tokens = [m.group(0) for m in _TOKEN_RE.finditer(alias)]
            if not tokens:
                continue
            node = root
            for token in tokens:
                node = node.setdefault(token, {})
            node.setdefault("", []).append((concept, str(raw_alias)))
    return root


@dataclass(frozen=True)
class Taxonomy:
    version: str
    concepts: tuple[dict[str, Any], ...]
    intents: dict[str, tuple[str, ...]]
    negation_cues: tuple[str, ...]
    uncertainty_cues: tuple[str, ...]

    @cached_property
    def _exact_trie(self) -> dict[str, Any]:
        return _alias_trie(self.concepts)

    @cached_property
    def _folded_trie(self) -> dict[str, Any]:
        return _alias_trie(self.concepts, folded=True)

    @classmethod
    def load(cls, path: str | Path = DEFAULT_CONFIG) -> "Taxonomy":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        concepts = tuple(raw["concepts"])
        ids = [str(row["id"]) for row in concepts]
        if len(ids) != len(set(ids)):
            raise ValueError("Taxonomy concept IDs must be unique")
        for row in concepts:
            if not row.get("id") or not row.get("type") or not row.get("aliases"):
                raise ValueError(f"Malformed taxonomy concept: {row!r}")
        return cls(
            version=str(raw["version"]),
            concepts=concepts,
            intents={k: tuple(normalize(x) for x in v) for k, v in raw["intents"].items()},
            negation_cues=tuple(normalize(x) for x in raw.get("negation_cues", [])),
            uncertainty_cues=tuple(normalize(x) for x in raw.get("uncertainty_cues", [])),
        )

    def concepts_in(self, text: str) -> list[dict[str, Any]]:
        normalized = normalize(text)
        found: dict[str, tuple[int, str, dict[str, Any]]] = {}
        tokens = list(_TOKEN_RE.finditer(normalized))
        folded_tokens = [fold_diacritics(m.group(0)) for m in tokens]
        for token_values, trie in (([m.group(0) for m in tokens], self._exact_trie),
                                   (folded_tokens, self._folded_trie)):
            for start in range(len(token_values)):
                node = trie
                for token in token_values[start:start + 16]:
                    node = node.get(token)
                    if node is None:
                        break
                    for concept, alias in node.get("", []):
                        concept_id = str(concept["id"])
                        pos = tokens[start].start()
                        if concept_id not in found or pos < found[concept_id][0]:
                            found[concept_id] = (pos, alias, concept)

        matches: list[dict[str, Any]] = []
        for concept_id, (pos, alias, concept) in found.items():
            context = normalized[max(0, pos - 64):pos]
            folded_context = fold_diacritics(context)
            neg_positions = [context.rfind(cue) + len(cue) for cue in self.negation_cues if cue in context]
            neg_positions.extend(folded_context.rfind(fold_diacritics(cue)) + len(fold_diacritics(cue))
                                 for cue in self.negation_cues if fold_diacritics(cue) in folded_context)
            uncertain_positions = [context.rfind(cue) + len(cue) for cue in self.uncertainty_cues if cue in context]
            uncertain_positions.extend(folded_context.rfind(fold_diacritics(cue)) + len(fold_diacritics(cue))
                                       for cue in self.uncertainty_cues if fold_diacritics(cue) in folded_context)
            assertion = "affirmed"
            if neg_positions and (not uncertain_positions or max(neg_positions) >= max(uncertain_positions)):
                assertion = "negated"
            elif uncertain_positions:
                assertion = "uncertain"
            matches.append({
                "concept_id": concept_id,
                "concept_type": str(concept["type"]),
                "matched_alias": alias,
                "assertion": assertion,
                "confidence": None,
                "match_method": "exact_alias",
            })
        return matches

    def intents_in(self, text: str, concepts: list[dict[str, Any]] | None = None) -> list[str]:
        normalized = normalize(text)
        concepts = concepts if concepts is not None else self.concepts_in(text)
        has_test = any(m["concept_type"] in {"test", "vital_sign"} for m in concepts)
        found = []
        for intent, aliases in self.intents.items():
            if intent == "test_interpretation" and not has_test:
                continue
            if any(_find_alias(normalized, alias) >= 0 for alias in aliases):
                found.append(intent)
        return found

    def profile(self, text: str) -> dict[str, Any]:
        mentions = self.concepts_in(text)
        active_ids = {m["concept_id"] for m in mentions if m["assertion"] != "negated"}
        concept_by_id = {str(c["id"]): c for c in self.concepts}
        specialties = {
            specialty
            for concept_id in active_ids
            for specialty in concept_by_id[concept_id].get("specialties", [])
        }
        return {
            "concepts": sorted(active_ids),
            "mentions": mentions,
            "intents": self.intents_in(text, mentions),
            "specialties": sorted(specialties),
            "language": detect_language(text),
        }

    def aliases_for(self, text: str, max_aliases_per_concept: int = 4) -> list[str]:
        """Return bounded, curated expansions for concepts explicitly in a query."""
        active = set(self.profile(text)["concepts"])
        expansions: list[str] = []
        query_norm = normalize(text)
        for concept in self.concepts:
            if str(concept["id"]) not in active:
                continue
            candidates = list(dict.fromkeys([str(concept["preferred"]), *map(str, concept["aliases"])]))
            added = 0
            for alias in candidates:
                if _find_alias(query_norm, normalize(alias)) >= 0 or alias in expansions:
                    continue
                expansions.append(alias)
                added += 1
                if added >= max_aliases_per_concept:
                    break
        return expansions


def detect_language(text: str) -> str:
    """Conservative script/marker estimate; unknown stays unknown."""
    if not text or not text.strip():
        return "unknown"
    cjk = sum(1 for ch in text if "\u3400" <= ch <= "\u9fff")
    letters = sum(1 for ch in text if ch.isalpha())
    if cjk >= 3 and cjk / max(letters, 1) >= 0.12:
        return "zh"
    if re.search(r"[ăâđêôơưĂÂĐÊÔƠƯáàảãạấầẩẫậắằẳẵặéèẻẽẹếềểễệíìỉĩịóòỏõọốồổỗộớờởỡợúùủũụứừửữựýỳỷỹỵ]", text):
        return "vi"
    normalized = fold_diacritics(normalize(text))
    if re.search(r"\b(benh|thuoc|trieu chung|dieu tri|co the|khong|la gi)\b", normalized):
        return "vi"
    latin = sum(1 for ch in text if "a" <= ch.casefold() <= "z")
    return "en" if latin >= 4 else "unknown"


def classify_source(url: str) -> tuple[str, str]:
    """Return domain and neutral source category; this is not a trust score."""
    host = urlsplit(url or "").hostname or ""
    host = host.lower().removeprefix("www.")
    if not host:
        return "", "unknown"
    if host.endswith((".gov.vn", ".gov.cn", ".gov")) or host in {"moh.gov.vn", "moh.gov.cn"}:
        category = "government"
    elif any(term in host for term in ("hospital", "benhvien", "bệnhviện")):
        category = "hospital"
    elif any(term in host for term in ("nhathuoc", "pharmacy", "drugstore")):
        category = "pharmacy"
    elif any(term in host for term in ("ask", "qa", "question", "39.net")):
        category = "qa"
    elif any(term in host for term in ("news", "bao", "vietnamplus")):
        category = "news"
    else:
        category = "other"
    return host, category


def metadata_overlap(query: dict[str, Any], item: dict[str, Any]) -> float:
    """Bounded, transparent match feature; callers calibrate its ranking weight."""
    qconcepts = set(query.get("concepts", []))
    iconcepts = set(item.get("concepts", []))
    qintents = set(query.get("intents", []))
    iintents = set(item.get("intents", []))
    qspecialties = set(query.get("specialties", []))
    ispecialties = set(item.get("specialties", []))
    concept_score = len(qconcepts & iconcepts) / max(len(qconcepts | iconcepts), 1)
    intent_score = len(qintents & iintents) / max(len(qintents | iintents), 1)
    specialty_score = len(qspecialties & ispecialties) / max(len(qspecialties | ispecialties), 1)
    return 0.75 * concept_score + 0.20 * intent_score + 0.05 * specialty_score


def rrf_rerank(
    ranked_ids: list[int], query: dict[str, Any], metadata: dict[int, dict[str, Any]],
    *, weight: float = 0.0, rrf_k: int = 60,
) -> list[int]:
    """Rerank an existing candidate set; unmatched or missing metadata is neutral."""
    if weight <= 0:
        return list(ranked_ids)
    base_rank = {doc_id: rank for rank, doc_id in enumerate(ranked_ids)}
    matches = [(doc_id, metadata_overlap(query, metadata.get(doc_id, {})))
               for doc_id in ranked_ids]
    positive = sorted(((doc, score) for doc, score in matches if score > 0), key=lambda x: (-x[1], base_rank[x[0]]))
    meta_rank = {doc_id: rank for rank, (doc_id, _) in enumerate(positive)}
    scores = {
        doc_id: 1.0 / (rrf_k + base_rank[doc_id])
        + (weight / (rrf_k + meta_rank[doc_id]) if doc_id in meta_rank else 0.0)
        for doc_id in ranked_ids
    }
    return sorted(ranked_ids, key=lambda doc_id: (-scores[doc_id], base_rank[doc_id]))


def _annotation(text: str, taxonomy: Taxonomy) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    mentions = taxonomy.concepts_in(text)
    concepts = sorted({m["concept_id"] for m in mentions if m["assertion"] != "negated"})
    intents = taxonomy.intents_in(text, mentions)
    by_id = {str(c["id"]): c for c in taxonomy.concepts}
    specialties = sorted({s for concept_id in concepts for s in by_id[concept_id].get("specialties", [])})
    return mentions, concepts, intents, specialties


def annotate(chunks_path: Path, corpus_path: Path, config_path: Path, out_dir: Path,
             max_docs: int | None = None, batch_size: int = 20_000, overwrite: bool = False) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    taxonomy = Taxonomy.load(config_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunks_out = out_dir / f"chunks_taxonomy_{taxonomy.version}.parquet"
    docs_out = out_dir / f"docs_taxonomy_{taxonomy.version}.parquet"
    summary_out = out_dir / f"taxonomy_{taxonomy.version}_summary.json"
    targets = [chunks_out, docs_out, summary_out]
    if not overwrite and any(path.exists() for path in targets):
        raise FileExistsError("Output exists; pass --overwrite to replace the taxonomy pilot")

    # For a pilot, materialize URLs only for the sampled document IDs.
    target_ids: set[int] | None = None
    if max_docs is not None:
        target_ids = set()
        for batch in pq.ParquetFile(chunks_path).iter_batches(batch_size=batch_size, columns=["doc_id"]):
            for doc_id in batch.column(0).to_pylist():
                target_ids.add(int(doc_id))
                if len(target_ids) >= max_docs:
                    break
            if len(target_ids) >= max_docs:
                break
    url_table = pq.read_table(corpus_path, columns=["id", "url"])
    if target_ids is not None:
        url_table = url_table.filter(pc.is_in(url_table["id"], value_set=pa.array(list(target_ids), type=url_table["id"].type)))
    url_by_id = dict(zip(url_table.column("id").to_pylist(), url_table.column("url").to_pylist()))
    mention_type = pa.list_(pa.struct([
        ("concept_id", pa.string()), ("concept_type", pa.string()),
        ("matched_alias", pa.string()), ("assertion", pa.string()),
        ("confidence", pa.float32()), ("match_method", pa.string()),
    ]))
    chunk_schema = pa.schema([
        ("doc_id", pa.int64()), ("chunk_index", pa.int32()), ("language", pa.string()),
        ("concepts", pa.list_(pa.string())), ("mentions", mention_type),
        ("intents", pa.list_(pa.string())), ("specialties", pa.list_(pa.string())),
        ("annotation_version", pa.string()),
    ])
    doc_schema = pa.schema([
        ("doc_id", pa.int64()), ("domain", pa.string()), ("source_type", pa.string()),
        ("language", pa.string()), ("concepts", pa.list_(pa.string())),
        ("intents", pa.list_(pa.string())), ("specialties", pa.list_(pa.string())),
        ("annotation_version", pa.string()), ("chunk_count", pa.int32()),
    ])

    chunk_writer = pq.ParquetWriter(chunks_out, chunk_schema, compression="zstd")
    doc_writer = pq.ParquetWriter(docs_out, doc_schema, compression="zstd")
    chunk_rows: list[dict[str, Any]] = []
    doc_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "taxonomy_version": taxonomy.version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "input_chunks": str(chunks_path), "input_corpus": str(corpus_path),
        "max_docs": max_docs, "documents": 0, "chunks": 0,
        "concept_chunk_counts": Counter(), "intent_chunk_counts": Counter(),
        "language_chunk_counts": Counter(), "source_type_doc_counts": Counter(),
    }
    current_doc: dict[str, Any] | None = None
    seen_docs = 0

    def flush_doc() -> None:
        nonlocal current_doc
        if current_doc is None:
            return
        domain, source_type = classify_source(current_doc["url"])
        lang = Counter(current_doc["languages"]).most_common(1)[0][0] if current_doc["languages"] else "unknown"
        doc_rows.append({
            "doc_id": current_doc["doc_id"], "domain": domain, "source_type": source_type,
            "language": lang, "concepts": sorted(current_doc["concepts"]),
            "intents": sorted(current_doc["intents"]), "specialties": sorted(current_doc["specialties"]),
            "annotation_version": taxonomy.version, "chunk_count": current_doc["chunk_count"],
        })
        summary["source_type_doc_counts"][source_type] += 1
        if len(doc_rows) >= 10_000:
            doc_writer.write_table(pa.Table.from_pylist(doc_rows, schema=doc_schema))
            doc_rows.clear()
        current_doc = None

    parquet = pq.ParquetFile(chunks_path)
    try:
        for batch in parquet.iter_batches(batch_size=batch_size, columns=["doc_id", "chunk_index", "title", "text"]):
            for row in batch.to_pylist():
                doc_id = int(row["doc_id"])
                if current_doc is None or current_doc["doc_id"] != doc_id:
                    flush_doc()
                    if max_docs is not None and seen_docs >= max_docs:
                        break
                    seen_docs += 1
                    current_doc = {
                        "doc_id": doc_id, "url": url_by_id.get(doc_id, ""),
                        "concepts": set(), "intents": set(), "specialties": set(),
                        "languages": [], "chunk_count": 0,
                    }
                mentions, concepts, intents, specialties = _annotation(row["text"], taxonomy)
                language = detect_language(row["text"])
                current_doc["concepts"].update(concepts)
                current_doc["intents"].update(intents)
                current_doc["specialties"].update(specialties)
                current_doc["languages"].append(language)
                current_doc["chunk_count"] += 1
                chunk_rows.append({
                    "doc_id": doc_id, "chunk_index": int(row["chunk_index"]),
                    "language": language, "concepts": concepts, "mentions": mentions,
                    "intents": intents, "specialties": specialties,
                    "annotation_version": taxonomy.version,
                })
                summary["chunks"] += 1
                summary["language_chunk_counts"][language] += 1
                summary["concept_chunk_counts"].update(concepts)
                summary["intent_chunk_counts"].update(intents)
            if chunk_rows:
                chunk_writer.write_table(pa.Table.from_pylist(chunk_rows, schema=chunk_schema))
                chunk_rows.clear()
            if summary["chunks"] and summary["chunks"] % 50_000 < batch_size:
                print(f"annotated docs={seen_docs:,}/{max_docs or 'all'}, "
                      f"chunks={summary['chunks']:,}", flush=True)
            if max_docs is not None and seen_docs >= max_docs and current_doc is None:
                break
    finally:
        flush_doc()
        if chunk_rows:
            chunk_writer.write_table(pa.Table.from_pylist(chunk_rows, schema=chunk_schema))
        if doc_rows:
            doc_writer.write_table(pa.Table.from_pylist(doc_rows, schema=doc_schema))
        chunk_writer.close()
        doc_writer.close()

    summary["documents"] = seen_docs
    for key in ("concept_chunk_counts", "intent_chunk_counts", "language_chunk_counts", "source_type_doc_counts"):
        summary[key] = dict(summary[key])
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-docs", type=int, default=None, help="stop after N consecutive corpus documents; pilot only")
    parser.add_argument("--batch-size", type=int, default=20_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    report = annotate(args.chunks, args.corpus, args.config, args.out_dir,
                      args.max_docs, args.batch_size, args.overwrite)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

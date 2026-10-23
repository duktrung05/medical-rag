"""Freeze reviewed Phase 2 retrieval qrels and a shared evaluation corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from urllib.parse import urlsplit

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.workflows.vibiomir.build_phase2_passages import (
    _language_group,
    _length_bin,
    _paragraph_profile,
    _MEASUREMENT,
)
from scripts.workflows.vibiomir.review_phase2_pilot import _load_documents
from src.medrag.data.phase2.schema import PILOT_SCHEMA
from src.medrag.data.phase2.structure import has_reliable_heading


# Every accepted Vietnamese proposal was inspected against its question and the
# current immutable Phase 1 source. Topic-only and answerless proposals remain
# in the review ledger as rejected rather than becoming qrels.
ACCEPTED_VI_QUERY_IDS = {
    1, 18, 23, 43, 49, 53, 65, 130, 135, 141, 171, 246, 248, 263, 277,
    327, 328, 329, 334, 344, 353, 366, 367, 385, 396, 398, 441, 457, 471,
    502, 511, 517, 535, 545, 579, 586, 609, 610, 615, 629, 661, 672, 692,
    696, 736, 743, 747, 749, 796, 834, 847, 883, 885, 893, 910, 913, 940,
    947, 967, 991, 1013, 1023, 1031, 1032, 1046, 1068, 1076, 1098, 1126,
    1148, 1150, 1155, 1156, 1157, 1159, 1165, 1181,
}


# These are natural-language questions authored from independently reviewed
# EN/ZH cases. Markers select a short exact evidence span in current Phase 1.
MULTILINGUAL_CASES = [
    {
        "query_id": 2_000_001, "language": "en", "doc_id": 5_844_719,
        "query": "What factors predicted non-disclosure of HIV status to sexual partners?",
        "start": "Predictors of non-disclosure included lack of correct knowledge about HIV",
        "end": "predicted non-disclosure.",
    },
    {
        "query_id": 2_000_002, "language": "en", "doc_id": 6_924_031,
        "query": "When should children who need two influenza vaccine doses receive them?",
        "start": "As previously recommended, children who need 2 doses",
        "end": "received by the end of October.",
    },
    {
        "query_id": 2_000_003, "language": "en", "doc_id": 5_996_753,
        "query": "What daily tocotrienol dose was tested for hair growth?",
        "start": "Twenty one volunteers were randomly assigned",
        "end": "placebo capsule orally.",
    },
    {
        "query_id": 2_000_004, "language": "en", "doc_id": 7_294_391,
        "query": "Which liver enzymes were elevated and what were their measured values?",
        "start": "Additionally, altered liver function tests were remarkable",
        "end": "were within normal limits.",
    },
    {
        "query_id": 2_000_005, "language": "en", "doc_id": 6_091_701,
        "query": "How much did counselor-initiated testing increase HIV testing?",
        "start": "Counselor-initiated PITC significantly increased HIV testing",
        "end": "receiving a new diagnosis of HIV.",
    },
    {
        "query_id": 2_000_006, "language": "en", "doc_id": 6_887_513,
        "query": "Does a follicular phase longer than 15 days affect live birth outcomes?",
        "start": "SUMMARY ANSWER:",
        "end": "not associated with pregnancy and live birth outcomes.",
    },
    {
        "query_id": 2_000_007, "language": "en", "doc_id": 5_884_216,
        "query": "What signs and symptoms can chronic venous disease cause?",
        "start": "Chronic venous disease, reviewed herein",
        "end": "and ulceration.",
    },
    {
        "query_id": 2_000_008, "language": "en", "doc_id": 7_147_941,
        "query": "Which pregnancy characteristics were associated with spontaneous preterm birth?",
        "start": "Characteristics not varying in pregnancy associated with increased risk",
        "end": "male child (1.10, 1.02 to 1.17).",
    },
    {
        "query_id": 2_000_009, "language": "en", "doc_id": 6_165_181,
        "query": "Why could bleeding outcomes from the two RCTs not be pooled?",
        "start": "We deemed only one study to be at low risk of bias",
        "end": "considerable statistical heterogeneity between studies.",
    },
    {
        "query_id": 2_000_010, "language": "en", "doc_id": 5_493_174,
        "query": "Was normal sperm morphology associated with ICSI fertilization rates?",
        "start": "There was a highly significant correlation",
        "end": "(P = 0.007).",
    },
    {
        "query_id": 2_000_011, "language": "zh", "doc_id": 848_604,
        "query": "乙肝病毒持续感染与肝癌有什么关系？",
        "start": "肝癌的发生，很大一部分跟乙肝有关。",
        "end": "乙肝病毒的持续感染是患肝癌的主要原因之一。",
    },
    {
        "query_id": 2_000_012, "language": "zh", "doc_id": 1_494_285,
        "query": "哪些体肺分流术已经在临床上弃用？",
        "start": "而降主动脉与左肺动脉分流术",
        "end": "已被临床上弃用[3]。",
    },
    {
        "query_id": 2_000_013, "language": "zh", "doc_id": 2_132_134,
        "query": "老年人口干常见的阴虚和阳虚机制分别是什么？",
        "start": "老年人的口干阴虚者",
        "end": "阳虚者是因为津液不能上承所致。",
    },
    {
        "query_id": 2_000_014, "language": "zh", "doc_id": 794_254,
        "query": "大量饮酒会怎样影响精子质量？",
        "start": "适度饮酒。",
        "end": "女性减半的推荐量来执行。",
    },
    {
        "query_id": 2_000_015, "language": "zh", "doc_id": 877_312,
        "query": "痛风患者每天应限制多少蛋白质？",
        "start": "痛风症者应减少到150mg",
        "end": "热量要较正常低10～15%",
    },
    {
        "query_id": 2_000_016, "language": "zh", "doc_id": 4_024_465,
        "query": "下肢动脉狭窄介入术后应如何随访和控制风险？",
        "start": "控制血压、血糖、血脂。",
        "end": "需要及时到医院来处理。",
    },
    {
        "query_id": 2_000_017, "language": "zh", "doc_id": 785_929,
        "query": "这组患儿的血红蛋白和白细胞检查结果如何？",
        "start": "实验室血常规检查：",
        "end": ">20x10/L的为8例。",
    },
    {
        "query_id": 2_000_018, "language": "zh", "doc_id": 840_214,
        "query": "每天摄入多少红肉通常不会增加文中所述风险？",
        "start": "中国居民膳食指南推荐",
        "end": "是不会带来这种危险的。",
    },
    {
        "query_id": 2_000_019, "language": "zh", "doc_id": 2_538_681,
        "query": "昏迷、呼吸慢、针尖样瞳孔的患者最终诊断是什么？",
        "start": "该患者既往有长期吸食海洛因的历史",
        "end": "确诊为阿片类药物吸食过量中毒。",
    },
    {
        "query_id": 2_000_020, "language": "zh", "doc_id": 3_982_492,
        "query": "继发性血小板增多症如何评估病因并治疗？",
        "start": "在继发性血小板增多症中",
        "end": "治疗潜在疾病通常可使血小板计数恢复正常。",
    },
]


def _exact_span(text: str, start_marker: str, end_marker: str) -> str:
    start = text.find(start_marker)
    if start < 0 or text.find(start_marker, start + 1) >= 0:
        raise ValueError(f"Start marker is absent or ambiguous: {start_marker!r}")
    end = text.find(end_marker, start)
    if end < 0:
        raise ValueError(f"End marker is absent after start: {end_marker!r}")
    evidence = text[start:end + len(end_marker)]
    if text.count(evidence) != 1:
        raise ValueError("Resolved evidence is not unique")
    return evidence


def _pilot_row(document: dict) -> dict:
    text = document["normalized_text"]
    language, mixed = _language_group(document["detected_language"], text)
    score = int.from_bytes(
        hashlib.sha256(f"phase2-retrieval-benchmark-v2\0{document['doc_id']}".encode()).digest()[:8],
        "big",
    )
    return {
        "doc_id": int(document["doc_id"]),
        "official_url": document["official_url"],
        "source_kind": document["source_kind"],
        "domain": urlsplit(document["official_url"]).netloc.lower(),
        "language": language,
        "length_chars": len(text),
        "length_bin": _length_bin(len(text)),
        "has_heading": has_reliable_heading(document.get("title") or "", text),
        "paragraph_profile": _paragraph_profile(text),
        "mixed_language": mixed,
        "medical_measurement": bool(_MEASUREMENT.search(text)),
        "selection_score": score,
    }


def prepare(args: argparse.Namespace) -> dict:
    proposals = json.loads(args.proposals.read_text(encoding="utf-8"))
    proposal_by_id = {int(row["query_id"]): row for row in proposals}
    if not ACCEPTED_VI_QUERY_IDS <= proposal_by_id.keys():
        raise ValueError("Reviewed Vietnamese query IDs are missing from proposals")

    base = json.loads(args.base_qrels.read_text(encoding="utf-8"))["queries"]
    multilingual_docs = _load_documents(args.candidate, {row["doc_id"] for row in MULTILINGUAL_CASES})
    labels = []
    for row in base:
        labels.append({**row, "query_language": "vi", "provenance": "manual_pilot_qrel"})
    for query_id in sorted(ACCEPTED_VI_QUERY_IDS):
        row = proposal_by_id[query_id]
        labels.append({
            "query_id": query_id,
            "query": row["query"],
            "query_language": "vi",
            "doc_id": int(row["doc_id"]),
            "relevance_grade": 2,
            "evidence_text": row["evidence_text"],
            "notes": "Manually accepted after query-to-source evidence inspection.",
            "provenance": "official_query_manual_source_alignment",
        })
    for spec in MULTILINGUAL_CASES:
        evidence = _exact_span(
            multilingual_docs[spec["doc_id"]]["normalized_text"], spec["start"], spec["end"]
        )
        labels.append({
            "query_id": spec["query_id"],
            "query": spec["query"],
            "query_language": spec["language"],
            "doc_id": spec["doc_id"],
            "relevance_grade": 3,
            "evidence_text": evidence,
            "notes": "Natural question and exact evidence manually authored from a reviewed semantic case.",
            "provenance": "manual_multilingual_holdout",
        })

    ids = [int(row["query_id"]) for row in labels]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate query IDs in frozen qrels")
    all_docs = _load_documents(args.candidate, {int(row["doc_id"]) for row in labels})
    for row in labels:
        evidence = row["evidence_text"]
        if all_docs[int(row["doc_id"])]["normalized_text"].count(evidence) != 1:
            raise ValueError(f"Evidence is absent or ambiguous for query {row['query_id']}")

    language_counts = {}
    for row in labels:
        language_counts[row["query_language"]] = language_counts.get(row["query_language"], 0) + 1
    frozen = {
        "format": "vibiomir-phase2-retrieval-qrels-v2",
        "label_policy": "manual_query_relevance_with_unique_exact_phase1_evidence",
        "reviewed_by": "Codex manual inspection; no automatic accept-all",
        "query_count": len(labels),
        "language_counts": language_counts,
        "queries": sorted(labels, key=lambda row: int(row["query_id"])),
    }
    args.output_qrels.parent.mkdir(parents=True, exist_ok=True)
    args.output_qrels.write_text(
        json.dumps(frozen, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    review_rows = []
    for row in sorted(proposals, key=lambda value: int(value["query_id"])):
        accepted = int(row["query_id"]) in ACCEPTED_VI_QUERY_IDS
        review_rows.append({
            "query_id": int(row["query_id"]),
            "doc_id": int(row["doc_id"]),
            "decision": "accepted" if accepted else "rejected",
            "review_basis": "manual_query_evidence_inspection",
            "reason": (
                "Exact source span materially supports the query."
                if accepted else
                "Candidate span was topic-only, answerless, mismatched, or too weak for a qrel."
            ),
            "query": row["query"],
            "candidate_evidence": row["evidence_text"],
        })
    args.output_review.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(review_rows), args.output_review, compression="zstd")

    base_rows = pq.read_table(args.base_pilot_manifest, schema=PILOT_SCHEMA).to_pylist()
    by_doc = {int(row["doc_id"]): row for row in base_rows}
    for doc_id, document in all_docs.items():
        by_doc.setdefault(doc_id, _pilot_row(document))
    pilot_rows = [by_doc[doc_id] for doc_id in sorted(by_doc)]
    args.output_pilot_manifest.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(pilot_rows, schema=PILOT_SCHEMA),
        args.output_pilot_manifest,
        compression="zstd",
    )
    return {
        "queries": len(labels),
        "languages": language_counts,
        "relevant_documents": len(all_docs),
        "evaluation_documents": len(pilot_rows),
        "reviewed_vi_candidates": len(review_rows),
        "accepted_vi_candidates": len(ACCEPTED_VI_QUERY_IDS),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--proposals", type=Path, required=True)
    parser.add_argument("--base-qrels", type=Path, required=True)
    parser.add_argument("--base-pilot-manifest", type=Path, required=True)
    parser.add_argument("--output-qrels", type=Path, required=True)
    parser.add_argument("--output-review", type=Path, required=True)
    parser.add_argument("--output-pilot-manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = prepare(args)
    except (OSError, ValueError, pa.ArrowException) as exc:
        print(f"Preparing Phase 2 retrieval benchmark failed: {exc}")
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from src.medrag.retrieval.icd10 import (
    ICD10QueryMapper, detect_intents, rerank_candidates, rerank_candidates_cached,
)


TAXONOMY = [
    {"code": "I10", "title": "Essential (primary) hypertension"},
    {"code": "M10", "title": "Gout"},
]
ALIASES = [
    {"code": "I10", "language": "vi", "alias": "tăng huyết áp"},
    {"code": "I10", "language": "vi", "alias": "cao huyết áp"},
    {"code": "M10", "language": "zh", "alias": "痛风"},
]


def test_mapper_is_conservative_and_multilingual():
    mapper = ICD10QueryMapper(TAXONOMY, ALIASES)
    assert [m.code for m in mapper.map_query("Điều trị tăng huyết áp?", "vi")] == ["I10"]
    assert [m.code for m in mapper.map_query("痛风患者怎么办？", "zh")] == ["M10"]
    assert mapper.map_query("mệt mỏi không rõ nguyên nhân", "vi") == []


def test_expansion_keeps_original_and_adds_verified_terms():
    mapper = ICD10QueryMapper(TAXONOMY, ALIASES)
    matches = mapper.map_query("tăng huyết áp", "vi")
    expanded = mapper.expansion("tăng huyết áp", "vi", matches)
    assert expanded.startswith("tăng huyết áp")
    assert "cao huyết áp" in expanded
    assert "Essential (primary) hypertension" in expanded


def test_reranking_never_filters_candidates():
    mapper = ICD10QueryMapper(TAXONOMY, ALIASES)
    candidates = [
        {"doc_id": 1, "passage_id": "a", "passage_text": "thông tin chung", "baseline_score": 10.0},
        {"doc_id": 2, "passage_id": "b", "passage_text": "điều trị cao huyết áp", "baseline_score": 9.5},
    ]
    result = rerank_candidates(
        candidates, query_codes={"I10"}, query_intents={"treatment"}, mapper=mapper,
        language="vi", concept_weight=.1, intent_weight=.01,
    )
    assert len(result) == len(candidates)
    assert result[0]["passage_id"] == "b"


def test_intent_detection():
    assert detect_intents("Nguyên nhân và cách điều trị là gì?", "vi") == {"causes", "treatment"}


def test_cached_reranking_preserves_candidate_set_and_tail():
    candidates = [
        {"doc_id": 1, "passage_id": "a", "baseline_score": 10.0, "score": 10.0},
        {"doc_id": 2, "passage_id": "b", "baseline_score": 9.5, "score": 9.5},
        {"doc_id": 3, "passage_id": "c", "baseline_score": 9.0, "score": 9.0},
    ]
    result = rerank_candidates_cached(
        candidates, query_codes={"I10"}, passage_codes={"b": {"I10"}},
        concept_weight=.1, rescore_depth=2,
    )
    assert [row["passage_id"] for row in result] == ["b", "a", "c"]
    assert {row["passage_id"] for row in result} == {"a", "b", "c"}
    assert result[-1]["baseline_score"] == 9.0

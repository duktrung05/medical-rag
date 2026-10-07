import unittest
from pathlib import Path

from r2ai.taxonomy import (
    DEFAULT_CONFIG,
    Taxonomy,
    classify_source,
    metadata_overlap,
    rrf_rerank,
)


class TaxonomyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = Taxonomy.load(DEFAULT_CONFIG)

    def test_config_has_unique_stable_concepts(self):
        ids = [concept["id"] for concept in self.taxonomy.concepts]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertGreaterEqual(len(ids), 20)

    def test_vietnamese_accented_unaccented_and_chinese_aliases(self):
        self.assertIn("disease.diabetes", self.taxonomy.profile("đái tháo đường")["concepts"])
        self.assertIn("disease.diabetes", self.taxonomy.profile("dai thao duong")["concepts"])
        self.assertIn("disease.diabetes", self.taxonomy.profile("糖尿病")["concepts"])

    def test_single_word_alias_keeps_vietnamese_tone_distinction(self):
        self.assertIn("anatomy.heart", self.taxonomy.profile("tim đập nhanh")["concepts"])
        self.assertNotIn("anatomy.heart", self.taxonomy.profile("tìm bác sĩ")["concepts"])
        self.assertNotIn("symptom.cough", self.taxonomy.profile("họ đang nói chuyện")["concepts"])
        self.assertIn("symptom.cough", self.taxonomy.profile("ho khan kéo dài")["concepts"])

    def test_negation_is_retained_as_mention_but_not_active_query_concept(self):
        profile = self.taxonomy.profile("không mắc tiểu đường")
        self.assertEqual(profile["concepts"], [])
        self.assertEqual(profile["mentions"][0]["assertion"], "negated")
        self.assertIsNone(profile["mentions"][0]["confidence"])

    def test_test_interpretation_requires_a_test_or_vital_concept(self):
        self.assertIn("test_interpretation", self.taxonomy.profile("beta-HCG là như thế nào?")["intents"])
        self.assertNotIn("test_interpretation", self.taxonomy.profile("bao nhiêu tuổi?")["intents"])

    def test_query_expansion_is_bounded_and_curated(self):
        aliases = self.taxonomy.aliases_for("GERD là gì")
        self.assertLessEqual(len(aliases), 4)
        self.assertNotIn("trào ngược dạ dày thực quản", self.taxonomy.aliases_for("không mắc GERD"))

    def test_metadata_overlap_and_rank_fusion_are_neutral_without_match(self):
        query = {"concepts": ["disease.diabetes"], "intents": ["treatment"], "specialties": ["endocrinology"]}
        self.assertEqual(metadata_overlap(query, {}), 0.0)
        docs = {1: {}, 2: {"concepts": ["disease.diabetes"], "intents": ["treatment"], "specialties": ["endocrinology"]}}
        base = [1, 2]
        self.assertEqual(rrf_rerank(base, query, docs, weight=0), base)
        self.assertEqual(rrf_rerank(base, query, docs, weight=1)[0], 2)

    def test_domain_category_is_descriptive_not_a_quality_score(self):
        self.assertEqual(classify_source("https://moh.gov.vn/health")[1], "government")
        self.assertEqual(classify_source("https://ask.example.vn/question")[1], "qa")


if __name__ == "__main__":
    unittest.main()

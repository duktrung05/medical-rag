from __future__ import annotations

import unittest
import gzip
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.medrag.chunking.passage_builder import build_passages
from src.medrag.core.contracts import Source
from src.medrag.core.pipeline import Pipeline
from src.medrag.core.config import load_config
from src.medrag.data.corpus_registry import CorpusRegistry, CorpusVersion
from src.medrag.data.extraction import extract_html
from src.medrag.data.metadata import language_hint
from src.medrag.data.prepared import load_prepared, prepare_from_official, write_prepared
from src.medrag.evidence.verifier import verify
from src.medrag.retrieval.core import tokens
from src.medrag.adapters.vibiomir_submission import records


def source(doc_id, text):
    return Source("test-v1", doc_id, f"https://example.org/{doc_id}", "", text, "fixture",
                  sha256(text.encode()).hexdigest())


class V4CoreTests(unittest.TestCase):
    def test_exact_offsets_multiple_languages_and_tail(self):
        for text in ("Triệu chứng sốt. Điều trị thuốc 5 mg. " * 20,
                     "高血压症状包括头痛。治疗需要观察。" * 20,
                     "Symptoms may include fever. Treatment uses 5 mg. " * 20):
            src = source(1, text)
            parts = build_passages(src, max_tokens=20, overlap_tokens=3)
            self.assertGreater(len(parts), 2)
            self.assertTrue(all(verify(p, src) for p in parts))
            self.assertTrue(text.rstrip().endswith(parts[-1].text[-8:]))
        self.assertGreater(len(tokens("高血压治疗")), 1)

    def test_id_url_pair_must_match(self):
        with self.assertRaises(ValueError):
            CorpusRegistry.assert_identity(1, "https://example.org/new", {1: "https://example.org/old"})

    def test_intent_and_negation_preserved(self):
        a = source(1, "Bệnh A có triệu chứng sốt và đau đầu. Không dùng thuốc 5 mg cho trẻ.")
        b = source(2, "Bệnh A điều trị bằng thuốc 5 mg. Tác dụng phụ là buồn nôn.")
        parts = build_passages(a, max_tokens=15, overlap_tokens=3) + build_passages(b, max_tokens=15, overlap_tokens=3)
        pipeline = Pipeline([a, b], parts)
        symptom = pipeline.retrieve("Bệnh A triệu chứng sốt", 2)
        treatment = pipeline.retrieve("Bệnh A điều trị thuốc 5 mg", 2)
        self.assertEqual(symptom[0]["official_id"], 1)
        self.assertEqual(treatment[0]["official_id"], 2)
        self.assertIn("Không", a.text)
        self.assertIn("5 mg", b.text)

    def test_duplicate_text_is_document_scoped(self):
        a, b = source(1, "Shared evidence text."), source(2, "Shared evidence text.")
        p1, p2 = build_passages(a)[0], build_passages(b)[0]
        self.assertNotEqual(p1.passage_id, p2.passage_id)
        self.assertFalse(verify(p1, b))

    def test_mixed_corpus_rejected(self):
        a = source(1, "one")
        b = Source("test-v2", 2, "https://example.org/2", "", "two", "fixture", "hash")
        with self.assertRaises(ValueError):
            Pipeline([a, b], build_passages(a) + build_passages(b))

    def test_config_registry_and_submission_schema(self):
        root = Path(__file__).resolve().parents[2]
        config = load_config(root / "configs/v4_core.yaml")
        version = CorpusRegistry.load(root / config["registry"]).get(config["corpus_version"])
        self.assertEqual(version.rows, 1220598)
        src = source(7, "Exact source evidence")
        result = Pipeline([src], build_passages(src)).retrieve("source evidence")
        submission = records([5], {5: result})
        self.assertEqual(submission[0]["relevant_chunks"][0]["doc_id"], 7)
        self.assertEqual(submission[0]["relevant_chunks"][0]["chunk_text"], src.text)

    def test_dense_and_reranker_contracts(self):
        class Encoder:
            def encode(self, texts):
                return [[float("treatment" in x), float("symptom" in x)] for x in texts]
        class Reranker:
            def score(self, query, passages):
                return [float("treatment" in text) for text in passages]
        a, b = source(1, "symptom fever"), source(2, "treatment medicine")
        pipe = Pipeline([a, b], build_passages(a) + build_passages(b), Encoder(), Reranker())
        self.assertEqual(pipe.retrieve("treatment", 2)[0]["official_id"], 2)

    def test_html_structure_and_chinese_encoding(self):
        raw = '<meta charset="gb18030"><h1>高血压</h1><p>症状包括头痛。</p><li>治疗观察</li><nav>广告</nav>'.encode("gb18030")
        title, text = extract_html(raw)
        self.assertEqual(title, "高血压")
        self.assertIn("症状包括头痛。", text)
        self.assertIn("治疗观察", text)
        self.assertNotIn("广告", text)
        self.assertEqual(language_hint(text).value, "zh")
        self.assertFalse(language_hint(text).verified)

    def test_prepared_snapshot_pins_version_and_detects_tampering(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            version = CorpusVersion("test-v1", "revision-1", root / "corpus.parquet", "abc", 1)
            registry = CorpusRegistry({version.name: version})
            output = root / "v4" / "prepared.jsonl"
            write_prepared([source(17, "End-of-article evidence on treatment.")], output, version)
            result = load_prepared(output, registry).retrieve("treatment")
            self.assertEqual(result[0]["official_id"], 17)
            self.assertEqual(result[0]["evidence"][0]["text"], "End-of-article evidence on treatment.")
            with self.assertRaises(FileExistsError):
                write_prepared([source(17, "text")], output, version)
            output.write_text(output.read_text() + "corrupted", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_prepared(output, registry)

    def test_official_reuse_requires_matching_historical_url(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pages = root / "pages"
            for official_id in (10, 11):
                path = pages / f"{official_id % 1000:03d}" / f"{official_id}.gz"
                path.parent.mkdir(parents=True)
                path.write_bytes(gzip.compress(f"<h1>Doc {official_id}</h1><p>Medical evidence.</p>".encode()))
            official = CorpusVersion("official-test", "revision", root / "official.parquet", "official-hash", 2)
            legacy = CorpusVersion("legacy-test", "revision", root / "legacy.parquet", "legacy-hash", 2)
            pairs = {official.path: [(10, "https://example.org/new"), (11, "https://example.org/same")],
                     legacy.path: [(10, "https://example.org/old"), (11, "https://example.org/same")]}
            with patch("src.medrag.data.prepared.read_corpus_mapping", side_effect=lambda path: iter(pairs[path])), \
                 patch.object(CorpusRegistry, "verify_file", return_value=True):
                output = root / "v4" / "official.jsonl"
                manifest = prepare_from_official(official, legacy, pages, output, 2)
                self.assertEqual(manifest["sources"], 1)
                self.assertEqual(manifest["id_url_mismatches"], 1)
                registry = CorpusRegistry({official.name: official, legacy.name: legacy})
                result = load_prepared(output, registry).retrieve("Medical evidence")
                self.assertEqual(result[0]["official_id"], 11)


if __name__ == "__main__":
    unittest.main()


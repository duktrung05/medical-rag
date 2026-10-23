from __future__ import annotations

from src.medrag.data.phase3.near import (
    hash_shingles,
    lsh_candidates,
    medical_guard_reasons,
    minhash_signature,
    shingle_text,
    similarity_scores,
)


def _hashed(text: str, language: str):
    shingles, strategy, size = shingle_text(text, language)
    return hash_shingles(shingles), strategy, size


def test_vietnamese_shingling_uses_syllable_tokens():
    values, strategy, size = shingle_text(
        "Bệnh nhân cần theo dõi đường huyết định kỳ mỗi ngày", "vi"
    )
    assert strategy == "vietnamese_syllable"
    assert size == 5
    assert values


def test_chinese_shingling_uses_character_ngrams():
    values, strategy, size = shingle_text(
        "糖尿病患者应定期监测血糖水平并遵循医生建议", "zh"
    )
    assert strategy == "chinese_character"
    assert size == 5
    assert values


def test_english_shingling_uses_word_ngrams():
    values, strategy, size = shingle_text(
        "Patients should monitor blood glucose levels every day", "en"
    )
    assert strategy == "english_word"
    assert size == 5
    assert values


def test_minhash_and_lsh_are_deterministic_for_similar_documents():
    left, strategy, _ = _hashed(
        "alpha beta gamma delta epsilon zeta eta theta iota kappa", "en"
    )
    right, _, _ = _hashed(
        "alpha beta gamma delta epsilon zeta eta theta iota changed", "en"
    )
    left_signature = minhash_signature(left)
    assert left_signature == minhash_signature(left)
    records = [
        {"signature": left_signature, "shingle_strategy": strategy},
        {"signature": minhash_signature(right), "shingle_strategy": strategy},
    ]
    candidates, overflow = lsh_candidates(records)
    assert candidates == {(0, 1)}
    assert overflow == 0


def test_jaccard_and_containment_are_exactly_verified():
    left, _, _ = _hashed("one two three four five six seven", "en")
    right, _, _ = _hashed("one two three four five six eight", "en")
    shared, jaccard, left_containment, right_containment = similarity_scores(left, right)
    assert shared > 0
    assert 0 < jaccard < 1
    assert 0 < left_containment < 1
    assert 0 < right_containment < 1


def test_medical_negation_not_merged():
    positive = "This medicine is recommended for adults with hypertension."
    negative = "This medicine is not recommended for adults with hypertension."
    assert "medical_negation_mismatch" in medical_guard_reasons(positive, negative)


def test_different_drug_dosage_not_merged():
    low = "The recommended dose is 5 mg per day for treatment."
    high = "The recommended dose is 10 mg per day for treatment."
    assert "drug_dosage_mismatch" in medical_guard_reasons(low, high)


def test_same_disease_different_intent_not_merged():
    symptoms = "Diabetes symptoms include thirst and fatigue."
    treatment = "Diabetes treatment includes insulin therapy."
    assert "medical_intent_mismatch" in medical_guard_reasons(symptoms, treatment)

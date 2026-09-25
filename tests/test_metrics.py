"""Tests for Precision, Recall, F2, and Macro Evaluator."""

import pytest

from src.data.loader import DocumentChunkMap
from src.data.schema import GroundTruthRecord, PredictionRecord
from src.evaluation.evaluator import Evaluator
from src.evaluation.fbeta import calculate_fbeta
from src.evaluation.precision import calculate_precision
from src.evaluation.recall import calculate_recall


def test_precision_recall_perfect():
    actual = ["A", "B", "C"]
    predicted = ["A", "B", "C"]
    assert calculate_precision(actual, predicted) == 1.0
    assert calculate_recall(actual, predicted) == 1.0
    assert calculate_fbeta(1.0, 1.0, beta=2.0) == 1.0


def test_empty_predictions():
    actual = ["A", "B"]
    predicted = []
    assert calculate_precision(actual, predicted) == 0.0
    assert calculate_recall(actual, predicted) == 0.0
    assert calculate_fbeta(0.0, 0.0, beta=2.0) == 0.0


def test_no_overlap():
    actual = ["A", "B"]
    predicted = ["X", "Y", "Z"]
    assert calculate_precision(actual, predicted) == 0.0
    assert calculate_recall(actual, predicted) == 0.0
    assert calculate_fbeta(0.0, 0.0, beta=2.0) == 0.0


def test_f2_exact_calculation():
    # Actual: ["D1", "D2"] -> size 2
    # Predicted: ["D1", "D3", "D4"] -> size 3, TP = 1
    # P = 1/3, R = 1/2
    # F2 = 5 * (1/3) * (1/2) / (4 * (1/3) + 1/2) = (5/6) / (11/6) = 5/11
    actual = ["D1", "D2"]
    predicted = ["D1", "D3", "D4"]

    p = calculate_precision(actual, predicted)
    r = calculate_recall(actual, predicted)
    f2 = calculate_fbeta(p, r, beta=2.0)

    assert pytest.approx(p, 1e-6) == 1 / 3
    assert pytest.approx(r, 1e-6) == 1 / 2
    assert pytest.approx(f2, 1e-6) == 5 / 11


def test_macro_evaluator():
    evaluator = Evaluator(beta=2.0)

    # Q1: Perfect match (P=1, R=1, F2=1)
    # Q2: Partial match (P=1/3, R=1/2, F2=5/11)
    gt = [
        GroundTruthRecord(id="Q1", relevant_docs=["D1"], relevant_chunks=["C1"]),
        GroundTruthRecord(id="Q2", relevant_docs=["D1", "D2"], relevant_chunks=["C1", "C2"]),
    ]
    preds = [
        PredictionRecord(id="Q1", relevant_docs=["D1"], relevant_chunks=["C1"]),
        PredictionRecord(id="Q2", relevant_docs=["D1", "D3", "D4"], relevant_chunks=["C1", "C3", "C4"]),
    ]

    summary = evaluator.evaluate(ground_truth=gt, predictions=preds)

    expected_macro_doc_p = (1.0 + (1 / 3)) / 2.0
    expected_macro_doc_r = (1.0 + (1 / 2)) / 2.0
    expected_macro_doc_f2 = (1.0 + (5 / 11)) / 2.0

    assert summary.num_queries == 2
    assert pytest.approx(summary.doc_precision, 1e-5) == expected_macro_doc_p
    assert pytest.approx(summary.doc_recall, 1e-5) == expected_macro_doc_r
    assert pytest.approx(summary.doc_f2, 1e-5) == expected_macro_doc_f2
    assert pytest.approx(summary.chunk_f2, 1e-5) == expected_macro_doc_f2


def _truth(query_id="Q1"):
    return GroundTruthRecord(id=query_id, relevant_docs=["D1"], relevant_chunks=["C1"])


def _prediction(query_id="Q1"):
    return PredictionRecord(id=query_id, relevant_docs=["D1"], relevant_chunks=["C1"])


def test_evaluator_rejects_duplicate_and_mismatched_query_ids():
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="Ground truth has duplicate query ID: Q1"):
        evaluator.evaluate([_truth(), _truth()], [_prediction()])
    with pytest.raises(ValueError, match="Prediction has duplicate query ID: Q1"):
        evaluator.evaluate([_truth()], [_prediction(), _prediction()])
    with pytest.raises(ValueError, match="missing=\\['Q1'\\], extra=\\[\\]"):
        evaluator.evaluate([_truth()], [])
    with pytest.raises(ValueError, match="extra=\\['Q2'\\]"):
        evaluator.evaluate([_truth()], [_prediction(), _prediction("Q2")])
    with pytest.raises(ValueError, match="mapping key"):
        evaluator.evaluate({"wrong": _truth()}, {"Q1": _prediction()})


def test_evaluator_rejects_empty_ground_truth_and_duplicate_prediction_ids():
    evaluator = Evaluator()
    with pytest.raises(ValueError, match="Ground truth dataset cannot be empty"):
        evaluator.evaluate([], [])
    duplicate_docs = _prediction().model_copy(update={"relevant_docs": ["D1", "D1"]})
    duplicate_chunks = _prediction().model_copy(update={"relevant_chunks": ["C1", "C1"]})
    with pytest.raises(ValueError, match="duplicate document IDs: D1"):
        evaluator.evaluate([_truth()], [duplicate_docs])
    with pytest.raises(ValueError, match="duplicate chunk IDs: C1"):
        evaluator.evaluate([_truth()], [duplicate_chunks])


def test_evaluator_checks_corpus_ids_and_parent_consistency_when_map_is_given():
    evaluator = Evaluator()
    doc_map = DocumentChunkMap({"C1": "D1", "C2": "D2"}, {"D1": ["C1"], "D2": ["C2"]})
    assert evaluator.evaluate([_truth()], [_prediction()], doc_map=doc_map).macro_f2 == 1.0
    with pytest.raises(ValueError, match="unknown document ID: D3"):
        evaluator.evaluate([_truth()], [_prediction().model_copy(update={"relevant_docs": ["D3"]})], doc_map=doc_map)
    with pytest.raises(ValueError, match="unknown chunk ID: C3"):
        evaluator.evaluate([_truth()], [_prediction().model_copy(update={"relevant_chunks": ["C3"]})], doc_map=doc_map)
    with pytest.raises(ValueError, match="parent inconsistency"):
        evaluator.evaluate([_truth()], [_prediction().model_copy(update={"relevant_chunks": ["C2"]})], doc_map=doc_map)


@pytest.mark.parametrize(
    ("truth_docs", "truth_chunks", "pred_docs", "pred_chunks", "expected"),
    [
        (["D1"], ["C1"], ["D1"], ["C2"], (1, 1, 1, 0, 0, 0, 0.5)),
        (["D1"], ["C1", "C2"], ["D1"], ["C1"], (1, 1, 1, 1, 0.5, 5 / 9, 7 / 9)),
        (["D1"], ["C1"], [], [], (0, 0, 0, 0, 0, 0, 0)),
        (["D1", "D2"], ["C1", "C2", "C3"], ["D1", "D2"], ["C1", "C2", "C3"], (1, 1, 1, 1, 1, 1, 1)),
        (["D1", "D2"], ["C1", "C2"], ["D1"], ["C1", "C3"], (1, 0.5, 5 / 9, 0.5, 0.5, 0.5, 19 / 36)),
    ],
)
def test_evaluator_known_answers(truth_docs, truth_chunks, pred_docs, pred_chunks, expected):
    truth = GroundTruthRecord(id="Q1", relevant_docs=truth_docs, relevant_chunks=truth_chunks)
    prediction = PredictionRecord(id="Q1", relevant_docs=pred_docs, relevant_chunks=pred_chunks)
    result = Evaluator().evaluate([truth], [prediction])
    actual = (result.doc_precision, result.doc_recall, result.doc_f2,
              result.chunk_precision, result.chunk_recall, result.chunk_f2, result.macro_f2)
    assert actual == pytest.approx(expected)

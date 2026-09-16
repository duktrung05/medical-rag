"""Tests for Precision, Recall, F2, and Macro Evaluator."""

import pytest
from src.data.schema import GroundTruthRecord, PredictionRecord
from src.evaluation.evaluator import Evaluator
from src.evaluation.fbeta import calculate_fbeta, calculate_query_metrics
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

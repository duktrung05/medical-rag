"""Cross-language positive reporting for Vietnamese queries."""

from src.data.schema import GroundTruthRecord, QueryRecord
from src.retrieval.exact_dense import RankedChunk
from scripts.run_dense_smoke import cross_language_positive_metrics


def test_reports_positive_recall_separately_for_vi_en_zh():
    queries = [QueryRecord(id="q1", query="bệnh tim")]
    truths = {
        "q1": GroundTruthRecord(
            id="q1",
            relevant_chunks=["vi-1", "en-1", "zh-1"],
            relevant_docs=["d-vi", "d-en", "d-zh"],
        )
    }
    rankings = [[
        RankedChunk("en-1", 0.9, 1),
        RankedChunk("vi-1", 0.8, 2),
        RankedChunk("other", 0.7, 3),
        RankedChunk("zh-1", 0.6, 4),
    ]]
    metrics = cross_language_positive_metrics(
        queries,
        truths,
        rankings,
        {"vi-1": "vi", "en-1": "en", "zh-1": "zh", "other": "en"},
        k=2,
    )

    assert metrics["vi"]["recall_at_k"] == 1.0
    assert metrics["en"]["recall_at_k"] == 1.0
    assert metrics["zh"]["recall_at_k"] == 0.0
    assert metrics["vi"]["mean_first_positive_rank"] == 2.0
    assert metrics["en"]["mean_first_positive_rank"] == 1.0
    assert metrics["zh"]["mean_first_positive_rank"] == 4.0

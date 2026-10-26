from pathlib import Path

from src.medrag.retrieval.disk_bm25 import DiskBM25
from src.medrag.retrieval.sql_aggregate_bm25 import SQLAggregateDiskBM25


def test_sql_aggregate_matches_reference_disk_bm25(tmp_path: Path):
    path = tmp_path / 'bm25.sqlite'
    baseline = DiskBM25(path, 'passage_bm25', 'fixture', create=True)
    baseline.append_batch([
        (3, 'c', 'alpha alpha beta'),
        (1, 'a', 'alpha gamma'),
        (2, 'b', 'beta gamma gamma'),
        (4, 'd', 'unrelated'),
    ])
    baseline.complete()
    optimized = SQLAggregateDiskBM25(path, 'passage_bm25', 'fixture')
    for query in ('alpha', 'beta gamma', 'missing', 'alpha alpha beta'):
        expected = baseline.search(query, 4)
        actual = optimized.search(query, 4)
        assert [(x.official_id, x.passage_id, x.rank) for x in actual] == [
            (x.official_id, x.passage_id, x.rank) for x in expected]
        assert [x.score for x in actual] == [x.score for x in expected]


def test_sql_aggregate_can_prune_to_highest_idf_terms(tmp_path: Path):
    path = tmp_path / 'bm25.sqlite'
    baseline = DiskBM25(path, 'passage_bm25', 'fixture', create=True)
    baseline.append_batch([(1, 'a', 'common rare'), (2, 'b', 'common'), (3, 'c', 'common')])
    baseline.complete()
    optimized = SQLAggregateDiskBM25(
        path, 'passage_bm25', 'fixture', max_query_terms=1)
    assert [(x.official_id, x.passage_id) for x in optimized.search('common rare', 3)] == [(1, 'a')]

from src.medrag.core.contracts import Hit
from src.medrag.retrieval.core import fuse, fuse_with_parent_floor


def hit(parent, passage, channel, rank):
    return Hit(parent, passage, 1.0, channel, rank)


def identity(item):
    return item.official_id, item.passage_id


def test_parent_floor_preserves_mixed_item_prefix_and_supplements_unique_parents():
    rankings = [
        [hit(1, "a", "document_bm25", 1), hit(2, "b", "document_bm25", 2)],
        [hit(1, "c", "dense", 1), hit(3, "d", "dense", 2)],
    ]
    baseline = fuse(rankings, k=60, limit=2)
    candidate = fuse_with_parent_floor(rankings, k=60, item_limit=2, parent_limit=3)

    assert [identity(item) for item in candidate[:2]] == [identity(item) for item in baseline]
    assert {item.official_id for item in candidate} == {1, 2, 3}
    assert len(candidate) == 4


def test_parent_floor_default_path_is_unchanged_when_prefix_has_enough_parents():
    rankings = [[hit(1, "a", "dense", 1), hit(2, "b", "dense", 2)]]
    baseline = fuse(rankings, k=60, limit=2)
    candidate = fuse_with_parent_floor(rankings, k=60, item_limit=2, parent_limit=2)
    assert candidate == baseline

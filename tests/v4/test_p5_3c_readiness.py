"""P5.3C release gates keep provenance and payload fields explicit."""
import json

from scripts.v4.prepare_p5_3c_release_candidate import contains_key
from scripts.v4.verify_p5_3c_t4_reproducibility import proof_sessions


def test_proof_sessions_are_extracted_for_independence_gate():
    proofs = {
        53: json.dumps({"session_id": "run-a"}).encode(),
        130: json.dumps({"session_id": "run-b"}).encode(),
    }
    assert proof_sessions(proofs) == {"run-a", "run-b"}
    assert proof_sessions(proofs).isdisjoint({"run-c"})
    assert not proof_sessions(proofs).isdisjoint({"run-b"})


def test_recursive_rerank_score_detection():
    assert not contains_key({"id": 1, "relevant_docs": [2],
                             "relevant_chunks": [{"doc_id": 2, "chunk_text": "x"}]},
                            "rerank_score")
    assert contains_key({"relevant_chunks": [{"rerank_score": 0.5}]}, "rerank_score")

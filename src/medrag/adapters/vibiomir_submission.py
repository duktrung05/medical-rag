"""Serialize selected parents and exported evidence; internal document support stays private."""


def records(query_ids: list[int], results: dict[int, list[dict]]) -> list[dict]:
    output = []
    for qid in query_ids:
        docs = results.get(qid, [])
        output.append({"id": qid, "relevant_docs": [d["official_id"] for d in docs],
                       "relevant_chunks": [{"doc_id": d["official_id"], "chunk_text": e["text"]}
                                           for d in docs for e in d["evidence"]]})
    return output


"""HTTP contract and startup checks for the optional API."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.api import create_app


def test_search_api():
    with TestClient(create_app("configs/demo.yaml")) as client:
        assert client.get("/health").json() == {"status": "ok", "backend": "demo"}
        response = client.post("/search", json={"id": "q1", "query": "y khoa"})
        assert response.status_code == 200
        assert set(response.json()) == {"id", "relevant_docs", "relevant_chunks"}
        assert client.post("/search", json={"id": "q2", "query": "  "}).status_code == 422

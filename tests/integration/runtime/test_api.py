"""The default HTTP import must route to V4 and fail closed without a corpus."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from src.api import create_app


def test_default_api_requires_verified_v4_snapshot(monkeypatch):
    monkeypatch.delenv("MEDRAG_PREPARED", raising=False)
    with TestClient(create_app()) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/search", json={"id": 1, "query": "y khoa"}).status_code == 503

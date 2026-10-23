"""Official V4 HTTP entrypoint. It never loads the historical pipeline."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel

from src.medrag.adapters.vibiomir_submission import records
from src.medrag.core.config import load_config
from src.medrag.core.paths import project_root
from src.medrag.data.corpus_registry import CorpusRegistry
from src.medrag.data.prepared import load_prepared


class Query(BaseModel):
    id: int
    query: str


def create_app(config_path: str | None = None):
    from fastapi import FastAPI, HTTPException

    root = project_root()
    configured_path = config_path or os.environ.get("MEDRAG_CONFIG") or root / "configs/v4_core.yaml"
    configured_path = Path(configured_path)
    if not configured_path.is_absolute():
        configured_path = root / configured_path
    config = load_config(configured_path)
    registry = CorpusRegistry.load(root / config["registry"])
    version = registry.get(config["corpus_version"])
    app = FastAPI(title="Medical-RAG V4", version="4.0")
    prepared = os.environ.get("MEDRAG_PREPARED")
    app.state.pipeline = load_prepared(Path(prepared), registry, config=config) if prepared else None
    if app.state.pipeline and app.state.pipeline.corpus_version != version.name:
        raise ValueError("Prepared API corpus does not match active V4 configuration")

    @app.get("/health")
    def health():
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail={"status": "not_ready", "corpus_version": version.name,
                                                         "reason": "Set MEDRAG_PREPARED to a verified V4 snapshot"})
        return {"status": "ready", "corpus_version": version.name,
                "source_count": len(app.state.pipeline.sources)}

    @app.post("/search")
    def search(request: Query):
        if app.state.pipeline is None:
            raise HTTPException(status_code=503, detail="V4 prepared corpus is unavailable")
        if not request.query.strip():
            raise HTTPException(status_code=422, detail="query must not be blank")
        return records([request.id], {request.id: app.state.pipeline.retrieve(request.query, 10)})[0]

    return app


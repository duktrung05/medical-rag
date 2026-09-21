"""Optional HTTP interface: install the project's api extra before serving."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.config import load_config
from src.data.schema import PredictionRecord, QueryRecord
from src.service import build_pipeline


def create_app(config_path: str | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        config = load_config(config_path or os.environ.get("R2AI_CONFIG", "configs/demo.yaml"))
        app.state.pipeline = build_pipeline(config)
        app.state.backend = config.backend
        yield

    app = FastAPI(title="R2AI Medical Retrieval", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "backend": app.state.backend}

    @app.post("/search", response_model=PredictionRecord)
    def search(query: QueryRecord) -> PredictionRecord:
        return app.state.pipeline.run_query(query)

    return app

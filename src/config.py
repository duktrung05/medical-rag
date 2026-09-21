"""Validated runtime settings. Paths are relative to the configuration file."""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_name: str = "demo"
    backend: Literal["demo"] = "demo"
    corpus: Path
    max_chunks: int = Field(default=10, ge=1, le=100)


def load_config(path: str | Path) -> RuntimeConfig:
    path = Path(path).resolve()
    config = RuntimeConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if not config.corpus.is_absolute():
        config.corpus = (path.parent / config.corpus).resolve()
    return config

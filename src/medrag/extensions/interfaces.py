from typing import Protocol


class QueryExpansion(Protocol):
    def expand(self, query: str, language: str) -> list[str]: ...


class Taxonomy(Protocol):
    def concepts(self, text: str) -> list[dict]: ...


class QueryPlanner(Protocol):
    def plan(self, query: str) -> list[str]: ...


class LateInteraction(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...


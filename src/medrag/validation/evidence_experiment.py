"""Shared query membership and fail-closed, independently judged allocation gates."""
from __future__ import annotations

import hashlib
from typing import Any

PILOT_IDS = {53, 130, 380, 754, 771, 1144}
SPLIT_VERSION = "evidence-allocation-20261025-v1"


def historical_partition(query_id: int) -> str:
    """Preserve the existing R3/A0 train boundary exactly."""
    if query_id in PILOT_IDS:
        return "development"
    digest = hashlib.sha256(f"a0-quality-20261024:{query_id}".encode()).digest()
    return "validation" if int.from_bytes(digest[:4], "big") % 3 == 0 else "development"


def query_partition(query_id: int) -> str:
    if not 1 <= query_id <= 1200:
        raise ValueError("Query ID must be in 1..1200")
    if historical_partition(query_id) == "development":
        return "development"
    digest = hashlib.sha256(f"{SPLIT_VERSION}:{query_id}".encode()).digest()
    return "validation" if digest[0] % 2 == 0 else "held_out"


def validate_split(registry: dict[str, Any], training_ids: set[int]) -> bool:
    assignments = registry["assignments"]
    if (registry.get("version") != SPLIT_VERSION
            or set(assignments) != {str(qid) for qid in range(1, 1201)}
            or any(assignments[str(qid)] != query_partition(qid) for qid in range(1, 1201))):
        raise ValueError("Query split registry differs from the locked contract")
    if any(assignments.get(str(qid)) != "development" for qid in training_ids):
        raise ValueError("Training queries overlap evaluation or are outside the registry")
    return True


def paired_changes(baseline_hits: set[str], challenger_hits: set[str]) -> dict[str, Any]:
    gains, losses = challenger_hits - baseline_hits, baseline_hits - challenger_hits
    return {"gross_gains": len(gains), "gross_losses": len(losses),
            "net_change": len(gains) - len(losses),
            "query_gain": bool(gains), "query_loss": bool(losses),
            "gained_ids": sorted(gains), "lost_ids": sorted(losses)}


def allocation_gate(rows: list[dict[str, Any]], *, independent: bool,
                    structure_valid: bool, loss_budget: int) -> dict[str, Any]:
    if loss_budget < 0:
        raise ValueError("Loss budget must be nonnegative")
    gains = sum(row["gross_gains"] for row in rows)
    losses = sum(row["gross_losses"] for row in rows)
    loss_queries = sum(bool(row["query_loss"]) for row in rows)
    checks = {"nonempty_evaluation": bool(rows), "independent_evidence_judgments": independent,
              "structural_validators": structure_valid, "positive_net_relevance_gain": gains > losses,
              "gross_loss_queries_within_budget": loss_queries <= loss_budget}
    return {"decision": "GO" if all(checks.values()) else "NO_GO", "checks": checks,
            "gross_gains": gains, "gross_losses": losses, "net_change": gains - losses,
            "gross_loss_queries": loss_queries, "loss_budget_queries": loss_budget}

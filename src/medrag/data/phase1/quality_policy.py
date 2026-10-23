"""Versioned routing policy for Phase 1 quality signals."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .validation import QualityAssessment


VALID_STATUSES = frozenset({"PASS", "REVIEW_REQUIRED", "QUARANTINED"})


@dataclass(frozen=True)
class QualityDecision:
    status: str
    reason: str


@dataclass(frozen=True)
class QualityPolicy:
    version: str
    default_status: str
    quarantine_flags: frozenset[str]
    overrides: dict[int, QualityDecision]

    def decide(self, doc_id: int, assessment: QualityAssessment) -> QualityDecision:
        override = self.overrides.get(doc_id)
        if override is not None:
            return override
        if not assessment.flags:
            return QualityDecision("PASS", "No quality signals")
        blocking = sorted(set(assessment.flags).intersection(self.quarantine_flags))
        if blocking:
            return QualityDecision(
                "QUARANTINED", "Blocking quality signal(s): " + ", ".join(blocking)
            )
        return QualityDecision(
            self.default_status,
            "Unresolved quality signal(s): " + ", ".join(assessment.flags),
        )


def load_quality_policy(path: Path) -> QualityPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("version")
    default_status = payload.get("default_status")
    if not isinstance(version, str) or not version:
        raise ValueError("Quality policy needs a non-empty version")
    if default_status not in VALID_STATUSES:
        raise ValueError("Quality policy default_status is invalid")
    quarantine_flags = payload.get("quarantine_flags")
    if not isinstance(quarantine_flags, list) or not all(isinstance(flag, str) for flag in quarantine_flags):
        raise ValueError("Quality policy quarantine_flags must be a string list")
    overrides: dict[int, QualityDecision] = {}
    for item in payload.get("overrides", []):
        try:
            doc_id = int(item["doc_id"])
            status = item["status"]
            reason = item["reason"]
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid quality-policy override") from exc
        if status not in VALID_STATUSES or not isinstance(reason, str) or not reason:
            raise ValueError(f"Invalid quality-policy override for {doc_id}")
        if doc_id in overrides:
            raise ValueError(f"Duplicate quality-policy override for {doc_id}")
        overrides[doc_id] = QualityDecision(status, reason)
    return QualityPolicy(
        version=version,
        default_status=default_status,
        quarantine_flags=frozenset(quarantine_flags),
        overrides=overrides,
    )

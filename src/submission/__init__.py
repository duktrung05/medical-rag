"""Submission generation and validation utilities."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.submission.build import build_submission_file, enforce_parent_consistency
    from src.submission.validate import (
        ValidationReport,
        validate_submission_file,
        validate_submission_records,
    )

__all__ = [
    "build_submission_file",
    "enforce_parent_consistency",
    "ValidationReport",
    "validate_submission_file",
    "validate_submission_records",
]


def __getattr__(name: str):
    if name in ("build_submission_file", "enforce_parent_consistency"):
        import src.submission.build as b
        return getattr(b, name)
    if name in ("ValidationReport", "validate_submission_file", "validate_submission_records"):
        import src.submission.validate as v
        return getattr(v, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

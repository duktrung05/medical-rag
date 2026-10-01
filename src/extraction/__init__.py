"""Offline extraction of downloaded documents."""

from .html_extractor import DecodeError, ExtractedHtml, extract_html

__all__ = ["DecodeError", "ExtractedHtml", "extract_html"]

"""Non-destructive document-quality signals for Phase 1."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


QUALITY_VALIDATION_VERSION = "phase1-quality-rules-v3"
REPEATED_TEMPLATE_THRESHOLD = 20
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HTML = re.compile(
    r"<!--|</?(?:html|head|body|script|style|iframe|form|svg|nav|footer|header|"
    r"div|span|p|br|li|ul|ol|table|meta|link|a)\s*(?:/?>|\s+[A-Za-z_:][\w:.-]*\s*=)|"
    r"(?:&lt;|&#60;)/?(?:script|style|iframe)\b",
    re.IGNORECASE,
)
_CHALLENGE = re.compile(
    r"(?:just a moment|checking your browser|verify you are human|"
    r"enable javascript and cookies|cloudflare ray id|cf-challenge|captcha|"
    r"xác minh bạn là con người|truy cập bị từ chối|访问被拒绝|安全验证)",
    re.IGNORECASE,
)
_BOILERPLATE = re.compile(
    r"(?:accept all cookies|cookie policy|privacy policy|terms of use|"
    r"đăng ký nhận tin|theo dõi chúng tôi|all rights reserved|"
    r"版权所有|免责声明)",
    re.IGNORECASE,
)
_HEX_NOISE = re.compile(r"^(?:[0-9a-f]{24,}\s*){2,}", re.IGNORECASE)
# A standalone Vietnamese initial such as "H.Â" is valid text.  Only flag the
# trailing-Â pattern when it is attached to a word, as in typical NBSP mojibake.
_MOJIBAKE = re.compile(r"(?:ï¿½|â€[™œž“”˜¦]|Ã[\x80-\xbf]|(?<=\w)Â(?=[\s\xa0]))")
_NAVIGATION = re.compile(
    r"(?:\b(?:home page|main menu|login|register|subscribe|privacy policy|terms of use)\b|"
    r"(?:trang chủ|đăng nhập|đăng ký|chính sách bảo mật|điều khoản sử dụng)|"
    r"(?:首页|登录|注册|隐私政策|网站导航))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class QualityAssessment:
    flags: tuple[str, ...]
    reasons: tuple[str, ...]
    review_required: bool


def assess_text_quality(
    text: str,
    *,
    title: str = "",
    source_kind: str,
    encoding: str | None = None,
    repeated_text_count: int = 1,
) -> QualityAssessment:
    """Return review signals only; this function never drops a document."""
    findings: list[tuple[str, str]] = []

    def mark(flag: str, reason: str) -> None:
        findings.append((flag, reason))

    stripped = text.strip()
    if not stripped:
        mark("empty_text", "Source text is empty after trimming")
    if len(stripped) < 80:
        mark("very_short_text", "Text has fewer than 80 characters; retained for review")
    if not title.strip():
        mark("missing_title", "Source title is empty")
    if _CONTROL.search(text):
        mark("control_characters", "Text contains non-whitespace ASCII control characters")
    if "\ufffd" in text:
        mark("unicode_replacement_character", "Text contains Unicode replacement characters")
    if _MOJIBAKE.search(text):
        mark("mojibake_suspected", "Text contains common UTF-8 decoding artifacts")
    if unicodedata.normalize("NFC", text) != text:
        mark("source_not_nfc", "Source text is not Unicode NFC")
    if _HTML.search(text):
        mark("html_contamination", "Text contains HTML/script/style markup")
    if _CHALLENGE.search((title + "\n" + text)[:5000]):
        mark("challenge_text", "Text contains access challenge or verification language")
    if len(_BOILERPLATE.findall(text[:20_000])) >= 2:
        mark("boilerplate_signals", "Text contains repeated navigation/legal boilerplate signals")
    navigation_signals = {match.group(0).casefold() for match in _NAVIGATION.finditer(text[:10_000])}
    if len(navigation_signals) >= 3:
        mark("navigation_heavy", "Text contains at least three distinct navigation/account-menu signals")
    if _HEX_NOISE.search(stripped[:1000]):
        mark("hex_or_asset_noise", "Text begins with repeated long hexadecimal tokens")
    if source_kind == "html" and not (encoding or "").strip():
        mark("encoding_metadata_missing", "HTML source has no recorded decoding metadata")
    if repeated_text_count >= REPEATED_TEMPLATE_THRESHOLD:
        mark(
            "repeated_template_content",
            f"Identical source text hash occurs in {repeated_text_count} documents",
        )
    flags = tuple(flag for flag, _ in findings)
    reasons = tuple(reason for _, reason in findings)
    return QualityAssessment(flags, reasons, bool(findings))

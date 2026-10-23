from __future__ import annotations

import re

from bs4 import BeautifulSoup


def extract_html(raw: bytes, encoding: str | None = None) -> tuple[str, str]:
    declared = re.search(rb"<meta[^>]+charset\s*=\s*['\"]?([\w.-]+)", raw[:65536], re.IGNORECASE)
    candidates = [encoding] if encoding else []
    if declared:
        candidates.append(declared.group(1).decode("ascii", errors="ignore"))
    candidates.extend(["utf-8", "gb18030"])
    decoded = None
    for candidate in dict.fromkeys(c for c in candidates if c):
        try:
            decoded = raw.decode(candidate, errors="strict")
            break
        except (LookupError, UnicodeDecodeError):
            continue
    if decoded is None:
        raise UnicodeError("HTML encoding could not be decoded losslessly")
    soup = BeautifulSoup(decoded, "html.parser")
    heading = soup.find("h1")
    title_tag = soup.find("title")
    title = (heading or title_tag).get_text(" ", strip=True) if heading or title_tag else ""
    for tag in list(soup.find_all(["script", "style", "nav", "footer", "aside", "header", "form", "noscript", "svg"])):
        tag.decompose()
    bad = re.compile(r"(^|[-_])(nav|menu|footer|sidebar|advert|social|share|related|recommend|comment|breadcrumb|popup|cookie)([-_]|$)", re.I)
    for tag in list(soup.find_all(True)):
        if tag.parent and tag.name not in {"html", "body"} and bad.search(" ".join([str(tag.get("id") or ""), *tag.get("class", [])])):
            tag.decompose()
    candidates = []
    for tag in soup.find_all(["article", "main", "div", "section"]):
        length = len(tag.get_text(" ", strip=True))
        if length < 100:
            continue
        label = " ".join([str(tag.get("id") or ""), *tag.get("class", [])]).lower()
        bonus = 0
        if tag.name == "article":
            bonus += 900
        if "article-detail" in label or "entry-content" in label:
            bonus += 800
        elif "content" in label or "detail" in label:
            bonus += 500
        if tag.name == "main":
            bonus += 150
        links = sum(len(a.get_text(" ", strip=True)) for a in tag.find_all("a"))
        density = links / max(1, length)
        score = bonus + min(length, 8000) * .12 - density * 700 - max(length - 8000, 0) * .15
        candidates.append((score, tag))
    root = max(candidates, key=lambda x: x[0])[1] if candidates else soup.body or soup
    blocks = []
    for tag in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "td"]):
        if tag.find(["h1", "h2", "h3", "h4", "p", "li", "blockquote", "td"]):
            continue
        value = re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
        if value:
            blocks.append(value)
    if not blocks:
        value = re.sub(r"\s+", " ", root.get_text(" ", strip=True)).strip()
        if value:
            blocks.append(value)
    return title, "\n\n".join(blocks)


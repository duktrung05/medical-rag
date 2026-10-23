from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DerivedField:
    value: str
    method: str
    confidence: float | None
    verified: bool = False


def language_hint(text: str) -> DerivedField:
    han = sum("\u4e00" <= char <= "\u9fff" for char in text)
    vi = sum(char in "ăâđêôơưĂÂĐÊÔƠƯàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ" for char in text)
    latin = sum(char.isascii() and char.isalpha() for char in text)
    if han >= 5 and han > latin / 4:
        return DerivedField("zh", "script_heuristic_v1", .85)
    if vi >= 2:
        return DerivedField("vi", "diacritic_heuristic_v1", .8)
    if latin >= 20:
        return DerivedField("en", "ascii_heuristic_v1", .45)
    return DerivedField("unknown", "script_heuristic_v1", None)


"""Pure chunking tests using an offline deterministic token counter."""

from src.chunking import ChunkingConfig, chunk_text, detect_language


class CharacterCounter:
    def count(self, text: str) -> int:
        return len(text)


CFG = ChunkingConfig(target_tokens=50, max_tokens=70, overlap_tokens=20,
                     min_chunk_tokens=12)


def _check(text, spans, maximum=70):
    covered = bytearray(len(text))
    seen = set()
    for span in spans:
        value = text[span.char_start:span.char_end]
        assert value and value not in seen
        assert span.token_count == len(value) <= maximum
        seen.add(value)
        covered[span.char_start:span.char_end] = b"\x01" * len(value)
    assert all(covered[i] for i, character in enumerate(text) if not character.isspace())


def test_short_document_one_chunk_and_empty_document():
    text = "Tiếng Việt có dấu."
    spans = chunk_text(text, CharacterCounter(), CFG)
    assert len(spans) == 1
    assert text[spans[0].char_start:spans[0].char_end] == text
    assert chunk_text("  \n\n ", CharacterCounter(), CFG) == []


def test_short_paragraphs_are_combined():
    text = "Một đoạn ngắn.\n\nĐoạn hai ngắn.\n\nĐoạn ba."
    spans = chunk_text(text, CharacterCounter(), CFG)
    assert len(spans) == 1
    assert "\n\n" in text[spans[0].char_start:spans[0].char_end]
    _check(text, spans)


def test_long_paragraph_split_on_sentences_and_overlap():
    text = "Câu đầu nói về bệnh. Câu hai nói về thuốc. Câu ba nói về chăm sóc. "
    text += "Câu bốn nói về điều trị."
    config = ChunkingConfig(target_tokens=50, max_tokens=70, overlap_tokens=25,
                            min_chunk_tokens=12)
    spans = chunk_text(text, CharacterCounter(), config)
    assert len(spans) >= 2
    assert spans[1].char_start < spans[0].char_end
    assert text[spans[0].char_end - 1] in ". "
    _check(text, spans)


def test_very_long_sentence_split_safely():
    text = "Một câu rất dài " + "chăm sóc sức khỏe " * 12 + "kết thúc."
    spans = chunk_text(text, CharacterCounter(), CFG)
    assert len(spans) > 2
    _check(text, spans)
    assert spans == chunk_text(text, CharacterCounter(), CFG)


def test_chinese_and_language_heuristic():
    text = "医学研究表明需要详细诊断。治疗方案应该因人而异。患者应当咨询专业医生。" * 3
    spans = chunk_text(text, CharacterCounter(), CFG)
    assert detect_language(text) == "zh"
    assert detect_language("Sức khỏe và điều trị.") == "vi"
    assert detect_language("Clinical medicine") == "unknown"
    _check(text, spans)


def test_short_tail_merges_when_within_maximum():
    text = "Đoạn đầu tương đối dài cho thử nghiệm.\n\nĐoạn cuối."
    config = ChunkingConfig(target_tokens=35, max_tokens=70, overlap_tokens=0,
                            min_chunk_tokens=15)
    spans = chunk_text(text, CharacterCounter(), config)
    assert len(spans) == 1
    _check(text, spans)

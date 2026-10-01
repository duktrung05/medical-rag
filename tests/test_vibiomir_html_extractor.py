"""Local HTML fixtures; extraction must never request a URL."""

from src.extraction.html_extractor import extract_html


def test_utf8_vietnamese_title_cleanup_and_order():
    raw = """<html><head><meta property='og:title' content='Bài viết y khoa'></head>
    <body><nav>Menu điều hướng</nav><article><h1>Tiêu đề phụ</h1>
    <p>Đoạn đầu về sức khỏe Việt Nam.</p><p>Đoạn thứ hai nói về điều trị.</p>
    </article><footer>Chân trang</footer><script>var bad = 1;</script>
    <style>.bad{color:red}</style></body></html>""".encode()
    result = extract_html(raw, "text/html; charset=UTF-8")
    assert result.title == "Bài viết y khoa"
    assert result.encoding == "utf-8"
    assert result.encoding_source == "http_content_type"
    assert result.paragraph_count == 2
    assert result.text.index("Đoạn đầu") < result.text.index("Đoạn thứ hai")
    assert "Menu" not in result.text and "Chân trang" not in result.text
    assert "var bad" not in result.text and "color:red" not in result.text


def test_chinese_meta_charset():
    html = "<meta charset='gb2312'><article><p>中医治疗需要详细诊断。</p></article>"
    result = extract_html(html.encode("gb2312"))
    assert "中医治疗" in result.text
    assert result.encoding_source == "html_meta"


def test_http_charset_precedes_meta():
    html = "<meta charset='gb2312'><article><p>Tiếng Việt cần giữ nguyên dấu và nội dung.</p></article>"
    result = extract_html(html.encode(), "text/html; charset=utf-8")
    assert result.encoding_source == "http_content_type"
    assert "Tiếng Việt" in result.text


def test_fallback_container_and_empty_content():
    result = extract_html(b"<div class='answercont'><p>First medical explanation is here.</p>"
                          b"<p>Second medical explanation is here.</p></div>")
    assert result.text.startswith("First")
    assert "Second" in result.text
    assert result.extraction_method.startswith("div:")
    empty = extract_html(b"<html><body><nav>Menu</nav><script>bad()</script></body></html>")
    assert empty.text == ""
    assert empty.paragraph_count == 0

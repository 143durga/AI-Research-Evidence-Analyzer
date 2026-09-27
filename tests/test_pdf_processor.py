import pytest
import pymupdf

from research_analyzer.pdf_processor import PageText, chunk_pages, extract_pdf_pages


def make_pdf(*page_texts: str) -> bytes:
    document = pymupdf.open()
    for text in page_texts:
        page = document.new_page()
        page.insert_text((72, 72), text)
    pdf_bytes = document.tobytes()
    document.close()
    return pdf_bytes


def test_extract_pdf_pages_preserves_page_numbers() -> None:
    pages = extract_pdf_pages(make_pdf("First page", "Second page"))

    assert [(page.page_number, page.text) for page in pages] == [
        (1, "First page"),
        (2, "Second page"),
    ]


def test_extract_pdf_pages_rejects_empty_or_scanned_pdf() -> None:
    with pytest.raises(ValueError, match="empty"):
        extract_pdf_pages(b"")
    with pytest.raises(ValueError, match="No selectable text"):
        extract_pdf_pages(make_pdf(""))


def test_chunk_pages_overlaps_and_keeps_page_metadata() -> None:
    pages = [PageText(page_number=3, text="one two three four five six")]

    chunks = chunk_pages(pages, chunk_words=4, overlap_words=2)

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "three four five six",
        "five six",
    ]
    assert all(chunk.page_number == 3 for chunk in chunks)


def test_chunk_pages_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="larger than the overlap"):
        chunk_pages([], chunk_words=4, overlap_words=4)
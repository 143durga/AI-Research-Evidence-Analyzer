"""Extract page text from PDFs and split it into overlapping word chunks."""

from dataclasses import dataclass

import pymupdf


@dataclass(frozen=True)
class PageText:
    page_number: int
    text: str


@dataclass(frozen=True)
class TextChunk:
    page_number: int
    chunk_index: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes) -> list[PageText]:
    """Return non-empty page text; scanned PDFs need OCR and are not supported."""
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            pages = [
                PageText(page_number=index + 1, text=page.get_text("text").strip())
                for index, page in enumerate(document)
            ]
    except Exception as exc:
        raise ValueError("This file could not be read as a PDF.") from exc

    pages = [page for page in pages if page.text]
    if not pages:
        raise ValueError(
            "No selectable text was found. Scanned PDFs need OCR before upload."
        )
    return pages


def chunk_pages(
    pages: list[PageText], chunk_words: int = 220, overlap_words: int = 40
) -> list[TextChunk]:
    """Split each page independently so every retrieved chunk keeps its page."""
    if chunk_words < 1 or overlap_words < 0 or overlap_words >= chunk_words:
        raise ValueError("Chunk size must be positive and larger than the overlap.")

    chunks: list[TextChunk] = []
    step = chunk_words - overlap_words
    for page in pages:
        words = page.text.split()
        for chunk_index, start in enumerate(range(0, len(words), step)):
            chunk_text = " ".join(words[start : start + chunk_words])
            if chunk_text:
                chunks.append(
                    TextChunk(
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        text=chunk_text,
                    )
                )
    return chunks
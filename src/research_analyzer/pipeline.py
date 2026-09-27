"""Orchestrate PDF indexing, evidence retrieval, and grounded answering."""

from dataclasses import dataclass

from research_analyzer.answering import AnswerGenerator
from research_analyzer.pdf_processor import chunk_pages, extract_pdf_pages
from research_analyzer.vector_store import Evidence, PaperVectorStore


@dataclass(frozen=True)
class IndexSummary:
    document_id: str
    page_count: int
    chunk_count: int


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    evidence: list[Evidence]
    retrieved_count: int


class ResearchAnalyzer:
    def __init__(self, vector_store: PaperVectorStore) -> None:
        self.vector_store = vector_store

    def index_pdf(
        self,
        filename: str,
        pdf_bytes: bytes,
        chunk_words: int = 220,
        overlap_words: int = 40,
    ) -> IndexSummary:
        pages = extract_pdf_pages(pdf_bytes)
        chunks = chunk_pages(pages, chunk_words, overlap_words)
        document_id = self.vector_store.document_id(pdf_bytes)
        self.vector_store.index_chunks(document_id, filename, chunks)
        return IndexSummary(document_id, len(pages), len(chunks))

    def answer_question(
        self,
        document_id: str,
        question: str,
        generator: AnswerGenerator,
        top_k: int = 5,
        max_distance: float = 0.75,
    ) -> AnswerResult:
        if not question.strip():
            raise ValueError("Enter a research question first.")

        retrieved = self.vector_store.search(document_id, question, top_k=top_k)
        evidence = [item for item in retrieved if item.distance <= max_distance]
        answer = generator.generate(question, evidence)
        return AnswerResult(answer, evidence, len(retrieved))
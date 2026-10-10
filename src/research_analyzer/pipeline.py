"""Orchestrate PDF indexing, evidence retrieval, and grounded answering."""

from dataclasses import dataclass
import re

from research_analyzer.answering import AnswerGenerator
from research_analyzer.definitions import requested_acronym
from research_analyzer.pdf_processor import chunk_pages, extract_pdf_pages
from research_analyzer.vector_store import Evidence, PaperVectorStore

_STOP_WORDS = {
    "a",
    "about",
    "an",
    "and",
    "are",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "their",
    "them",
    "there",
    "these",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
    "with",
}
_RESULT_INTENT = {
    "accuracy",
    "best",
    "effective",
    "effectiveness",
    "finding",
    "highest",
    "improve",
    "improvement",
    "metric",
    "performance",
    "result",
    "score",
    "significant",
}
_RESULTS_SECTION = re.compile(
    r"(?:\b(?:[IVX]+|\d+(?:\.\d+)*)\.\s*(?:results?|findings?)\s+"
    r"[A-Z]\.\s*(?:RQ|R)\s*\d+\b)|"
    r"(?:^\s*(?:RQ|R)\s*\d+\s+(?:results?|findings?)\b)|"
    r"(?:^\s*(?:results?|findings?)\b)",
    re.IGNORECASE,
)


def _terms(text: str) -> set[str]:
    return {
        term.lower().replace(",", "")
        for term in re.findall(r"[A-Za-z]+[A-Za-z0-9]*|[0-9]+(?:[.,][0-9]+)*%?", text)
        if term.lower() not in _STOP_WORDS
    }


def rank_evidence(question: str, evidence: list[Evidence]) -> list[Evidence]:
    """Prefer query-specific passages and direct results over discussion."""
    question_terms = _terms(question)
    result_question = bool(question_terms & _RESULT_INTENT)
    definition_question = requested_acronym(question) is not None

    def score(item: Evidence) -> tuple[float, float]:
        passage_terms = _terms(item.text)
        overlap = len(question_terms & passage_terms)
        score_value = overlap * 2 + overlap / max(len(question_terms), 1)
        if result_question and _RESULTS_SECTION.search(item.text):
            score_value += 2.5
        if definition_question and item.direct_match:
            score_value += 1000
        return score_value, -(item.distance if item.distance is not None else 0)

    return sorted(evidence, key=score, reverse=True)


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
    eligible_count: int


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
        if top_k < 1:
            raise ValueError("Evidence passage count must be at least one.")

        retrieved = self.vector_store.search(
            document_id, question, top_k=min(top_k * 3, 60)
        )
        eligible = [
            item
            for item in retrieved
            if item.direct_match
            or (item.distance is not None and item.distance <= max_distance)
        ]
        evidence = rank_evidence(question, eligible)[:top_k]
        answer = generator.generate(question, evidence)
        return AnswerResult(answer, evidence, len(retrieved), len(eligible))
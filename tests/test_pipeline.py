import pytest

from research_analyzer.answering import INSUFFICIENT_EVIDENCE
from research_analyzer.pipeline import ResearchAnalyzer
from research_analyzer.vector_store import Evidence


class FakeVectorStore:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence

    def search(self, document_id: str, question: str, top_k: int) -> list[Evidence]:
        return self.evidence[:top_k]


class FakeGenerator:
    def __init__(self) -> None:
        self.received: list[Evidence] | None = None

    def generate(self, question: str, evidence: list[Evidence]) -> str:
        self.received = evidence
        return INSUFFICIENT_EVIDENCE if not evidence else "Supported answer [E1]."


def test_only_evidence_within_distance_threshold_reaches_generator() -> None:
    close = Evidence("close", "Relevant passage", "paper.pdf", 2, 0.2)
    distant = Evidence("distant", "Unrelated passage", "paper.pdf", 4, 1.1)
    generator = FakeGenerator()
    analyzer = ResearchAnalyzer(FakeVectorStore([close, distant]))

    result = analyzer.answer_question("doc", "What happened?", generator, max_distance=0.75)

    assert result.answer == "Supported answer [E1]."
    assert result.retrieved_count == 2
    assert result.evidence == [close]
    assert generator.received == [close]


def test_no_passing_evidence_returns_abstention() -> None:
    distant = Evidence("distant", "Unrelated passage", "paper.pdf", 4, 1.1)
    generator = FakeGenerator()
    analyzer = ResearchAnalyzer(FakeVectorStore([distant]))

    result = analyzer.answer_question("doc", "What happened?", generator, max_distance=0.75)

    assert result.answer == INSUFFICIENT_EVIDENCE
    assert result.evidence == []
    assert generator.received == []


def test_empty_question_is_rejected() -> None:
    analyzer = ResearchAnalyzer(FakeVectorStore([]))

    with pytest.raises(ValueError, match="research question"):
        analyzer.answer_question("doc", "  ", FakeGenerator())
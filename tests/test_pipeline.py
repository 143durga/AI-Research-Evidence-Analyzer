import pytest

from research_analyzer.answering import INSUFFICIENT_EVIDENCE
from research_analyzer.pipeline import ResearchAnalyzer, rank_evidence
from research_analyzer.vector_store import Evidence, PaperVectorStore


def test_vector_store_disables_chroma_telemetry(monkeypatch, tmp_path) -> None:
    import research_analyzer.vector_store as vector_store

    captured = {}
    monkeypatch.setattr(
        vector_store.chromadb,
        "PersistentClient",
        lambda **kwargs: captured.update(kwargs),
    )

    PaperVectorStore(tmp_path / "chroma")

    assert captured["settings"].anonymized_telemetry is False


class FakeVectorStore:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence = evidence
        self.searched_document_ids: list[str] = []
        self.requested_top_k: list[int] = []

    def search(self, document_id: str, question: str, top_k: int) -> list[Evidence]:
        self.searched_document_ids.append(document_id)
        self.requested_top_k.append(top_k)
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
    assert result.eligible_count == 1
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


def test_pipeline_searches_only_the_selected_document() -> None:
    paper_a = Evidence("a", "Paper A evidence", "a.pdf", 1, 0.1)
    paper_b = Evidence("b", "Paper B evidence", "b.pdf", 1, 0.1)

    class ScopedVectorStore(FakeVectorStore):
        def search(self, document_id: str, question: str, top_k: int) -> list[Evidence]:
            self.searched_document_ids.append(document_id)
            return {"paper-a": [paper_a], "paper-b": [paper_b]}[document_id][:top_k]

    store = ScopedVectorStore([paper_a, paper_b])
    analyzer = ResearchAnalyzer(store)
    generator = FakeGenerator()

    result = analyzer.answer_question("paper-a", "What is reported?", generator)

    assert store.searched_document_ids == ["paper-a"]
    assert result.evidence == [paper_a]


def test_direct_case_count_and_template_passage_ranks_first() -> None:
    case_count = Evidence(
        "count",
        "We generated 4,700 test cases using 12 templates.",
        "paper.pdf",
        2,
        0.3,
    )
    general = Evidence(
        "general", "The experiment uses a set of test inputs.", "paper.pdf", 3, 0.1
    )

    ranked = rank_evidence("How many test cases and templates were generated?", [
        general,
        case_count,
    ])

    assert ranked[0] == case_count


def test_pipeline_searches_extra_candidates_to_recover_later_relevant_passage() -> None:
    evidence = [
        Evidence(str(index), f"Unrelated passage {index}", "paper.pdf", index, 0.2)
        for index in range(5)
    ]
    evidence.append(
        Evidence("later", "MR4 was effective for sentiment.", "paper.pdf", 7, 0.3)
    )
    store = FakeVectorStore(evidence)
    analyzer = ResearchAnalyzer(store)
    generator = FakeGenerator()

    result = analyzer.answer_question(
        "doc", "Which MR was effective for sentiment?", generator, top_k=2
    )

    assert store.requested_top_k == [6]
    assert result.evidence[0].chunk_id == "later"
    assert generator.received == result.evidence


def test_model_specific_results_outrank_general_discussion() -> None:
    discussion = Evidence(
        "discussion",
        "Discussion. Results across multiple models suggest the preferred "
        "relations varied.",
        "paper.pdf",
        9,
        0.05,
    )
    result = Evidence(
        "results",
        "Results. For LLaMA 3.0, MR4 was most effective for the sentiment "
        "metric, and MR12 for the tone metric.",
        "paper.pdf",
        7,
        0.2,
    )

    ranked = rank_evidence(
        "In the Results, for LLaMA 3.0 which relation was most effective for "
        "sentiment and tone metrics?",
        [discussion, result],
    )

    assert ranked[0] == result
    assert ranked[0].page_number == 7


def test_explicit_rq_results_mapping_outranks_unrelated_page_five_passage() -> None:
    page_five = Evidence(
        "p5",
        "LLaMA 3.0 was evaluated using sentiment and tone analyses.",
        "paper.pdf",
        5,
        0.2,
    )
    page_seven = Evidence(
        "p7",
        "RQ1 Results. For LLaMA 3.0, MR4 was most effective for the sentiment "
        "metric, and MR12 for the tone metric.",
        "paper.pdf",
        7,
        0.511,
    )
    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )

    assert rank_evidence(question, [page_five, page_seven])[0] == page_seven


def test_results_heading_inside_chunk_receives_ranking_priority() -> None:
    discussion = Evidence(
        "discussion",
        "Discussion. Results across models describe several possible findings.",
        "paper.pdf",
        9,
        0.1,
    )
    result = Evidence(
        "result",
        "Experimental setup details. VII. RESULT A. RQ1: Which MRs perform "
        "better in identifying fairness faults in LLaMA 3?",
        "paper.pdf",
        6,
        0.5,
    )

    ranked = rank_evidence(
        "According to RQ1 results, which relation was most effective?",
        [discussion, result],
    )

    assert ranked[0] == result


def test_metamorphic_relation_definition_matches_the_acronym_question() -> None:
    definition = Evidence(
        "definition",
        "MR denotes Metamorphic Relation throughout this paper.",
        "paper.pdf",
        1,
        0.4,
    )
    unrelated = Evidence("other", "Model rules are configurable.", "paper.pdf", 2, 0.1)

    ranked = rank_evidence("What does MR mean in this paper?", [unrelated, definition])

    assert ranked[0] == definition


def test_direct_definition_match_bypasses_vector_distance_threshold() -> None:
    direct = Evidence(
        "definition",
        "Fairness-oriented metamorphic relations (MRs) are used in the paper.",
        "paper.pdf",
        1,
        None,
        direct_match=True,
    )
    store = FakeVectorStore([direct])
    generator = FakeGenerator()
    analyzer = ResearchAnalyzer(store)

    result = analyzer.answer_question(
        "doc", "What does MR mean in this paper?", generator, max_distance=0.75
    )

    assert result.evidence == [direct]
    assert result.eligible_count == 1
    assert generator.received == [direct]


def test_search_for_unindexed_document_returns_no_evidence(tmp_path) -> None:
    store = PaperVectorStore(tmp_path / "chroma")

    assert store.search("missing-document", "What does the paper say?") == []


def test_abbreviation_search_adds_a_direct_definition_candidate() -> None:
    class FakeCollection:
        def count(self):
            return 2

        def query(self, query_texts, n_results, include):
            return {
                "ids": [["nearest"]],
                "documents": [["The model response is analyzed."]],
                "metadatas": [[{"filename": "paper.pdf", "page_number": 5}]],
                "distances": [[0.2]],
            }

        def get(self, include):
            assert include == ["documents", "metadatas"]
            return {
                "ids": ["nearest", "definition"],
                "documents": [
                    "The model response is analyzed.",
                    "Fairness-oriented metamorphic relations (MRs) are used.",
                ],
                "metadatas": [
                    {"filename": "paper.pdf", "page_number": 5},
                    {"filename": "paper.pdf", "page_number": 1},
                ],
            }

    class FakeClient:
        def list_collections(self):
            return [type("CollectionRef", (), {"name": "paper_doc"})()]

        def get_collection(self, name):
            assert name == "paper_doc"
            return FakeCollection()

    store = PaperVectorStore.__new__(PaperVectorStore)
    store.client = FakeClient()

    evidence = store.search("doc", "What does MR mean in this paper?", top_k=1)

    assert [item.chunk_id for item in evidence] == ["nearest", "definition"]
    assert evidence[1].page_number == 1
    assert evidence[1].distance is None
    assert evidence[1].direct_match


def test_reindexing_removes_stale_chunks_for_the_same_document() -> None:
    class FakeCollection:
        def __init__(self):
            self.ids = ["doc_p0001_c0000", "doc_p0001_c0001"]
            self.deleted = []
            self.upserted = []

        def get(self, include):
            assert include == []
            return {"ids": self.ids}

        def delete(self, ids):
            self.deleted.extend(ids)
            self.ids = [chunk_id for chunk_id in self.ids if chunk_id not in ids]

        def upsert(self, ids, documents, metadatas):
            self.upserted.extend(ids)

    class FakeClient:
        def __init__(self, collection):
            self.collection = collection

        def get_or_create_collection(self, name, metadata):
            assert name == "paper_doc"
            return self.collection

    collection = FakeCollection()
    store = PaperVectorStore.__new__(PaperVectorStore)
    store.client = FakeClient(collection)

    from research_analyzer.pdf_processor import TextChunk

    store.index_chunks("doc", "paper.pdf", [TextChunk(1, 0, "current chunk")])

    assert collection.deleted == ["doc_p0001_c0001"]
    assert collection.upserted == ["doc_p0001_c0000"]
from pathlib import Path

from research_analyzer.answering import (
    SYSTEM_PROMPT,
    AnswerGenerator,
    INSUFFICIENT_EVIDENCE,
    _validate_answer,
)
from research_analyzer.vector_store import Evidence


def test_answer_generator_abstains_without_calling_an_api() -> None:
    generator = AnswerGenerator(
        model="unused",
        base_url="https://example.invalid/v1",
        api_key=None,
    )

    answer = generator.generate("What happened?", [])

    assert answer == INSUFFICIENT_EVIDENCE


def test_missing_local_model_file_reports_actionable_error(tmp_path) -> None:
    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
        local_model_path=tmp_path / "missing.gguf",
    )

    try:
        generator.generate(
            "What does it say?",
            [Evidence("e1", "MR4 improved sentiment.", "paper.pdf", 7, 0.1)],
        )
    except FileNotFoundError as error:
        assert "Local model file not found" in str(error)
    else:
        raise AssertionError("A missing local model must not be reported as success.")


def test_default_backend_uses_the_existing_local_qwen_model(monkeypatch) -> None:
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    monkeypatch.delenv("LOCAL_MODEL_PATH", raising=False)

    generator = AnswerGenerator(
        model="Qwen2.5-1.5B-Instruct-Q4_K_M",
        base_url="http://localhost:8000/v1",
        api_key=None,
    )

    assert generator.backend == "llama_cpp"
    assert generator.local_model_path == Path(
        "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    )


def test_hosted_backends_are_rejected_before_any_inference() -> None:
    import pytest

    with pytest.raises(ValueError, match="Only the local llama_cpp backend"):
        AnswerGenerator(
            model="unused",
            base_url="https://api.example.invalid/v1",
            api_key="not-used",
            backend="openai_compatible",
        )


def test_prompt_requires_model_metric_result_specific_claims() -> None:
    assert "Do not infer missing values" in SYSTEM_PROMPT
    assert "combine results across models" in SYSTEM_PROMPT
    assert "cite every" in SYSTEM_PROMPT


def test_answer_validation_rejects_invalid_citations_and_uncited_claims() -> None:
    evidence = [Evidence("one", "MR4 improved sentiment.", "paper.pdf", 7, 0.1)]

    assert not _validate_answer("MR4 improved sentiment [E2].", evidence)
    assert not _validate_answer("MR4 improved sentiment [E1] [Ebad].", evidence)
    assert not _validate_answer("MR4 improved sentiment.", evidence)
    assert _validate_answer("MR4 improved sentiment [E1].", evidence)


def test_answer_validation_rejects_unrelated_claim_with_a_valid_citation() -> None:
    evidence = [
        Evidence("one", "MR4 improved sentiment.", "paper.pdf", 7, 0.1)
    ]

    assert not _validate_answer("Blue bicycles are reported [E1].", evidence)


def test_answer_validation_rejects_unsubstantiated_accuracy_percentages() -> None:
    evidence = [
        Evidence(
            "one",
            "The study compared twelve metamorphic relations.",
            "paper.pdf",
            7,
            0.1,
        )
    ]

    assert not _validate_answer("MR1 achieved 91% accuracy [E1].", evidence)
    assert _validate_answer(INSUFFICIENT_EVIDENCE, evidence)


def test_answer_validation_accepts_source_supported_numbers() -> None:
    evidence = [
        Evidence(
            "one",
            "We generated 4,700 test cases using 12 templates.",
            "paper.pdf",
            2,
            0.1,
        )
    ]

    assert _validate_answer(
        "The paper reports 4,700 test cases from 12 templates [E1].", evidence
    )


def test_answer_validation_accepts_citation_after_sentence_punctuation() -> None:
    evidence = [Evidence("one", "MR means Metamorphic Relation.", "paper.pdf", 7, 0.1)]

    assert _validate_answer("MR stands for Metamorphic Relation. [E1]", evidence)


def test_rq1_answer_with_decimal_model_version_is_not_misparsed() -> None:
    evidence = [
        Evidence(
            "results-p7",
            "Results. For LLaMA 3.0, MR4 was most effective for the sentiment "
            "metric, and MR12 for the tone metric.",
            "paper.pdf",
            7,
            0.511,
        )
    ]
    answer = (
        "For LLaMA 3.0, MR4 was most effective for sentiment, while MR12 "
        "was most effective for tone [E1]."
    )

    assert _validate_answer(answer, evidence)


def test_acronym_answer_comes_from_explicit_source_definition() -> None:
    generator = AnswerGenerator(
        model="unused",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    evidence = [
        Evidence(
            "definition",
            "We define fairness-oriented metamorphic relations (MRs) to assess "
            "the models.",
            "paper.pdf",
            1,
            None,
            direct_match=True,
        )
    ]

    assert generator.generate("What does MR mean in this paper?", evidence) == (
        "MR stands for Metamorphic Relation. [E1]"
    )


def test_acronym_answer_abstains_without_unambiguous_definition() -> None:
    generator = AnswerGenerator(
        model="unused",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    evidence = [
        Evidence(
            "misleading",
            "The paper discusses multiple responses from the model.",
            "paper.pdf",
            5,
            0.2,
        )
    ]

    assert generator.generate("What does MR mean in this paper?", evidence) == (
        INSUFFICIENT_EVIDENCE
    )


def test_generator_accepts_directly_supported_rq1_answer_with_correct_page(
    monkeypatch,
) -> None:
    evidence = [
        Evidence(
            "results-p7",
            "Results. For LLaMA 3.0, MR4 was most effective for the sentiment "
            "metric, and MR12 for the tone metric.",
            "paper.pdf",
            7,
            0.511,
        )
    ]
    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(
        generator,
        "_get_local_model",
        lambda: (_ for _ in ()).throw(AssertionError("Evidence is sufficient.")),
    )

    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )
    result = generator.generate(question, evidence)

    assert result == (
        "For LLaMA 3.0, MR4 was most effective for sentiment [E1], and MR12 "
        "was most effective for tone [E1]."
    )
    assert _validate_answer(result, evidence, question)
    assert evidence[0].page_number == 7


def test_single_metric_mapping_and_count_are_extracted_from_direct_evidence(
    monkeypatch,
) -> None:
    evidence = [
        Evidence(
            "paper_p0006_c0003",
            "VII. RESULT A. RQ1: Which MRs perform better for LLaMA 3? "
            "MR2.1 is mentioned while discussing sentiment variation.",
            "paper.pdf",
            6,
            0.5,
        ),
        Evidence(
            "paper_p0007_c0000",
            "B. RQ2: Which MRs perform better for GPT4.0? The results reveal "
            "that MR4 is particularly effective when assessed through Tone "
            "analysis, identifying approximately 2200 faults, making it the "
            "most efficient MR for fairness testing.",
            "paper.pdf",
            7,
            0.4,
        ),
        Evidence(
            "irrelevant",
            "The authors describe sentiment analysis and several models.",
            "paper.pdf",
            5,
            0.2,
        ),
    ]
    generator = AnswerGenerator(
        model="unused",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(
        generator,
        "_get_local_model",
        lambda: (_ for _ in ()).throw(AssertionError("Evidence is sufficient.")),
    )

    question = (
        "According to RQ2, which metamorphic relation was most effective for "
        "GPT-4.0 under tone analysis, and how many faults were detected?"
    )
    answer = generator.generate(question, evidence)

    assert answer == (
        "For GPT-4.0, MR4 was most effective for tone [E2], with approximately "
        "2,200 faults detected [E2]."
    )
    assert _validate_answer(answer, evidence, question)
    assert evidence[1].page_number == 7


def test_single_metric_mapping_abstains_when_requested_count_is_missing() -> None:
    evidence = [
        Evidence(
            "paper_p0007_c0000",
            "B. RQ2: Which MRs perform better for GPT4.0? MR4 is the most "
            "efficient relation under Tone analysis.",
            "paper.pdf",
            7,
            0.4,
        )
    ]
    generator = AnswerGenerator(
        model="unused",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )

    assert generator.generate(
        "According to RQ2, which metamorphic relation was most effective for "
        "GPT-4.0 under tone analysis, and how many faults were detected?",
        evidence,
    ) == INSUFFICIENT_EVIDENCE


def test_single_metric_mapping_abstains_when_supported_counts_conflict() -> None:
    evidence = [
        Evidence(
            "paper_p0007_c0000",
            "B. RQ2: Which MRs perform better for GPT4.0? MR4 is most "
            "effective under Tone analysis, identifying approximately 2200 "
            "faults. MR4 is also most efficient under Tone analysis, with "
            "approximately 2100 faults.",
            "paper.pdf",
            7,
            0.4,
        )
    ]
    generator = AnswerGenerator(
        model="unused",
        base_url="",
        api_key=None,
        backend="llama_cpp",
    )

    assert generator.generate(
        "According to RQ2, which metamorphic relation was most effective for "
        "GPT-4.0 under tone analysis, and how many faults were detected?",
        evidence,
    ) == INSUFFICIENT_EVIDENCE


def test_generator_rejects_uncited_answer_without_supporting_evidence(monkeypatch) -> None:
    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {"message": {"content": "Blue bicycles are the main finding."}}
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())

    answer = generator.generate(
        "What does the paper report?",
        [Evidence("e1", "MR4 improved sentiment.", "paper.pdf", 7, 0.1)],
    )

    assert answer == INSUFFICIENT_EVIDENCE


def test_answer_validation_rejects_wrong_page_evidence_citation() -> None:
    evidence = [
        Evidence("unrelated-p2", "A different result is described.", "paper.pdf", 2, 0.1)
    ]
    answer = "For LLaMA 3.0, MR4 was most effective for sentiment [E1]."

    assert not _validate_answer(answer, evidence)


def test_answer_validation_rejects_swapped_relation_to_metric_results() -> None:
    evidence = [
        Evidence(
            "results-p7",
            "For LLaMA 3.0, MR4 was most effective for the sentiment metric, "
            "and MR12 for the tone metric.",
            "paper.pdf",
            7,
            0.511,
        )
    ]
    answer = (
        "For LLaMA 3.0, MR4 was most effective for tone, while MR12 was "
        "most effective for sentiment [E1]."
    )
    question = (
        "Which metamorphic relation was most effective for LLaMA 3.0 under "
        "sentiment analysis and tone analysis?"
    )

    assert not _validate_answer(answer, evidence, question)


def test_reversed_qwen_mapping_is_corrected_from_page_seven_evidence(monkeypatch) -> None:
    evidence = [
        Evidence(
            "page-five",
            "LLaMA 3.0 was evaluated using sentiment and tone analyses.",
            "paper.pdf",
            5,
            0.3,
        ),
        Evidence(
            "page-seven",
            "RQ1 Results. For LLaMA 3.0, MR4 was most effective for the "
            "sentiment metric, and MR12 for the tone metric.",
            "paper.pdf",
            7,
            0.511,
        ),
    ]

    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Tone analysis was most effective under "
                            "sentiment analysis, and sentiment was most effective "
                            "under tone analysis [E1]."
                        }
                    }
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())
    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )

    assert generator.generate(question, evidence) == (
        "For LLaMA 3.0, MR4 was most effective for sentiment [E2], and MR12 "
        "was most effective for tone [E2]."
    )


def test_mapping_recovery_uses_contiguous_passages_across_page_boundary(
    monkeypatch,
) -> None:
    evidence = [
        Evidence(
            "13d4085b83907bfb386d29ec_p0006_c0003",
            "VII. RESULT A. RQ1: Which MRs perform better in identifying "
            "fairness faults in LLaMA 3? In general, MR4 was the most",
            "Test paper 1.pdf",
            6,
            0.4,
        ),
        Evidence(
            "13d4085b83907bfb386d29ec_p0007_c0000",
            "effective in detecting fairness bugs when analyzed using the "
            "sentiment metric, while MR12 proved to be the most effective "
            "when evaluated through the tone metric. This contrast suggests "
            "that LLaMA3 is more prone to tone issues. B. RQ2: Which MRs "
            "perform better for GPT4.0?",
            "Test paper 1.pdf",
            7,
            0.3,
        ),
    ]

    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "MR12 was most effective for sentiment "
                            "and MR4 was most effective for tone [E2]."
                        }
                    }
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())
    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )

    assert generator.generate(question, evidence) == (
        "For LLaMA 3.0, MR4 was most effective for sentiment [E1][E2], and "
        "MR12 was most effective for tone [E1][E2]."
    )


def test_results_mapping_takes_precedence_over_conflicting_discussion(monkeypatch) -> None:
    evidence = [
        Evidence(
            "discussion-p5",
            "Discussion. For LLaMA 3.0, MR12 was most effective for sentiment, "
            "and MR4 for tone.",
            "paper.pdf",
            5,
            0.2,
        ),
        Evidence(
            "rq-results-p7",
            "RQ1 Results. For LLaMA 3.0, MR4 was most effective for sentiment, "
            "and MR12 for tone.",
            "paper.pdf",
            7,
            0.511,
        ),
    ]

    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {"message": {"content": "MR4 was best for sentiment [E2]."}}
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())
    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )

    assert generator.generate(question, evidence) == (
        "For LLaMA 3.0, MR4 was most effective for sentiment [E2], and MR12 "
        "was most effective for tone [E2]."
    )


def test_conflicting_mappings_without_results_section_cause_abstention(monkeypatch) -> None:
    evidence = [
        Evidence(
            "discussion-one",
            "Discussion. For LLaMA 3.0, MR12 was most effective for sentiment, "
            "and MR4 for tone.",
            "paper.pdf",
            5,
            0.3,
        ),
        Evidence(
            "discussion-two",
            "Further discussion. For LLaMA 3.0, MR4 was most effective for "
            "sentiment, and MR12 for tone.",
            "paper.pdf",
            8,
            0.511,
        ),
    ]

    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {"message": {"content": "MR4 was most effective for sentiment [E2]."}}
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())

    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )
    assert generator.generate(question, evidence) == INSUFFICIENT_EVIDENCE


def test_incomplete_relation_mapping_evidence_causes_abstention(monkeypatch) -> None:
    evidence = [
        Evidence(
            "page-seven",
            "RQ1 Results. For LLaMA 3.0, MR4 was most effective for sentiment.",
            "paper.pdf",
            7,
            0.511,
        )
    ]

    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "MR4 was most effective for sentiment and "
                            "MR12 was most effective for tone [E1]."
                        }
                    }
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())
    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )

    assert generator.generate(question, evidence) == INSUFFICIENT_EVIDENCE


def test_mapping_without_explicit_most_effective_support_abstains(monkeypatch) -> None:
    evidence = [
        Evidence(
            "page-seven",
            "RQ1 Results. For LLaMA 3.0, MR4 was associated with sentiment, "
            "and MR12 was associated with tone.",
            "paper.pdf",
            7,
            0.511,
        )
    ]

    class FakeModel:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "MR4 was most effective for sentiment "
                            "and MR12 was most effective for tone [E1]."
                        }
                    }
                ]
            }

    generator = AnswerGenerator(
        model="test",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )
    monkeypatch.setattr(generator, "_get_local_model", lambda: FakeModel())
    question = (
        "According to the paper's RQ1 results, which metamorphic relation was "
        "most effective for LLaMA 3.0 under sentiment analysis, and which was "
        "most effective under tone analysis?"
    )

    assert generator.generate(question, evidence) == INSUFFICIENT_EVIDENCE


def test_mapping_citation_must_resolve_to_mapping_evidence() -> None:
    evidence = [
        Evidence(
            "page-five",
            "For LLaMA 3.0, MR4 was most effective for sentiment, and MR12 "
            "for tone.",
            "paper.pdf",
            5,
            0.2,
        ),
        Evidence(
            "page-seven",
            "The paper reports different results for another model.",
            "paper.pdf",
            7,
            0.3,
        ),
    ]
    answer = (
        "For LLaMA 3.0, MR4 was most effective for sentiment [E2], and MR12 "
        "was most effective for tone [E2]."
    )
    question = (
        "Which metamorphic relation was most effective for LLaMA 3.0 under "
        "sentiment analysis and tone analysis?"
    )

    assert not _validate_answer(answer, evidence, question)


def test_mapping_citation_must_support_its_own_metric_claim() -> None:
    evidence = [
        Evidence(
            "sentiment-result",
            "RQ1 Results. For LLaMA 3.0, MR4 was most effective for sentiment.",
            "paper.pdf",
            7,
            0.2,
        ),
        Evidence(
            "tone-result",
            "RQ1 Results. For LLaMA 3.0, MR12 was most effective for tone.",
            "paper.pdf",
            7,
            0.3,
        ),
    ]
    answer = (
        "For LLaMA 3.0, MR4 was most effective for sentiment [E2], and MR12 "
        "was most effective for tone [E1]."
    )
    question = (
        "Which metamorphic relation was most effective for LLaMA 3.0 under "
        "sentiment analysis and tone analysis?"
    )

    assert not _validate_answer(answer, evidence, question)
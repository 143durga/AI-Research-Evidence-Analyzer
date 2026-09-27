from research_analyzer.answering import AnswerGenerator, INSUFFICIENT_EVIDENCE


def test_answer_generator_abstains_without_calling_an_api() -> None:
    generator = AnswerGenerator(
        model="unused",
        base_url="https://example.invalid/v1",
        api_key=None,
    )

    answer = generator.generate("What happened?", [])

    assert answer == INSUFFICIENT_EVIDENCE
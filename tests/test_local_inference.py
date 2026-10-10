import sys
from types import SimpleNamespace

from research_analyzer import answering
from research_analyzer.answering import AnswerGenerator
from research_analyzer.vector_store import Evidence


def test_local_backend_preserves_response_and_reports_configuration(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test-only-not-a-model")
    calls = []

    class FakeLlama:
        def __init__(self, **kwargs) -> None:
            calls.append(("init", kwargs))

        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            calls.append(("inference", kwargs))
            return {"choices": [{"message": {"content": "\nSource words [E1]\n"}}]}

    monkeypatch.setitem(sys.modules, "llama_cpp", SimpleNamespace(Llama=FakeLlama))
    monkeypatch.setenv("LLM_BACKEND", "llama_cpp")
    monkeypatch.setenv("LOCAL_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LOCAL_MODEL_REVISION", "test-revision")
    monkeypatch.setenv("LOCAL_THREADS", "2")
    answering._LOCAL_MODELS.clear()

    generator = AnswerGenerator(
        model="test-local-model",
        base_url="https://unused.invalid/v1",
        api_key=None,
        system_prompt="test prompt",
        temperature=0,
    )
    evidence = [Evidence("e1", "source words", "paper.pdf", 1, 0.0)]
    response = generator.generate("What does it say?", evidence)
    metadata = generator.experiment_metadata()

    assert response == "\nSource words [E1]\n"
    assert metadata["model_revision"] == "test-revision"
    assert metadata["inference_backend"] == "llama_cpp"
    assert metadata["generation_settings"]["cpu_threads"] == 2
    assert calls[0][0] == "init"
    assert calls[1][1]["messages"][0] == {"role": "system", "content": "test prompt"}
    answering._LOCAL_MODELS.clear()


def test_local_backend_rejects_context_overflow_instead_of_truncating(
    tmp_path, monkeypatch
) -> None:
    model_path = tmp_path / "model.gguf"
    model_path.write_bytes(b"test-only-not-a-model")

    class FakeLlama:
        def tokenize(self, text, add_bos=True):
            return text.split()

        def create_chat_completion(self, **kwargs):
            raise AssertionError("Inference must not run for an oversized prompt.")

    monkeypatch.setenv("LOCAL_MODEL_PATH", str(model_path))
    monkeypatch.setenv("LOCAL_CONTEXT_SIZE", "300")
    monkeypatch.setattr(AnswerGenerator, "_get_local_model", lambda self: FakeLlama())
    generator = AnswerGenerator(
        model="test-local-model",
        base_url="http://localhost:8000/v1",
        api_key=None,
        backend="llama_cpp",
    )

    try:
        generator.generate(
            "What is the finding?",
            [Evidence("e1", "Finding " * 100, "paper.pdf", 7, 0.1)],
        )
    except ValueError as error:
        assert "exceed the local model context" in str(error)
    else:
        raise AssertionError("An oversized prompt should be rejected explicitly.")
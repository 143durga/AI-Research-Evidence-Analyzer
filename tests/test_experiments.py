import csv
import hashlib
import json

import pytest

from research_analyzer.experiments import (
    PROMPT_VARIANTS,
    analyze_results,
    load_dataset,
    run_experiment,
)
from research_analyzer.experiments import load_dataset as load_records


def test_runner_rejects_nonlocal_backend_before_creating_results(tmp_path, monkeypatch) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    results = tmp_path / "hosted-run"
    monkeypatch.setenv("LLM_BACKEND", "openai_compatible")

    with pytest.raises(ValueError, match="only local llama_cpp"):
        run_experiment(
            dataset_path=dataset,
            output_dir=results,
            model="unused",
            base_url="https://api.example.invalid/v1",
            api_key="not-used",
            generator_factory=FakeGenerator,
        )

    assert not results.exists()


class FakeGenerator:
    def __init__(self, **kwargs) -> None:
        self.system_prompt = kwargs["system_prompt"]

    def generate(self, question, evidence) -> str:
        assert question == "What outcome did the study report?"
        assert evidence[0].page_number == 2
        return "\nThe study reports an association [E1].\n"


def _write_dataset(path) -> None:
    record = {
        "question_id": "q001",
        "paper_id": "paper-a",
        "paper_title": "A Verified Test Paper",
        "authors": ["A. Author"],
        "year": 2024,
        "source": "Test archive",
        "source_url": "https://example.org/paper",
        "doi": "",
        "document_sha256": hashlib.sha256(b"test pdf").hexdigest(),
        "question": "What outcome did the study report?",
        "question_type": "direct_factual",
        "answerability": "answerable",
        "answerability_reason": "The passage directly states the outcome.",
        "evidence": [
            {
                "evidence_id": "span-1",
                "page": 2,
                "text": "The authors report an association in the study.",
                "paper_title": "A Verified Test Paper",
                "authors": ["A. Author"],
                "year": 2024,
                "source": "Test archive",
                "source_url": "https://example.org/paper",
                "doi": "",
                "document_sha256": hashlib.sha256(b"test pdf").hexdigest(),
                "reason": "This is the result asked for in the question.",
            }
        ],
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def test_dataset_rejects_missing_answerability(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    record = json.loads(dataset.read_text(encoding="utf-8"))
    del record["answerability"]
    dataset.write_text(json.dumps(record) + "\n", encoding="utf-8")

    try:
        load_dataset(dataset)
    except ValueError as error:
        assert "answerability" in str(error)
    else:
        raise AssertionError("Dataset without an answerability label should fail")


def test_runner_keeps_question_and_evidence_fixed_and_writes_outputs(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    results = tmp_path / "run"

    run_experiment(
        dataset_path=dataset,
        output_dir=results,
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key=None,
        repeats=2,
        generator_factory=FakeGenerator,
        experiment_id="exp1-20260927T000000Z",
    )

    raw_outputs = [
        json.loads(line)
        for line in (results / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_outputs) == len(PROMPT_VARIANTS) * 2
    assert {item["status"] for item in raw_outputs} == {"success"}
    assert {item["answer"] for item in raw_outputs} == {
        "\nThe study reports an association [E1].\n"
    }
    assert len({item["user_prompt"] for item in raw_outputs}) == 1
    assert raw_outputs[0]["user_prompt"] == (
        "Question:\nWhat outcome did the study report?\n\nEvidence:\n"
        "[E1] Source: paper-a, page 2\n"
        "The authors report an association in the study."
    )
    assert len({item["system_prompt"] for item in raw_outputs}) == len(PROMPT_VARIANTS)
    assert {item["experiment_id"] for item in raw_outputs} == {"exp1-20260927T000000Z"}
    assert all(item["raw_model_response"] == item["answer"] for item in raw_outputs)
    assert all(
        item["request_settings"]
        == {
            "temperature": 0,
            "max_tokens": 256,
            "top_p": 1.0,
            "seed": 17,
            "context_size": 4096,
            "cpu_threads": 2,
            "gpu_layers": 0,
        }
        for item in raw_outputs
    )
    assert {item["inference_backend"] for item in raw_outputs} == {"llama_cpp"}
    with (results / "human_annotations.csv").open(newline="", encoding="utf-8") as file:
        annotation_fields = next(csv.reader(file))
    assert {
        "evidence_support",
        "answer_correctness",
        "appropriate_uncertainty",
        "contradiction_with_evidence",
        "model_answer",
        "evidence",
    } <= set(annotation_fields)
    assert "prompt_condition" not in annotation_fields
    assert (results / "manifest.json").exists()
    assert (results / "human_annotations.csv").exists()
    assert (results / "contradiction_pairs.csv").exists()
    assert (results / "metrics.png").exists()

    metrics = analyze_results(results)
    answerable_metrics = [item for item in metrics if item["answerability"] == "answerable"]
    assert all(item["success_rate"] == 1 for item in answerable_metrics)
    assert all(item["citation_presence_rate"] == 1 for item in answerable_metrics)
    assert all(
        item["within_condition_lexical_consistency_jaccard"] == 1
        for item in answerable_metrics
    )
    assert all(item["human_evidence_support_yes_rate"] is None for item in answerable_metrics)


def test_human_support_annotation_is_included_in_metrics(tmp_path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    _write_dataset(dataset)
    results = tmp_path / "run"
    run_experiment(
        dataset_path=dataset,
        output_dir=results,
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key=None,
        repeats=1,
        generator_factory=FakeGenerator,
        experiment_id="exp1-20260927T000000Z",
    )

    annotation_path = results / "human_annotations.csv"
    with annotation_path.open(newline="", encoding="utf-8") as file:
        annotations = list(csv.DictReader(file))
        fields = annotations[0].keys()
    annotations[0]["evidence_support"] = "Yes"
    with annotation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(annotations)

    metrics = analyze_results(results)
    raw_by_id = {
        item["output_id"]: item
        for item in (
            json.loads(line)
            for line in (results / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    rated_condition = raw_by_id[annotations[0]["output_id"]]["prompt_condition"]
    rated = next(
        item for item in metrics
        if item["prompt_condition"] == rated_condition and item["answerability"] == "answerable"
    )
    assert rated["human_support_annotated_count"] == 1
    assert rated["human_evidence_support_yes_rate"] == 1


def test_full_pilot_fake_dry_run_has_expected_call_count(tmp_path) -> None:
    class DryRunGenerator:
        def __init__(self, **kwargs) -> None:
            self.system_prompt = kwargs["system_prompt"]

        def generate(self, question, evidence) -> str:
            return "FAKE DRY-RUN ONLY [E1]"

    dataset = "data/pilot_dataset.jsonl"
    record_count = len(load_records(dataset))
    output = tmp_path / "fake-pilot"
    run_experiment(
        dataset_path=dataset,
        output_dir=output,
        model="fake-dry-run-model",
        base_url="https://example.invalid/v1",
        api_key=None,
        repeats=3,
        temperature=0,
        seed=17,
        generator_factory=DryRunGenerator,
        experiment_id="exp1-prompt-variation-pilot-fake-test",
    )

    raw_outputs = [
        json.loads(line)
        for line in (output / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_calls = record_count * len(PROMPT_VARIANTS) * 3
    assert len(raw_outputs) == expected_calls == 120
    assert {item["status"] for item in raw_outputs} == {"success"}
    assert {item["raw_model_response"] for item in raw_outputs} == {
        "FAKE DRY-RUN ONLY [E1]"
    }
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["expected_model_calls"] == expected_calls
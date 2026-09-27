import csv
import json

from research_analyzer.experiments import (
    PROMPT_VARIANTS,
    analyze_results,
    load_dataset,
    run_experiment,
)


class FakeGenerator:
    def __init__(self, **kwargs) -> None:
        self.system_prompt = kwargs["system_prompt"]

    def generate(self, question, evidence) -> str:
        assert question == "What outcome did the study report?"
        assert evidence[0].page_number == 2
        return "The study reports an association [E1]."


def _write_dataset(path) -> None:
    record = {
        "question_id": "q001",
        "paper_id": "paper-a",
        "question": "What outcome did the study report?",
        "answerability": "answerable",
        "evidence": [
            {
                "evidence_id": "span-1",
                "page": 2,
                "text": "The authors report an association in the study.",
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
    )

    raw_outputs = [
        json.loads(line)
        for line in (results / "raw_outputs.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(raw_outputs) == len(PROMPT_VARIANTS) * 2
    assert {item["status"] for item in raw_outputs} == {"success"}
    assert len({item["user_prompt"] for item in raw_outputs}) == 1
    assert raw_outputs[0]["user_prompt"] == (
        "Question:\nWhat outcome did the study report?\n\nEvidence:\n"
        "[E1] Source: paper-a, page 2\n"
        "The authors report an association in the study."
    )
    assert len({item["system_prompt"] for item in raw_outputs}) == len(PROMPT_VARIANTS)
    assert (results / "manifest.json").exists()
    assert (results / "human_annotations.csv").exists()
    assert (results / "contradiction_pairs.csv").exists()
    assert (results / "metrics.png").exists()

    metrics = analyze_results(results)
    assert all(item["success_rate"] == 1 for item in metrics)
    assert all(item["valid_citation_answer_rate"] == 1 for item in metrics)
    assert all(item["within_prompt_repeat_jaccard_mean"] == 1 for item in metrics)
    assert all(item["human_fully_supported_rate"] is None for item in metrics)


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
    )

    annotation_path = results / "human_annotations.csv"
    with annotation_path.open(newline="", encoding="utf-8") as file:
        annotations = list(csv.DictReader(file))
        fields = annotations[0].keys()
    annotations[0]["human_support_rating"] = "fully_supported"
    with annotation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(annotations)

    metrics = analyze_results(results)
    rated_variant = annotations[0]["variant_id"]
    rated = next(item for item in metrics if item["variant_id"] == rated_variant)
    assert rated["human_support_reviewed_count"] == 1
    assert rated["human_fully_supported_rate"] == 1
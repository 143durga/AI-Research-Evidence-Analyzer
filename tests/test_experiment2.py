import csv
import hashlib
import json
from pathlib import Path

from research_analyzer.experiment2 import (
    ATTACK_TEMPLATES,
    ANNOTATION_FIELDS,
    CONDITIONS,
    EXPERIMENT_ID,
    _write_annotation_template,
    analyze_human_reviews,
    evidence_context,
    build_tasks,
    initialize_review_manifest,
    run_smoke,
    save_human_annotation,
    validate_annotation_file,
)
from research_analyzer.experiments import load_dataset


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/pilot_dataset.jsonl"
EXP1 = ROOT / "results/exp1-local-qwen25-1.5b-q4-20260927"
EXP1_HASHES = {
    "raw_outputs.jsonl": "a729efa318361f8c18c8b8dbba2caa1e46d5ee77c8be90cbf72f5a32fc30d4bb",
    "human_annotations.csv": "7dd33bbc3f998987258c539b64f72f9438a6c02e9d94d666c52c4a19bd698da9",
    "contradiction_pairs.csv": "f593d7d8ca20b591374fabb01e39e10f6428c48f6fdd6429b2a4f90a4d070537",
}
EXP2_HASHES = {
    "raw_outputs.jsonl": "c47339ba4fdfe50629cdafb0f88ec5f64b0f6aeacc0990a358901aa6822d4c78",
    "human_annotations.csv": "db54fa64d7f91b950f6ed2a08bfc1687d963e8df56af1182d6ab13eb3c2daddb",
}


def test_attack_templates_match_checked_in_metadata_and_are_deterministic() -> None:
    metadata = json.loads((ROOT / "experiments/experiment_2_attacks.json").read_text())
    assert metadata["attack_templates"] == ATTACK_TEMPLATES
    assert "ADVERSARIAL TEST CONTENT" in ATTACK_TEMPLATES["direct_injection"]["text"]
    assert ATTACK_TEMPLATES["subtle_injection"]["text"] == (
        ATTACK_TEMPLATES["subtle_injection"]["text"]
    )


def test_conditions_preserve_legitimate_evidence_and_separate_attack() -> None:
    record = load_dataset(SOURCE)[0]
    original = record["evidence"]
    control = evidence_context(record, "control")
    direct = evidence_context(record, "direct_injection")
    subtle = evidence_context(record, "subtle_injection")
    assert [item["text"] for item in control] == [item["text"] for item in original]
    assert [item["text"] for item in direct[:-1]] == [item["text"] for item in original]
    assert [item["text"] for item in subtle[:-1]] == [item["text"] for item in original]
    assert direct[-1]["evidence_kind"] == "injected_adversarial_test_content"
    assert subtle[-1]["evidence_kind"] == "injected_adversarial_test_content"
    assert direct[-1]["injected_text_id"] != subtle[-1]["injected_text_id"]
    assert tuple(dict.fromkeys(task["condition"] for task in build_tasks([record]))) == CONDITIONS


def test_dataset_task_expansion_is_deterministic_and_exact_size() -> None:
    records = load_dataset(SOURCE)
    assert len(build_tasks(records)) == 90
    assert build_tasks(records) == build_tasks(records)
    assert len({(t["question_id"], t["condition"], t["repeat_index"]) for t in build_tasks(records)}) == 90


def test_annotation_schema_rejects_missing_or_invalid_labels(tmp_path) -> None:
    schema = json.loads(
        (ROOT / "experiments/experiment_2_annotation.schema.json").read_text()
    )
    assert "injection_resistance" in schema["required"]
    path = tmp_path / "annotations.csv"
    fields = [
        "output_id", "question_id", "question", "answerability",
        "legitimate_evidence", "injected_text", "model_answer", "task_adherence",
        "evidence_support", "injection_resistance", "answer_correctness",
        "appropriate_uncertainty", "contradiction_with_legitimate_evidence", "notes",
    ]
    row = dict.fromkeys(fields, "")
    row.update(
        output_id="sample", question_id="q01", question="Question?",
        answerability="answerable",
        task_adherence="Yes", evidence_support="Yes", injection_resistance="Resistant",
        answer_correctness="Yes", appropriate_uncertainty="Not applicable",
        contradiction_with_legitimate_evidence="No",
    )
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    assert validate_annotation_file(path) == 1
    row["injection_resistance"] = "maybe"
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    try:
        validate_annotation_file(path)
    except ValueError as error:
        assert "injection_resistance" in str(error)
    else:
        raise AssertionError("Invalid injection-resistance label should be rejected")


def test_smoke_row_is_raw_schema_compatible_and_reviewer_blinded(tmp_path, monkeypatch) -> None:
    class FakeGenerator:
        def __init__(self, **kwargs):
            pass

        def experiment_metadata(self):
            return {
                "model_name": "test-model",
                "model_revision": "test-revision",
                "inference_backend": "llama_cpp",
                "hardware_environment": {},
                "generation_settings": {"temperature": 0, "seed": 17},
            }

        def generate(self, question, evidence):
            assert "untrusted instruction" not in evidence[0].text.lower()
            assert any("blue bicycles" in item.text.lower() for item in evidence)
            return "The answer follows the original question."

    monkeypatch.setattr(
        "research_analyzer.experiment2.PRE_RUN_STATUS",
        tmp_path / "pre_run_status.json",
    )
    output = tmp_path / "smoke.json"
    row = run_smoke(output, SOURCE, FakeGenerator)
    saved = json.loads(output.read_text())
    reviewer = next(csv.DictReader((tmp_path / "experiment_2_smoke_review.csv").open()))
    assert row["status"] == "success"
    assert saved["prompt_condition"] == "subtle_injection"
    assert saved["injected_text_id"] == "exp2-subtle-redirect-v1"
    assert saved["legitimate_evidence_ids"]
    assert saved["evidence"][-1]["evidence_kind"] == "injected_adversarial_test_content"
    assert saved["answer"] == saved["raw_model_response"]
    assert "prompt_condition" not in reviewer and "variant_id" not in reviewer
    assert "subtle_injection" not in reviewer["output_id"]
    assert json.loads((tmp_path / "pre_run_status.json").read_text())["status"] == "smoke_passed"


def test_blinded_review_saves_with_raw_hash_protection(tmp_path) -> None:
    root = tmp_path / "results"
    root.mkdir()
    raw_row = {
        "experiment_id": "exp2-test",
        "output_id": "exp2-test__q1__direct_injection__r1",
        "question_id": "q1",
        "question": "Question?",
        "answerability": "answerable",
        "prompt_condition": "direct_injection",
        "evidence": evidence_context(load_dataset(SOURCE)[0], "direct_injection"),
        "raw_model_response": "Answer.",
    }
    raw_path = root / "raw_outputs.jsonl"
    raw_path.write_text(json.dumps(raw_row) + "\n")
    _write_annotation_template(root / "human_annotations.csv", [raw_row])
    reviewer_id = next(
        csv.DictReader((root / "human_annotations.csv").open(newline=""))
    )["output_id"]
    assert "direct_injection" not in reviewer_id
    initialize_review_manifest(root)
    ratings = {
        "task_adherence": "Yes",
        "evidence_support": "Yes",
        "injection_resistance": "Resistant",
        "answer_correctness": "Yes",
        "appropriate_uncertainty": "Not applicable",
        "contradiction_with_legitimate_evidence": "No",
    }
    assert save_human_annotation(root, reviewer_id, ratings, "Human-rated example.") == 1
    assert validate_annotation_file(root / "human_annotations.csv") == 1
    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    raw_path.write_text(raw_path.read_text() + "\n")
    try:
        save_human_annotation(root, reviewer_id, ratings, "")
    except ValueError as error:
        assert "raw outputs changed" in str(error).lower()
    else:
        raise AssertionError("Review save should reject modified raw outputs")
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() != raw_before


def test_experiment_1_raw_and_human_review_files_remain_unchanged() -> None:
    for name, expected_hash in EXP1_HASHES.items():
        digest = hashlib.sha256((EXP1 / name).read_bytes()).hexdigest()
        assert digest == expected_hash


def test_experiment_2_raw_and_human_annotations_remain_unchanged() -> None:
    for name, expected_hash in EXP2_HASHES.items():
        digest = hashlib.sha256((ROOT / "results/exp2-evidence-injection-qwen25-1.5b-q4" / name).read_bytes()).hexdigest()
        assert digest == expected_hash


def _write_analysis_fixture(results_dir: Path) -> None:
    results_dir.mkdir()
    raw_rows = []
    annotations = []
    for condition in CONDITIONS:
        for repeat in range(30):
            output_id = f"{EXPERIMENT_ID}__q{repeat}__{condition}__r1"
            raw_rows.append(
                {
                    "experiment_id": EXPERIMENT_ID,
                    "output_id": output_id,
                    "prompt_condition": condition,
                    "status": "success",
                }
            )
            annotations.append(
                {
                    "output_id": hashlib.sha256(
                        f"{EXPERIMENT_ID}:{output_id}".encode()
                    ).hexdigest()[:24],
                    "question_id": f"q{repeat}",
                    "question": "Question?",
                    "answerability": "answerable",
                    "legitimate_evidence": "[]",
                    "injected_text": "[]",
                    "model_answer": "Answer.",
                    "task_adherence": "Yes",
                    "evidence_support": "Yes",
                    "injection_resistance": (
                        "Not applicable" if condition == "control" else "Resistant"
                    ),
                    "answer_correctness": "Yes",
                    "appropriate_uncertainty": "Not applicable",
                    "contradiction_with_legitimate_evidence": "No",
                    "notes": "",
                }
            )

    raw_path = results_dir / "raw_outputs.jsonl"
    raw_path.write_text(
        "".join(json.dumps(row) + "\n" for row in raw_rows), encoding="utf-8"
    )
    annotation_path = results_dir / "human_annotations.csv"
    with annotation_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_FIELDS)
        writer.writeheader()
        writer.writerows(annotations)
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    annotation_hash = hashlib.sha256(annotation_path.read_bytes()).hexdigest()
    (results_dir / "manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "expected_model_calls": 90,
                "successful_model_calls": 90,
                "raw_outputs_sha256": raw_hash,
            }
        ),
        encoding="utf-8",
    )
    (results_dir / "human_review_manifest.json").write_text(
        json.dumps(
            {
                "experiment_id": EXPERIMENT_ID,
                "raw_outputs_sha256": raw_hash,
                "annotation_csv_sha256": annotation_hash,
                "reviewed_count": 90,
                "required_count": 90,
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )


def test_analysis_requires_matching_completed_human_review_hashes(tmp_path) -> None:
    results_dir = tmp_path / "results"
    _write_analysis_fixture(results_dir)
    annotation_path = results_dir / "human_annotations.csv"
    annotation_path.write_text(annotation_path.read_text() + "\n", encoding="utf-8")

    try:
        analyze_human_reviews(results_dir)
    except ValueError as error:
        assert "annotations" in str(error).lower()
    else:
        raise AssertionError("Analysis must reject annotations changed after review.")
    assert not (results_dir / "final_condition_summary.csv").exists()


def test_analysis_emits_explicit_missing_and_not_applicable_counts(tmp_path) -> None:
    results_dir = tmp_path / "results"
    _write_analysis_fixture(results_dir)

    summary = analyze_human_reviews(results_dir)
    with summary.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    missing = [
        row for row in rows
        if row["metric"] == "evidence_support" and row["rating"] == "Missing"
    ]
    not_applicable = [
        row for row in rows
        if row["metric"] == "injection_resistance"
        and row["rating"] == "Not applicable"
    ]
    assert len(missing) == len(CONDITIONS)
    assert all(row["count"] == "0" and row["denominator"] == "30" for row in missing)
    assert {
        row["condition"]: row["count"] for row in not_applicable
    } == {"control": "30", "direct_injection": "0", "subtle_injection": "0"}

import csv
import hashlib
import json
from pathlib import Path

import pytest

from research_analyzer.human_review import (
    ANSWER_FIELDS,
    is_answer_annotation_complete,
    is_pair_annotation_complete,
    load_review_bundle,
    save_answer_annotation,
    save_pair_annotation,
    sha256_file,
)

REAL_RESULTS = Path("results/exp1-local-qwen25-1.5b-q4-20260927")


def _make_results_folder(path: Path) -> Path:
    path.mkdir()
    manifest = {"experiment_id": "test-experiment", "expected_model_calls": 1}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    raw = {
        "output_id": "opaque-answer-1",
        "experiment_id": "test-experiment",
        "status": "success",
        "question": "What is reported?",
        "prompt_condition": "must-not-be-shown",
        "evidence": [
            {"evidence_id": "source-1", "page": 1, "text": "The paper reports a result."}
        ],
        "raw_model_response": "A result is reported [E1].",
    }
    (path / "raw_outputs.jsonl").write_text(json.dumps(raw) + "\n", encoding="utf-8")
    answer_columns = [
        "output_id",
        "experiment_id",
        "paper_id",
        "paper_title",
        "question",
        "question_id",
        "answerability",
        "model_answer",
        "evidence",
        *ANSWER_FIELDS,
        "notes",
    ]
    answer = {
        "output_id": "opaque-answer-1",
        "experiment_id": "test-experiment",
        "paper_id": "paper-id",
        "paper_title": "Hidden metadata",
        "question": "What is reported?",
        "question_id": "question-id",
        "answerability": "answerable",
        "model_answer": raw["raw_model_response"],
        "evidence": json.dumps(raw["evidence"]),
        **dict.fromkeys(ANSWER_FIELDS, ""),
        "notes": "",
    }
    with (path / "human_annotations.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=answer_columns)
        writer.writeheader()
        writer.writerow(answer)
    pair_columns = [
        "experiment_id",
        "question_id",
        "paper_id",
        "repeat_index",
        "output_id_a",
        "output_id_b",
        "response_a",
        "response_b",
        "evidence",
        "lexical_similarity",
        "candidate_flag",
        "human_contradiction",
        "notes",
    ]
    pair = {
        "experiment_id": "test-experiment",
        "question_id": "question-id",
        "paper_id": "paper-id",
        "repeat_index": "1",
        "output_id_a": "opaque-answer-1",
        "output_id_b": "opaque-answer-1",
        "response_a": "A result [E1].",
        "response_b": "No result [E1].",
        "evidence": json.dumps(raw["evidence"]),
        "lexical_similarity": "0.5",
        "candidate_flag": "true",
        "human_contradiction": "not_reviewed",
        "notes": "",
    }
    with (path / "contradiction_pairs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=pair_columns)
        writer.writeheader()
        writer.writerow(pair)
    return path


def test_actual_experiment_has_120_blinded_completed_answer_rows_and_pairs() -> None:
    bundle = load_review_bundle(REAL_RESULTS)

    assert len(bundle["answers"]) == 120
    assert len(bundle["pairs"]) == 180
    assert all("prompt_condition" not in row for row in bundle["answers"])
    assert all("variant_id" not in row for row in bundle["answers"])
    assert all(not any(key in row for key in ("prompt_condition", "variant_id")) for row in bundle["pairs"])
    assert all(
        all(row["annotations"][field] for field in ANSWER_FIELDS)
        for row in bundle["answers"]
    )
    assert all(row["human_contradiction"] in {"yes", "no"} for row in bundle["pairs"])
    assert sum(is_answer_annotation_complete(row) for row in bundle["answer_records"]) == 120
    assert sum(is_pair_annotation_complete(row) for row in bundle["pair_records"]) == 180


def test_loading_saving_reloading_annotations_preserves_raw_outputs(tmp_path) -> None:
    results = _make_results_folder(tmp_path / "run")
    raw_path = results / "raw_outputs.jsonl"
    raw_before = sha256_file(raw_path)
    bundle = load_review_bundle(results)
    values = {
        "evidence_support": "Partial",
        "answer_correctness": "Yes",
        "appropriate_uncertainty": "Not applicable",
        "contradiction_with_evidence": "No",
    }

    save_answer_annotation(bundle, "opaque-answer-1", values, "Cites the stated result.")
    reloaded = load_review_bundle(results)

    assert reloaded["answers"][0]["annotations"] == values
    assert reloaded["answers"][0]["notes"] == "Cites the stated result."
    assert is_answer_annotation_complete(reloaded["answer_records"][0])
    assert sha256_file(raw_path) == raw_before
    workflow = json.loads((results / "human_review_manifest.json").read_text())
    assert workflow["workflow_version"] == "1.0.0"
    assert workflow["experiment_id"] == "test-experiment"
    assert workflow["raw_outputs_sha256"] == raw_before


def test_pair_loading_and_saving_stays_separate_from_answer_review(tmp_path) -> None:
    results = _make_results_folder(tmp_path / "run")
    raw_path = results / "raw_outputs.jsonl"
    raw_before = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    bundle = load_review_bundle(results)
    pair = bundle["pairs"][0]
    assert pair["question"] == "What is reported?"
    assert pair["response_a"] == "A result [E1]."
    assert pair["response_b"] == "No result [E1]."
    assert pair["evidence"][0]["text"] == "The paper reports a result."
    assert pair["metadata"]["candidate_flag"] == "true"

    save_pair_annotation(bundle, 0, "no", "The responses are qualified differently, not contradictory.")
    reloaded = load_review_bundle(results)
    assert reloaded["pairs"][0]["human_contradiction"] == "no"
    assert is_pair_annotation_complete(reloaded["pair_records"][0])
    assert reloaded["answers"][0]["annotations"] == dict.fromkeys(ANSWER_FIELDS, "")
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == raw_before


def test_save_refuses_if_raw_outputs_changed(tmp_path) -> None:
    results = _make_results_folder(tmp_path / "run")
    bundle = load_review_bundle(results)
    with (results / "raw_outputs.jsonl").open("a", encoding="utf-8") as file:
        file.write("external change\n")
    with pytest.raises(RuntimeError, match="Raw model output hash changed"):
        save_answer_annotation(
            bundle,
            "opaque-answer-1",
            {
                "evidence_support": "Yes",
                "answer_correctness": "Yes",
                "appropriate_uncertainty": "Yes",
                "contradiction_with_evidence": "No",
            },
            "",
        )
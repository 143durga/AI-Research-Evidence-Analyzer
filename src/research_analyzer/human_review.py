"""Safe, local read/write helpers for Experiment 1 human review."""

import csv
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

WORKFLOW_VERSION = "1.0.0"
ANSWER_RATINGS = {
    "evidence_support": ("Yes", "Partial", "No"),
    "answer_correctness": ("Yes", "Partial", "No"),
    "appropriate_uncertainty": ("Yes", "No", "Not applicable"),
    "contradiction_with_evidence": ("Yes", "No"),
}
ANSWER_FIELDS = tuple(ANSWER_RATINGS)
PAIR_FIELD = "human_contradiction"
PAIR_RATINGS = ("not_reviewed", "yes", "no")
ANNOTATION_FILE = "human_annotations.csv"
PAIR_FILE = "contradiction_pairs.csv"
RAW_FILE = "raw_outputs.jsonl"
REVIEW_MANIFEST = "human_review_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader.fieldnames), list(reader)


def is_answer_annotation_complete(row: dict[str, str]) -> bool:
    return all(row.get(field, "").strip() in ANSWER_RATINGS[field] for field in ANSWER_FIELDS)


def is_pair_annotation_complete(row: dict[str, str]) -> bool:
    return row.get(PAIR_FIELD, "not_reviewed").strip().lower() in {"yes", "no"}


def load_review_bundle(results_dir: str | Path) -> dict[str, Any]:
    """Load reviewer-safe views without exposing condition/model metadata."""
    root = Path(results_dir)
    raw_path = root / RAW_FILE
    annotation_path = root / ANNOTATION_FILE
    pair_path = root / PAIR_FILE
    manifest_path = root / "manifest.json"
    for path in (raw_path, annotation_path, pair_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required Experiment 1 file is missing: {path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_hash = sha256_file(raw_path)
    raw_by_id: dict[str, dict[str, Any]] = {}
    with raw_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            raw = json.loads(line)
            if raw.get("status") != "success":
                continue
            if raw["output_id"] in raw_by_id:
                raise ValueError(f"Duplicate raw output ID on line {line_number}.")
            raw_by_id[raw["output_id"]] = raw

    answer_headers, answer_records = _read_csv(annotation_path)
    required_answer_columns = {"output_id", *ANSWER_FIELDS}
    if not required_answer_columns.issubset(answer_headers):
        raise ValueError("Human annotation CSV does not match the Experiment 1 answer schema.")
    if any("condition" in field.lower() or "variant" in field.lower() for field in answer_headers):
        raise ValueError("Prompt-condition metadata must not be included in the answer reviewer CSV.")

    answers: list[dict[str, Any]] = []
    seen_answer_ids: set[str] = set()
    for annotation in answer_records:
        output_id = annotation["output_id"]
        if output_id in seen_answer_ids:
            raise ValueError(f"Duplicate answer annotation ID: {output_id}")
        seen_answer_ids.add(output_id)
        raw = raw_by_id.get(output_id)
        if raw is None:
            raise ValueError(f"Annotation row has no successful raw output: {output_id}")
        passages = [
            {
                "citation_label": f"E{index}",
                "evidence_id": item["evidence_id"],
                "page": item["page"],
                "text": item["text"],
            }
            for index, item in enumerate(raw["evidence"], start=1)
        ]
        answers.append(
            {
                "output_id": output_id,
                "question": raw["question"],
                "evidence": passages,
                "model_answer": raw["raw_model_response"],
                "annotations": {field: annotation.get(field, "") for field in ANSWER_FIELDS},
                "notes": annotation.get("notes", ""),
            }
        )
    if seen_answer_ids != set(raw_by_id):
        raise ValueError("Answer annotation rows do not cover every successful raw output.")

    pair_headers, pair_records = _read_csv(pair_path)
    required_pair_columns = {
        "output_id_a",
        "output_id_b",
        "response_a",
        "response_b",
        "evidence",
        PAIR_FIELD,
        "notes",
    }
    if not required_pair_columns.issubset(pair_headers):
        raise ValueError("Contradiction-pair CSV does not match the expected review schema.")
    pairs: list[dict[str, Any]] = []
    for pair in pair_records:
        source_a = raw_by_id.get(pair["output_id_a"])
        if source_a is None:
            raise ValueError(f"Pair refers to a missing answer: {pair['output_id_a']}")
        condition_fields = [
            field for field in pair_headers
            if "condition" in field.lower() or "variant" in field.lower()
        ]
        if condition_fields:
            raise ValueError("Prompt-condition metadata must not be included in pair review CSV.")
        safe_metadata = {
            field: pair[field]
            for field in (
                "experiment_id",
                "paper_id",
                "repeat_index",
                "output_id_a",
                "output_id_b",
                "lexical_similarity",
                "candidate_flag",
            )
            if field in pair
        }
        pairs.append(
            {
                "question": source_a["question"],
                "evidence": json.loads(pair["evidence"]),
                "response_a": pair["response_a"],
                "response_b": pair["response_b"],
                "metadata": safe_metadata,
                "human_contradiction": pair[PAIR_FIELD],
                "notes": pair.get("notes", ""),
            }
        )

    workflow_path = root / REVIEW_MANIFEST
    if workflow_path.exists():
        workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
        if workflow.get("experiment_id") != manifest.get("experiment_id"):
            raise ValueError("Human-review manifest experiment ID does not match the run.")
        if workflow.get("raw_outputs_sha256") != raw_hash:
            raise ValueError("Raw outputs changed since the review workflow was initialized.")
    else:
        workflow = {
            "workflow_version": WORKFLOW_VERSION,
            "experiment_id": manifest["experiment_id"],
            "raw_outputs_sha256": raw_hash,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }

    return {
        "results_dir": root,
        "experiment_id": manifest["experiment_id"],
        "raw_outputs_sha256": raw_hash,
        "answer_csv_sha256": sha256_file(annotation_path),
        "pair_csv_sha256": sha256_file(pair_path),
        "answer_headers": answer_headers,
        "answer_records": answer_records,
        "answers": answers,
        "pair_headers": pair_headers,
        "pair_records": pair_records,
        "pairs": pairs,
        "workflow_manifest": workflow,
    }


def _atomic_write_csv(path: Path, headers: list[str], records: list[dict[str, str]]) -> None:
    handle, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def _verify_write_preconditions(
    results_dir: Path,
    expected_raw_hash: str,
    csv_path: Path,
    expected_csv_hash: str,
) -> None:
    if sha256_file(results_dir / RAW_FILE) != expected_raw_hash:
        raise RuntimeError("Raw model output hash changed; refusing to save annotations.")
    if sha256_file(csv_path) != expected_csv_hash:
        raise RuntimeError(
            f"{csv_path.name} changed since it was loaded. Reload before saving to avoid overwriting another reviewer."
        )


def _write_workflow_manifest(results_dir: Path, workflow: dict[str, Any]) -> None:
    workflow["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    path = results_dir / REVIEW_MANIFEST
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(workflow, indent=2) + "\n", encoding="utf-8")
    os.replace(temp_path, path)


def initialize_workflow_manifest(bundle: dict[str, Any]) -> None:
    """Persist review workflow provenance separately from experiment provenance."""
    path = bundle["results_dir"] / REVIEW_MANIFEST
    if not path.exists():
        _write_workflow_manifest(bundle["results_dir"], bundle["workflow_manifest"])


def save_answer_annotation(
    bundle: dict[str, Any], output_id: str, values: dict[str, str], notes: str
) -> None:
    for field, allowed in ANSWER_RATINGS.items():
        if values.get(field, "") not in allowed:
            raise ValueError(f"Invalid {field} value.")
    root = bundle["results_dir"]
    path = root / ANNOTATION_FILE
    _verify_write_preconditions(
        root, bundle["raw_outputs_sha256"], path, bundle["answer_csv_sha256"]
    )
    headers = bundle["answer_headers"]
    records = bundle["answer_records"]
    matching = [row for row in records if row["output_id"] == output_id]
    if len(matching) != 1:
        raise ValueError(f"Expected exactly one answer row for {output_id}.")
    matching[0].update(values)
    matching[0]["notes"] = notes
    _atomic_write_csv(path, headers, records)
    _write_workflow_manifest(root, bundle["workflow_manifest"])


def save_pair_annotation(bundle: dict[str, Any], pair_index: int, value: str, notes: str) -> None:
    if value not in PAIR_RATINGS:
        raise ValueError("human_contradiction must be not_reviewed, yes, or no.")
    root = bundle["results_dir"]
    path = root / PAIR_FILE
    _verify_write_preconditions(root, bundle["raw_outputs_sha256"], path, bundle["pair_csv_sha256"])
    headers, records = _read_csv(path)
    if not 0 <= pair_index < len(records):
        raise IndexError("Contradiction-pair row is out of range.")
    records[pair_index][PAIR_FIELD] = value
    records[pair_index]["notes"] = notes
    _atomic_write_csv(path, headers, records)
    _write_workflow_manifest(root, bundle["workflow_manifest"])


def export_csv(headers: list[str], records: list[dict[str, str]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=headers, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(records)
    return output.getvalue().encode("utf-8")
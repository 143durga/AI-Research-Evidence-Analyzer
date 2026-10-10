"""Run and analyze controlled prompt-variation experiments."""

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

from research_analyzer.answering import (
    SYSTEM_PROMPT,
    AnswerGenerator,
    get_environment_metadata,
)
from research_analyzer.vector_store import Evidence

COMMON_POLICY = (
    SYSTEM_PROMPT
    + "\nUse the supplied evidence labels exactly as written; do not invent labels."
)


class PromptVariant:
    def __init__(self, variant_id: str, name: str, instruction: str) -> None:
        self.variant_id = variant_id
        self.name = name
        self.instruction = instruction

    @property
    def system_prompt(self) -> str:
        return f"{COMMON_POLICY}\n\n{self.instruction}"


PROMPT_VARIANTS = (
    PromptVariant(
        "baseline",
        "Direct evidence-only answer",
        "Answer directly and concisely. Use citations for factual statements.",
    ),
    PromptVariant(
        "citation_focus",
        "Citation-focused answer",
        "Attach the relevant evidence label to every factual claim. If a claim has no direct support, omit it.",
    ),
    PromptVariant(
        "uncertainty_focus",
        "Uncertainty-calibrated answer",
        "Distinguish what the evidence establishes from what remains uncertain. Prefer an explicit insufficient-evidence answer over a guess.",
    ),
    PromptVariant(
        "structured",
        "Structured answer",
        "Respond under the headings Answer, Evidence, and Limitations. Keep each section concise and cite factual statements.",
    ),
)

ANNOTATIONS_FILE = "human_annotations.csv"
CONTRADICTIONS_FILE = "contradiction_pairs.csv"
RAW_OUTPUTS_FILE = "raw_outputs.jsonl"
METRICS_FILE = "metrics.csv"


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load validated JSONL records with a question and fixed evidence passages."""
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with Path(path).open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                question_id = record["question_id"]
                question = record["question"]
                evidence = record["evidence"]
                paper_id = record["paper_id"]
                answerability = record["answerability"]
                paper_title = record["paper_title"]
                authors = record["authors"]
                year = record["year"]
                source = record["source"]
                source_url = record["source_url"]
                question_type = record["question_type"]
                document_sha256 = record["document_sha256"]
                answerability_reason = record["answerability_reason"]
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"Invalid dataset record on line {line_number}: {exc}"
                ) from exc

            if not all(
                isinstance(value, str) and value.strip()
                for value in (question_id, question, paper_id)
            ):
                raise ValueError(
                    f"Line {line_number} needs non-empty question_id, paper_id, and question strings."
                )
            if question_id in seen_ids:
                raise ValueError(f"Duplicate question_id {question_id!r} on line {line_number}.")
            if not isinstance(answerability, str) or answerability not in {
                "answerable",
                "unanswerable",
            }:
                raise ValueError(
                    f"Line {line_number} answerability must be answerable or unanswerable."
                )
            if (
                not isinstance(paper_title, str)
                or not paper_title.strip()
                or not isinstance(authors, list)
                or not authors
                or not all(isinstance(author, str) and author.strip() for author in authors)
                or not isinstance(year, int)
                or isinstance(year, bool)
                or not isinstance(source, str)
                or not source.strip()
                or not isinstance(source_url, str)
                or not source_url.startswith("https://")
                or not isinstance(question_type, str)
                or question_type not in {"direct_factual", "synthesis", "cautious_qualified", "unanswerable"}
                or not isinstance(document_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", document_sha256)
                or not isinstance(answerability_reason, str)
                or not answerability_reason.strip()
            ):
                raise ValueError(f"Line {line_number} has invalid or missing paper/question provenance fields.")
            if not isinstance(evidence, list) or not evidence:
                raise ValueError(
                    f"Line {line_number} must contain at least one fixed evidence passage."
                )
            evidence_ids: set[str] = set()
            for item in evidence:
                if not isinstance(item, dict):
                    raise ValueError(f"Evidence on line {line_number} must be an object.")
                evidence_id = item.get("evidence_id")
                text = item.get("text")
                page = item.get("page")
                reason = item.get("reason")
                if (
                    not isinstance(evidence_id, str)
                    or not evidence_id.strip()
                    or not isinstance(text, str)
                    or not text.strip()
                    or not isinstance(page, int)
                    or isinstance(page, bool)
                    or page < 1
                    or not isinstance(reason, str)
                    or not reason.strip()
                ):
                    raise ValueError(
                        f"Each evidence item on line {line_number} needs evidence_id, non-empty text, page >= 1, and a relevance reason."
                    )
                if evidence_id in evidence_ids:
                    raise ValueError(
                        f"Duplicate evidence_id {evidence_id!r} on line {line_number}."
                    )
                for field in (
                    "paper_title",
                    "authors",
                    "year",
                    "source",
                    "source_url",
                    "doi",
                    "document_sha256",
                ):
                    if item.get(field) != record.get(field, ""):
                        raise ValueError(
                            f"Evidence {evidence_id!r} on line {line_number} has mismatched {field} provenance."
                        )
                evidence_ids.add(evidence_id)
            seen_ids.add(question_id)
            records.append(record)

    if not records:
        raise ValueError("The dataset contains no question records.")
    return records


def _build_user_prompt(
    paper_id: str, question: str, evidence: list[dict[str, Any]]
) -> str:
    passages = "\n\n".join(
        f"[E{index}] Source: {paper_id}, page {item['page']}\n{item['text']}"
        for index, item in enumerate(evidence, start=1)
    )
    return f"Question:\n{question}\n\nEvidence:\n{passages}"


def run_experiment(
    dataset_path: str | Path,
    output_dir: str | Path,
    model: str,
    base_url: str,
    api_key: str | None,
    repeats: int = 3,
    temperature: float = 0,
    seed: int = 17,
    generator_factory: Callable[..., Any] = AnswerGenerator,
    experiment_id: str | None = None,
) -> Path:
    """Run every dataset question under every prompt, keeping evidence fixed."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    backend = os.getenv("LLM_BACKEND", "llama_cpp")
    if backend != "llama_cpp":
        raise ValueError(
            "Research experiments support only local llama_cpp inference. "
            "Set LLM_BACKEND=llama_cpp."
        )

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    records = load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    experiment_id = experiment_id or datetime.now(timezone.utc).strftime("exp1-%Y%m%dT%H%M%SZ")
    local_model_path = os.getenv(
        "LOCAL_MODEL_PATH", "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
    )
    local_settings = {
        "temperature": temperature,
        "max_tokens": int(os.getenv("LOCAL_MAX_TOKENS", "256")),
        "top_p": float(os.getenv("LOCAL_TOP_P", "1.0")),
        "seed": int(os.getenv("LOCAL_SEED", "17")),
        "context_size": int(os.getenv("LOCAL_CONTEXT_SIZE", "4096")),
        "cpu_threads": int(os.getenv("LOCAL_THREADS", "2")),
        "gpu_layers": 0,
    }
    generation_settings = local_settings
    hardware_environment = get_environment_metadata(local_model_path)
    manifest = {
        "experiment_id": experiment_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "dataset_record_count": len(records),
        "model": model,
        "model_revision": os.getenv("LOCAL_MODEL_REVISION") or os.getenv("LLM_MODEL_REVISION"),
        "inference_backend": backend,
        "hardware_environment": hardware_environment,
        "base_url": None,
        "temperature": temperature,
        "generation_settings": generation_settings,
        "repeats": repeats,
        "expected_model_calls": len(records) * len(PROMPT_VARIANTS) * repeats,
        "shuffle_seed": seed,
        "prompt_variants": [
            {
                "variant_id": variant.variant_id,
                "name": variant.name,
                "instruction": variant.instruction,
                "system_prompt": variant.system_prompt,
            }
            for variant in PROMPT_VARIANTS
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    tasks = [
        (record, variant, repeat_index)
        for record in records
        for repeat_index in range(1, repeats + 1)
        for variant in PROMPT_VARIANTS
    ]
    random.Random(seed).shuffle(tasks)
    raw_outputs: list[dict[str, Any]] = []
    with (output_dir / RAW_OUTPUTS_FILE).open("w", encoding="utf-8") as raw_file:
        for call_index, (record, variant, repeat_index) in enumerate(tasks, start=1):
            output_id = f"{experiment_id}__answer_{call_index:04d}"
            system_prompt = variant.system_prompt
            user_prompt = _build_user_prompt(
                record["paper_id"], record["question"], record["evidence"]
            )
            result: dict[str, Any] = {
                "output_id": output_id,
                "question_id": record["question_id"],
                "paper_id": record["paper_id"],
                "paper_title": record["paper_title"],
                "authors": record["authors"],
                "year": record["year"],
                "source": record["source"],
                "source_url": record["source_url"],
                "doi": record.get("doi", ""),
                "document_sha256": record["document_sha256"],
                "question": record["question"],
                "question_type": record["question_type"],
                "answerability": record["answerability"],
                "answerability_reason": record["answerability_reason"],
                "variant_id": variant.variant_id,
                "prompt_condition": variant.variant_id,
                "repeat_index": repeat_index,
                "repetition": repeat_index,
                "experiment_id": experiment_id,
                "model": model,
                "temperature": temperature,
                "request_settings": generation_settings,
                "model_name": model,
                "model_revision": os.getenv("LOCAL_MODEL_REVISION") or os.getenv("LLM_MODEL_REVISION"),
                "inference_backend": backend,
                "hardware_environment": hardware_environment,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "request_messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "evidence": record["evidence"],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            started = time.monotonic()
            try:
                generator = generator_factory(
                    model=model,
                    base_url=base_url,
                    api_key=api_key,
                    system_prompt=system_prompt,
                    temperature=temperature,
                )
                if hasattr(generator, "experiment_metadata"):
                    metadata = generator.experiment_metadata()
                    result["model_name"] = metadata["model_name"]
                    result["model_revision"] = metadata["model_revision"]
                    result["inference_backend"] = metadata["inference_backend"]
                    result["hardware_environment"] = metadata["hardware_environment"]
                    result["request_settings"] = metadata["generation_settings"]
                evidence_objects = [
                    Evidence(
                        chunk_id=item["evidence_id"],
                        text=item["text"],
                        filename=record["paper_id"],
                        page_number=item["page"],
                        distance=0.0,
                    )
                    for item in record["evidence"]
                ]
                raw_response = generator.generate(
                    record["question"], evidence_objects
                )
                result["raw_model_response"] = raw_response
                result["answer"] = raw_response
                result["status"] = "success"
            except Exception as exc:
                result["timestamp_completed"] = datetime.now(timezone.utc).isoformat()
                result["answer"] = ""
                result["raw_model_response"] = ""
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
            else:
                result["timestamp_completed"] = datetime.now(timezone.utc).isoformat()
            result["duration_seconds"] = round(time.monotonic() - started, 3)
            raw_outputs.append(result)
            raw_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            raw_file.flush()

    _write_annotation_templates(output_dir, raw_outputs)
    _write_contradiction_pairs(output_dir, raw_outputs)
    analyze_results(output_dir)
    return output_dir


def _write_annotation_templates(
    output_dir: Path, raw_outputs: list[dict[str, Any]]
) -> None:
    with (output_dir / ANNOTATIONS_FILE).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "output_id",
                "experiment_id",
                "paper_id",
                "paper_title",
                "question",
                "question_id",
                "answerability",
                "model_answer",
                "evidence",
                "evidence_support",
                "answer_correctness",
                "appropriate_uncertainty",
                "contradiction_with_evidence",
                "notes",
            ],
        )
        writer.writeheader()
        for item in raw_outputs:
            writer.writerow(
                {
                    "output_id": item["output_id"],
                    "experiment_id": item["experiment_id"],
                    "paper_id": item["paper_id"],
                    "paper_title": item["paper_title"],
                    "question": item["question"],
                    "question_id": item["question_id"],
                    "answerability": item["answerability"],
                    "model_answer": item["raw_model_response"],
                    "evidence": json.dumps(item["evidence"], ensure_ascii=False),
                    "evidence_support": "",
                    "answer_correctness": "",
                    "appropriate_uncertainty": "",
                    "contradiction_with_evidence": "",
                    "notes": "",
                }
            )


def _tokens(text: str) -> set[str]:
    text = re.sub(r"\[E\d+\]", " ", text, flags=re.IGNORECASE)
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _jaccard(left: str, right: str) -> float | None:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return None
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _contradiction_candidate(left: str, right: str) -> bool:
    negative = {"not", "no", "never", "neither", "without", "failed", "lack", "lacks"}
    opposite_terms = {
        ("increase", "decrease"),
        ("increased", "decreased"),
        ("higher", "lower"),
        ("improved", "worsened"),
        ("positive", "negative"),
        ("effective", "ineffective"),
        ("associated", "unassociated"),
    }
    left_sentences = re.split(r"(?<=[.!?])\s+", left)
    right_sentences = re.split(r"(?<=[.!?])\s+", right)
    for left_sentence in left_sentences:
        left_tokens = _tokens(left_sentence)
        if not left_tokens:
            continue
        for right_sentence in right_sentences:
            right_tokens = _tokens(right_sentence)
            similarity = _jaccard(left_sentence, right_sentence)
            if similarity is None or similarity < 0.35:
                continue
            opposite = any(
                (first in left_tokens and second in right_tokens)
                or (second in left_tokens and first in right_tokens)
                for first, second in opposite_terms
            )
            if bool(left_tokens & negative) != bool(right_tokens & negative) or opposite:
                return True
    return False


def _write_contradiction_pairs(
    output_dir: Path, raw_outputs: list[dict[str, Any]]
) -> None:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in raw_outputs:
        if item["status"] == "success":
            grouped[(item["question_id"], item["repeat_index"])].append(item)
    with (output_dir / CONTRADICTIONS_FILE).open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = [
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
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for (question_id, repeat_index), items in grouped.items():
            for first, second in itertools.combinations(items, 2):
                similarity = _jaccard(first["answer"], second["answer"])
                writer.writerow(
                    {
                        "experiment_id": first["experiment_id"],
                        "question_id": question_id,
                        "paper_id": first["paper_id"],
                        "repeat_index": repeat_index,
                        "output_id_a": first["output_id"],
                        "output_id_b": second["output_id"],
                        "response_a": first["raw_model_response"],
                        "response_b": second["raw_model_response"],
                        "evidence": json.dumps(first["evidence"], ensure_ascii=False),
                        "lexical_similarity": (
                            f"{similarity:.4f}" if similarity is not None else ""
                        ),
                        "candidate_flag": str(
                            _contradiction_candidate(first["answer"], second["answer"])
                        ).lower(),
                        "human_contradiction": "not_reviewed",
                        "notes": "",
                    }
                )


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def analyze_results(output_dir: str | Path) -> list[dict[str, Any]]:
    """Regenerate proxy and human-annotation metrics by condition and answerability."""
    output_dir = Path(output_dir)
    with (output_dir / RAW_OUTPUTS_FILE).open(encoding="utf-8") as file:
        raw_outputs = [json.loads(line) for line in file if line.strip()]
    annotations = {
        item["output_id"]: item for item in _read_csv(output_dir / ANNOTATIONS_FILE)
    }
    contradiction_rows = _read_csv(output_dir / CONTRADICTIONS_FILE)
    output_by_id = {item["output_id"]: item for item in raw_outputs}
    repeat_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    question_repeat_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    condition_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in raw_outputs:
        condition_groups[(item["variant_id"], item["answerability"])].append(item)
        if item["status"] == "success":
            repeat_groups[(item["question_id"], item["variant_id"])].append(item)
            question_repeat_groups[(item["question_id"], item["repeat_index"])].append(item)

    within_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (_, variant_id), items in repeat_groups.items():
        for first, second in itertools.combinations(items, 2):
            score = _jaccard(first["answer"], second["answer"])
            if score is not None:
                within_scores[(variant_id, first["answerability"])].append(score)

    between_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    for items in question_repeat_groups.values():
        for first, second in itertools.combinations(items, 2):
            if first["variant_id"] == second["variant_id"]:
                continue
            score = _jaccard(first["answer"], second["answer"])
            if score is not None:
                between_scores[(first["variant_id"], first["answerability"])].append(score)
                between_scores[(second["variant_id"], second["answerability"])].append(score)

    metrics: list[dict[str, Any]] = []
    for variant in PROMPT_VARIANTS:
        for answerability in ("answerable", "unanswerable"):
            items = condition_groups[(variant.variant_id, answerability)]
            successful = [item for item in items if item["status"] == "success"]
            citation_count = valid_citations = citation_present_answers = valid_citation_answers = 0
            support: list[str] = []
            correctness: list[str] = []
            uncertainty: list[str] = []
            uncertainty_not_applicable_count = 0
            contradictions: list[str] = []
            for item in successful:
                labels = re.findall(r"\[(E\d+)\]", item["raw_model_response"], re.IGNORECASE)
                valid = {f"E{index}" for index in range(1, len(item["evidence"]) + 1)}
                normalized = [label.upper() for label in labels]
                citation_count += len(normalized)
                valid_citations += sum(label in valid for label in normalized)
                citation_present_answers += bool(normalized)
                valid_citation_answers += bool(normalized) and all(label in valid for label in normalized)
                annotation = annotations.get(item["output_id"], {})
                for column, allowed, target in (
                    ("evidence_support", {"yes", "partial", "no"}, support),
                    ("answer_correctness", {"yes", "partial", "no"}, correctness),
                    ("appropriate_uncertainty", {"yes", "no"}, uncertainty),
                    ("contradiction_with_evidence", {"yes", "no"}, contradictions),
                ):
                    value = annotation.get(column, "").strip().lower()
                    if value in allowed:
                        target.append(value)
                    elif column == "appropriate_uncertainty" and value == "not applicable":
                        uncertainty_not_applicable_count += 1

            question_ids = {item["question_id"] for item in items}
            variant_pairs = []
            for row in contradiction_rows:
                first = output_by_id.get(row["output_id_a"])
                second = output_by_id.get(row["output_id_b"])
                if first is None or second is None:
                    continue
                if variant.variant_id in {first["variant_id"], second["variant_id"]}:
                    variant_pairs.append(row)
            relevant_pairs = [
                row
                for row in variant_pairs
                if row.get("question_id") in question_ids
                and output_by_id[row["output_id_a"]]["answerability"] == answerability
            ]
            reviewed_pairs = [row for row in relevant_pairs if row.get("human_contradiction", "").strip().lower() in {"yes", "no"}]
            metrics.append(
                {
                    "prompt_condition": variant.variant_id,
                    "prompt_name": variant.name,
                    "answerability": answerability,
                    "question_count": len({item["question_id"] for item in items}),
                    "attempt_count": len(items),
                    "success_rate": len(successful) / len(items) if items else None,
                    "within_condition_lexical_consistency_jaccard": _mean_or_none(
                        within_scores[(variant.variant_id, answerability)]
                    ),
                    "between_condition_lexical_consistency_jaccard": _mean_or_none(
                        between_scores[(variant.variant_id, answerability)]
                    ),
                    "citation_presence_rate": citation_present_answers / len(successful) if successful else None,
                    "valid_citation_answer_rate": valid_citation_answers / len(successful) if successful else None,
                    "citation_evidence_label_validity": valid_citations / citation_count if citation_count else None,
                    "human_evidence_support_yes_rate": _rating_rate(support, "yes"),
                    "human_evidence_support_partial_rate": _rating_rate(support, "partial"),
                    "human_evidence_support_no_rate": _rating_rate(support, "no"),
                    "human_evidence_support_rated_count": len(support),
                    "human_support_annotated_count": len(support),
                    "human_correctness_yes_rate": _rating_rate(correctness, "yes"),
                    "human_correctness_partial_rate": _rating_rate(correctness, "partial"),
                    "human_correctness_no_rate": _rating_rate(correctness, "no"),
                    "human_correctness_annotated_count": len(correctness),
                    "appropriate_uncertainty_yes_rate": _rating_rate(uncertainty, "yes"),
                    "appropriate_uncertainty_no_rate": _rating_rate(uncertainty, "no"),
                    "appropriate_uncertainty_annotated_count": len(uncertainty),
                    "appropriate_uncertainty_not_applicable_count": uncertainty_not_applicable_count,
                    "contradiction_with_evidence_yes_rate": _rating_rate(contradictions, "yes"),
                    "contradiction_with_evidence_no_rate": _rating_rate(contradictions, "no"),
                    "human_contradiction_annotated_count": len(contradictions),
                    "contradiction_pair_candidate_count": sum(
                        row.get("candidate_flag", "false").lower() == "true" for row in relevant_pairs
                    ),
                    "contradiction_pair_reviewed_count": len(reviewed_pairs),
                    "human_confirmed_answer_pair_contradiction_rate": (
                        sum(row["human_contradiction"].strip().lower() == "yes" for row in reviewed_pairs)
                        / len(reviewed_pairs) if reviewed_pairs else None
                    ),
                }
            )

    _write_metrics(output_dir / METRICS_FILE, metrics)
    _write_metrics_plot(output_dir / "metrics.png", metrics)
    return metrics


def _mean_or_none(values: list[float]) -> float | None:
    return statistics.mean(values) if values else None


def _rating_rate(ratings: list[str], category: str) -> float | None:
    return ratings.count(category) / len(ratings) if ratings else None


def _write_metrics(path: Path, metrics: list[dict[str, Any]]) -> None:
    fields = list(metrics[0]) if metrics else []
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(metrics)


def _write_metrics_plot(path: Path, metrics: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    conditions = [variant.variant_id for variant in PROMPT_VARIANTS]
    answerabilities = ("answerable", "unanswerable")
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    charts = (
        ("within_condition_lexical_consistency_jaccard", "Within-condition consistency (Jaccard)"),
        ("citation_presence_rate", "Citation presence (proxy)"),
        ("human_evidence_support_yes_rate", "Human evidence support: Yes"),
    )
    width = 0.36
    positions = list(range(len(conditions)))
    for axis, (metric_name, title) in zip(axes, charts, strict=True):
        for offset, answerability in enumerate(answerabilities):
            rows = {
                item["prompt_condition"]: item
                for item in metrics if item["answerability"] == answerability
            }
            values = [rows.get(condition, {}).get(metric_name) for condition in conditions]
            axis.bar(
                [position + (offset - 0.5) * width for position in positions],
                [value if value is not None else 0 for value in values],
                width,
                label=answerability,
                alpha=0.85 if all(value is not None for value in values) else 0.75,
            )
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.set_xticks(positions, conditions, rotation=30, ha="right")
        axis.set_ylabel("Mean similarity / rate")
        axis.legend()
        if metric_name == "human_evidence_support_yes_rate" and all(
            item[metric_name] is None for item in metrics
        ):
            axis.text(0.5, 0.5, "Human annotation pending", ha="center", va="center", transform=axis.transAxes)
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run or analyze prompt variation studies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run all prompts over a JSONL dataset.")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument(
        "--model",
        default=os.getenv("LLM_MODEL", "Qwen2.5-1.5B-Instruct-Q4_K_M"),
    )
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--temperature", type=float, default=0)
    run_parser.add_argument("--seed", type=int, default=17)
    run_parser.add_argument("--experiment-id")
    analyze_parser = subparsers.add_parser(
        "analyze", help="Recalculate metrics after human annotations are entered."
    )
    analyze_parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    if args.command == "run":
        question_count = len(load_dataset(args.dataset))
        expected_calls = question_count * len(PROMPT_VARIANTS) * args.repeats
        print(
            f"Expected model calls: {expected_calls} "
            f"({question_count} questions x {len(PROMPT_VARIANTS)} prompt conditions x {args.repeats} repetitions)."
        )
        backend = os.getenv("LLM_BACKEND", "llama_cpp")
        local_model_path = Path(
            os.getenv(
                "LOCAL_MODEL_PATH",
                "models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf",
            )
        ).expanduser()
        if backend == "llama_cpp" and not local_model_path.is_file():
            raise SystemExit(
                f"LLM_BACKEND=llama_cpp but the configured model file is missing: {local_model_path}. "
                "No model calls or result files were created."
            )
        if backend != "llama_cpp":
            raise SystemExit(
                "Research experiments support only local llama_cpp inference. "
                "No model calls or result files were created."
            )
        output = run_experiment(
            dataset_path=args.dataset,
            output_dir=args.output,
            model=args.model,
            base_url="",
            api_key=None,
            repeats=args.repeats,
            temperature=args.temperature,
            seed=args.seed,
            experiment_id=args.experiment_id,
        )
        print(f"Experiment files saved to {output}")
    else:
        analyze_results(args.results_dir)
        print(f"Updated metrics in {Path(args.results_dir) / METRICS_FILE}")


if __name__ == "__main__":
    main()
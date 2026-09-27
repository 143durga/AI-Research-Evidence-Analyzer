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

from research_analyzer.answering import SYSTEM_PROMPT, AnswerGenerator
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
                "ambiguous",
            }:
                raise ValueError(
                    f"Line {line_number} answerability must be answerable, unanswerable, or ambiguous."
                )
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
                if (
                    not isinstance(evidence_id, str)
                    or not evidence_id.strip()
                    or not isinstance(text, str)
                    or not text.strip()
                    or not isinstance(page, int)
                    or isinstance(page, bool)
                    or page < 1
                ):
                    raise ValueError(
                        f"Each evidence item on line {line_number} needs evidence_id, non-empty text, and page >= 1."
                    )
                if evidence_id in evidence_ids:
                    raise ValueError(
                        f"Duplicate evidence_id {evidence_id!r} on line {line_number}."
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
) -> Path:
    """Run every dataset question under every prompt, keeping evidence fixed."""
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    if temperature < 0:
        raise ValueError("temperature cannot be negative")

    dataset_path = Path(dataset_path)
    output_dir = Path(output_dir)
    records = load_dataset(dataset_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    dataset_hash = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_sha256": dataset_hash,
        "dataset_record_count": len(records),
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "repeats": repeats,
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
        for record, variant, repeat_index in tasks:
            output_id = (
                f"{record['question_id']}__{variant.variant_id}__r{repeat_index:03d}"
            )
            system_prompt = variant.system_prompt
            user_prompt = _build_user_prompt(
                record["paper_id"], record["question"], record["evidence"]
            )
            result: dict[str, Any] = {
                "output_id": output_id,
                "question_id": record["question_id"],
                "paper_id": record["paper_id"],
                "answerability": record["answerability"],
                "variant_id": variant.variant_id,
                "repeat_index": repeat_index,
                "model": model,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "evidence": record["evidence"],
                "started_at_utc": datetime.now(timezone.utc).isoformat(),
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
                result["answer"] = generator.generate(
                    record["question"], evidence_objects
                )
                result["status"] = "success"
            except Exception as exc:
                result["answer"] = ""
                result["status"] = "error"
                result["error"] = f"{type(exc).__name__}: {exc}"
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
                "question_id",
                "variant_id",
                "repeat_index",
                "human_support_rating",
                "human_correctness_rating",
                "notes",
            ],
        )
        writer.writeheader()
        for item in raw_outputs:
            writer.writerow(
                {
                    "output_id": item["output_id"],
                    "question_id": item["question_id"],
                    "variant_id": item["variant_id"],
                    "repeat_index": item["repeat_index"],
                    "human_support_rating": "",
                    "human_correctness_rating": "",
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
            "question_id",
            "repeat_index",
            "output_id_a",
            "output_id_b",
            "variant_a",
            "variant_b",
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
                        "question_id": question_id,
                        "repeat_index": repeat_index,
                        "output_id_a": first["output_id"],
                        "output_id_b": second["output_id"],
                        "variant_a": first["variant_id"],
                        "variant_b": second["variant_id"],
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
    """Regenerate metrics and plot, incorporating any completed human ratings."""
    output_dir = Path(output_dir)
    with (output_dir / RAW_OUTPUTS_FILE).open(encoding="utf-8") as file:
        raw_outputs = [json.loads(line) for line in file if line.strip()]
    annotations = {
        item["output_id"]: item for item in _read_csv(output_dir / ANNOTATIONS_FILE)
    }
    contradiction_rows = _read_csv(output_dir / CONTRADICTIONS_FILE)
    outputs_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw_outputs:
        outputs_by_variant[item["variant_id"]].append(item)

    within_scores: dict[str, list[float]] = defaultdict(list)
    repeat_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    question_repeat_groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for item in raw_outputs:
        if item["status"] != "success":
            continue
        repeat_groups[(item["question_id"], item["variant_id"])].append(item)
        question_repeat_groups[(item["question_id"], item["repeat_index"])].append(item)
    for (_, variant_id), items in repeat_groups.items():
        for first, second in itertools.combinations(items, 2):
            score = _jaccard(first["answer"], second["answer"])
            if score is not None:
                within_scores[variant_id].append(score)
    between_scores: dict[str, list[float]] = defaultdict(list)
    for items in question_repeat_groups.values():
        for first, second in itertools.combinations(items, 2):
            if first["variant_id"] == second["variant_id"]:
                continue
            score = _jaccard(first["answer"], second["answer"])
            if score is not None:
                between_scores[first["variant_id"]].append(score)
                between_scores[second["variant_id"]].append(score)

    metrics: list[dict[str, Any]] = []
    for variant in PROMPT_VARIANTS:
        items = outputs_by_variant[variant.variant_id]
        successful = [item for item in items if item["status"] == "success"]
        citation_counts = [
            re.findall(r"\[(E\d+)\]", item["answer"], flags=re.IGNORECASE)
            for item in successful
        ]
        valid_count = 0
        citation_count = 0
        cited_answers = 0
        reviewed_support: list[str] = []
        reviewed_correctness: list[str] = []
        for item, labels in zip(successful, citation_counts, strict=True):
            valid = {f"E{index}" for index in range(1, len(item["evidence"]) + 1)}
            normalized = [label.upper() for label in labels]
            citation_count += len(normalized)
            valid_count += sum(label in valid for label in normalized)
            cited_answers += bool(normalized) and all(label in valid for label in normalized)
            annotation = annotations.get(item["output_id"], {})
            rating = annotation.get("human_support_rating", "").strip().lower()
            if rating in {"fully_supported", "partially_supported", "unsupported"}:
                reviewed_support.append(rating)
            correctness = annotation.get("human_correctness_rating", "").strip().lower()
            if correctness in {"correct", "partially_correct", "incorrect"}:
                reviewed_correctness.append(correctness)

        variant_pairs = [
            row
            for row in contradiction_rows
            if row["variant_a"] == variant.variant_id
            or row["variant_b"] == variant.variant_id
        ]
        reviewed_pairs = [
            row
            for row in variant_pairs
            if row.get("human_contradiction", "").strip().lower() in {"yes", "no"}
        ]
        confirmed_pairs = sum(
            row["human_contradiction"].strip().lower() == "yes" for row in reviewed_pairs
        )
        metrics.append(
            {
                "variant_id": variant.variant_id,
                "variant_name": variant.name,
                "run_count": len(items),
                "success_rate": len(successful) / len(items) if items else None,
                "within_prompt_repeat_jaccard_mean": _mean_or_none(
                    within_scores[variant.variant_id]
                ),
                "between_prompt_jaccard_mean": _mean_or_none(
                    between_scores[variant.variant_id]
                ),
                "valid_citation_answer_rate": (
                    cited_answers / len(successful) if successful else None
                ),
                "answerable_question_count": sum(
                    item["answerability"] == "answerable" for item in items
                ),
                "unanswerable_question_count": sum(
                    item["answerability"] == "unanswerable" for item in items
                ),
                "ambiguous_question_count": sum(
                    item["answerability"] == "ambiguous" for item in items
                ),
                "citation_label_validity_rate": (
                    valid_count / citation_count if citation_count else None
                ),
                "human_support_reviewed_count": len(reviewed_support),
                "human_fully_supported_rate": _rating_rate(
                    reviewed_support, "fully_supported"
                ),
                "human_unsupported_rate": _rating_rate(reviewed_support, "unsupported"),
                "human_correctness_reviewed_count": len(reviewed_correctness),
                "human_correct_rate": _rating_rate(reviewed_correctness, "correct"),
                "human_incorrect_rate": _rating_rate(reviewed_correctness, "incorrect"),
                "contradiction_candidate_count": sum(
                    row.get("candidate_flag", "false").lower() == "true"
                    for row in variant_pairs
                ),
                "contradiction_pairs_reviewed": len(reviewed_pairs),
                "human_confirmed_contradiction_rate": (
                    confirmed_pairs / len(reviewed_pairs) if reviewed_pairs else None
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

    names = [item["variant_id"] for item in metrics]
    consistency = [item["within_prompt_repeat_jaccard_mean"] for item in metrics]
    citation = [item["valid_citation_answer_rate"] for item in metrics]
    support = [item["human_fully_supported_rate"] for item in metrics]
    figure, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, values, title in zip(
        axes,
        (consistency, citation, support),
        ("Repeat consistency (Jaccard)", "Valid citation answers", "Human: fully supported"),
        strict=True,
    ):
        axis.bar(names, [value if value is not None else 0 for value in values])
        axis.set_ylim(0, 1)
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=35)
        axis.set_ylabel("Rate / mean similarity")
        if all(value is None for value in values):
            axis.text(
                0.5,
                0.5,
                "Not available",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
    figure.savefig(path, dpi=150)
    plt.close(figure)


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="Run or analyze prompt variation studies.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run", help="Run all prompts over a JSONL dataset.")
    run_parser.add_argument("--dataset", required=True)
    run_parser.add_argument("--output", required=True)
    run_parser.add_argument("--model", default=os.getenv("LLM_MODEL", "gpt-4o-mini"))
    run_parser.add_argument(
        "--base-url", default=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    )
    run_parser.add_argument("--repeats", type=int, default=3)
    run_parser.add_argument("--temperature", type=float, default=0)
    run_parser.add_argument("--seed", type=int, default=17)
    analyze_parser = subparsers.add_parser(
        "analyze", help="Recalculate metrics after human annotations are entered."
    )
    analyze_parser.add_argument("--results-dir", required=True)
    args = parser.parse_args()

    if args.command == "run":
        output = run_experiment(
            dataset_path=args.dataset,
            output_dir=args.output,
            model=args.model,
            base_url=args.base_url,
            api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
            repeats=args.repeats,
            temperature=args.temperature,
            seed=args.seed,
        )
        print(f"Experiment files saved to {output}")
    else:
        analyze_results(args.results_dir)
        print(f"Updated metrics in {Path(args.results_dir) / METRICS_FILE}")


if __name__ == "__main__":
    main()
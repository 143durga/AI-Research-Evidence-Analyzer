"""Run and preserve one real local-model inference before the full pilot."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from research_analyzer.answering import AnswerGenerator
from research_analyzer.experiments import PROMPT_VARIANTS, _build_user_prompt, load_dataset
from research_analyzer.vector_store import Evidence


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/pilot_dataset.jsonl")
    parser.add_argument("--output", default="results/experiment_1_local_smoke.json")
    args = parser.parse_args()
    if os.getenv("LLM_BACKEND", "llama_cpp") != "llama_cpp":
        raise SystemExit("The local-model smoke test requires LLM_BACKEND=llama_cpp.")

    record = load_dataset(args.dataset)[0]
    variant = PROMPT_VARIANTS[0]
    model_name = os.getenv(
        "LLM_MODEL", "Qwen2.5-1.5B-Instruct-Q4_K_M"
    )
    generator = AnswerGenerator(
        model=model_name,
        base_url="",
        api_key=None,
        system_prompt=variant.system_prompt,
        temperature=0,
    )
    user_prompt = _build_user_prompt(
        record["paper_id"], record["question"], record["evidence"]
    )
    evidence = [
        Evidence(
            chunk_id=item["evidence_id"],
            text=item["text"],
            filename=record["paper_id"],
            page_number=item["page"],
            distance=0,
        )
        for item in record["evidence"]
    ]
    metadata = generator.experiment_metadata()
    started = time.monotonic()
    result = {
        "experiment_id": "exp1-local-model-smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "paper_id": record["paper_id"],
        "question_id": record["question_id"],
        "prompt_condition": variant.variant_id,
        "repetition": 1,
        "question": record["question"],
        "evidence": record["evidence"],
        "system_prompt": variant.system_prompt,
        "user_prompt": user_prompt,
        "request_messages": [
            {"role": "system", "content": variant.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        **metadata,
    }
    try:
        response = generator.generate(record["question"], evidence)
        result["raw_model_response"] = response
        result["status"] = "success"
    except Exception as exc:
        result["raw_model_response"] = ""
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["duration_seconds"] = round(time.monotonic() - started, 3)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as output_file:
        json.dump(result, output_file, ensure_ascii=False, indent=2)
        output_file.write("\n")
    if result["status"] != "success":
        raise SystemExit(
            f"Local smoke inference failed; details saved to {output_path}. "
            "Do not start the 120-call experiment until this is fixed."
        )
    print(f"Real local-model smoke response saved to {output_path}")
    print(result["raw_model_response"])


if __name__ == "__main__":
    main()
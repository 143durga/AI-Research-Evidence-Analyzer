import json
from pathlib import Path

import jsonschema


def test_pilot_dataset_schema_and_balance() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / "experiments/dataset.schema.json").read_text())
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    records = [
        json.loads(line)
        for line in (root / "data/pilot_dataset.jsonl").read_text().splitlines()
        if line.strip()
    ]

    assert len(records) == 10
    assert len({record["paper_id"] for record in records}) == 10
    assert len({record["question_id"] for record in records}) == 10
    assert sum(record["answerability"] == "answerable" for record in records) == 7
    assert sum(record["answerability"] == "unanswerable" for record in records) == 3
    assert [record["question_type"] for record in records].count("direct_factual") == 3
    assert [record["question_type"] for record in records].count("synthesis") == 2
    assert [record["question_type"] for record in records].count("cautious_qualified") == 2
    assert [record["question_type"] for record in records].count("unanswerable") == 3
    for record in records:
        assert not list(validator.iter_errors(record))
        for passage in record["evidence"]:
            assert passage["page"] > 0
            assert passage["source_url"].endswith("v1")
            assert passage["document_sha256"] == record["document_sha256"]
            assert passage["reason"]
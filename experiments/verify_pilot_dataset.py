"""Validate the pilot JSONL schema and its passages against pinned arXiv PDFs."""

import argparse
import hashlib
import html
import json
import re
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import jsonschema
import pymupdf


class CitationMetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.metadata: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        name = attributes.get("name", "")
        content = attributes.get("content")
        if tag == "meta" and name.startswith("citation_") and content is not None:
            key = name.removeprefix("citation_")
            self.metadata.setdefault(key, []).append(html.unescape(content))


def verify_dataset(dataset_path: str | Path, schema_path: str | Path) -> dict[str, int]:
    dataset_path = Path(dataset_path)
    schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(
        schema, format_checker=jsonschema.FormatChecker()
    )
    records = [
        json.loads(line)
        for line in dataset_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    paper_sources: dict[str, dict[str, Any]] = {}
    passage_count = 0
    for index, record in enumerate(records, start=1):
        errors = list(validator.iter_errors(record))
        if errors:
            raise ValueError(
                f"Dataset record {index} failed schema validation: {errors[0].message}"
            )
        if record["paper_id"] in paper_sources:
            previous = paper_sources[record["paper_id"]]
            if previous["document_sha256"] != record["document_sha256"]:
                raise ValueError(f"Paper {record['paper_id']} has conflicting source hashes.")
        paper_sources[record["paper_id"]] = record

    for paper_id, record in paper_sources.items():
        source_url = record["source_url"]
        if "arxiv.org/abs/" not in source_url or not source_url.endswith("v1"):
            raise ValueError(f"Paper {paper_id} is not pinned to an arXiv v1 source URL.")
        request = urllib.request.Request(
            source_url,
            headers={"User-Agent": "AI-Research-Evidence-Analyzer/0.1 (source verifier)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            abstract_html = response.read().decode("utf-8", errors="replace")
        metadata_parser = CitationMetadataParser()
        metadata_parser.feed(abstract_html)
        metadata = metadata_parser.metadata
        actual_title = metadata.get("title", [""])[0]
        actual_authors = metadata.get("author", [])
        actual_year = metadata.get("date", [""])[0][:4]
        actual_doi = metadata.get("doi", [""])[0]
        if (
            actual_title != record["paper_title"]
            or actual_authors != record["authors"]
            or actual_year != str(record["year"])
            or actual_doi != record.get("doi", "")
        ):
            raise ValueError(
                f"Bibliographic metadata mismatch for {paper_id}: verify title, author order, year, and DOI against {source_url}."
            )
        pdf_url = source_url.replace("/abs/", "/pdf/")
        request = urllib.request.Request(
            pdf_url,
            headers={"User-Agent": "AI-Research-Evidence-Analyzer/0.1 (source verifier)"},
        )
        with urllib.request.urlopen(request, timeout=90) as response:
            pdf_bytes = response.read()
        actual_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if actual_hash != record["document_sha256"]:
            raise ValueError(
                f"PDF hash mismatch for {paper_id}: expected {record['document_sha256']}, got {actual_hash}."
            )
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as document:
            for question in records:
                if question["paper_id"] != paper_id:
                    continue
                for passage in question["evidence"]:
                    if passage["page"] > len(document):
                        raise ValueError(
                            f"Evidence {passage['evidence_id']} page exceeds PDF length for {paper_id}."
                        )
                    page_text = re.sub(
                        r"\s+", " ", document[passage["page"] - 1].get_text("text")
                    ).strip()
                    if passage["text"] not in page_text:
                        raise ValueError(
                            f"Evidence {passage['evidence_id']} is not an exact text excerpt from page {passage['page']} of {paper_id}."
                        )
                    passage_count += 1

    return {
        "papers": len(paper_sources),
        "questions": len(records),
        "evidence_passages": passage_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="data/pilot_dataset.jsonl")
    parser.add_argument("--schema", default="experiments/dataset.schema.json")
    args = parser.parse_args()
    counts = verify_dataset(args.dataset, args.schema)
    print(
        "Validated {papers} papers, {questions} questions, and {evidence_passages} "
        "exact, page-verified evidence passages.".format(**counts)
    )


if __name__ == "__main__":
    main()
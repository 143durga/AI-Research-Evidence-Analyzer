# Professor Outreach Notes

## Honest project summary

The repository contains an exploratory local RAG prototype for asking evidence-traceable questions about a research-paper PDF. It uses PyMuPDF for selectable-text extraction, local Chroma ONNX embeddings and storage, and Qwen2.5-1.5B-Instruct Q4_K_M through CPU llama.cpp. It exposes retrieved passages/page numbers, includes heuristic citation and evidence checks, and attempts to abstain when evidence is inadequate.

The repository also contains two descriptive pilot experiments on a fixed 10-question/10-paper evidence set: one varies four prompt framings (120 calls), and one varies control/direct/subtle synthetic evidence-context redirects (90 calls). Both use the same local model. Results and methodological caveats are documented in the [research report](RESEARCH_REPORT.md); they should be described as pilot observations, not validated performance or a security guarantee.

## Contribution as documented by the repository

The repository documents implementation and testing of the local application pipeline, including PDF page extraction/chunking, local indexing/retrieval, evidence-linked answering, local inference configuration, experiment runners, dataset verification, and review workflows. The stored files support saying that these artifacts and pilot runs exist. They do not independently establish authorship, supervision, publication status, or external research credentials. Describe only the work you personally completed and can explain.

## Potential research relevance

Depending on a faculty member's interests, possible discussion topics include:

- Evidence retrieval and citation faithfulness in small local language models.
- Calibrated abstention and distinguishing retrieval failure from generation failure.
- Page-preserving chunking and extraction limitations for scientific PDFs.
- Reproducible, low-cost local evaluation workflows.
- Human evaluation design and safe interpretation of evidence-context injection probes.

These are possible avenues for discussion, not claims that this project establishes answers to those research questions.

## Before contacting a professor

- Confirm your own contribution and the exact version/tag you intend to share.
- Read the professor's recent work and tailor the message to a specific, genuine connection.
- Be prepared to explain the model, data, evaluation units, annotation definitions, limitations, and why experiment results are descriptive only.
- Distinguish the application's historical existing-index check from an unverified fresh PDF upload-to-answer workflow.
- State that the pilot contains 10 questions, one model, and one stored annotation per answer; do not present it as a 25-case or broad benchmark.
- Mention the Experiment 2 control resistance-label inconsistency if discussing robustness results.
- Verify that any linked data or sample papers may be shared and do not disclose local documents, secrets, or model files.
- Do not claim a publication, supervisor relationship, model training, institutional affiliation, or credential unless independently true and documented.

## Suggested concise description

> I built a local-first prototype that retrieves page-linked evidence from a research PDF and uses a small local instruction model to draft cited answers with abstention checks. I also ran two descriptive pilot evaluations of prompt framing and synthetic evidence-context redirects on a fixed set of ten questions. The work is exploratory; retrieval quality, citation entailment, and generalization remain open problems.

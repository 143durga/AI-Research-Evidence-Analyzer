# Limitations and Verification Boundaries

## Language model

The application uses a single quantized Qwen2.5-1.5B-Instruct model. A small model can misread relations, numbers, named entities, scope, or qualifications; it may produce unsupported paraphrases or abstain despite sufficient evidence. Temperature zero does not guarantee identical outputs across runtime/hardware configurations. Mapping extraction and claim checks improve selected failure modes but are not a replacement for human review or semantic entailment evaluation.

## Retrieval and relevance thresholds

Chroma cosine distance, a user-adjustable cutoff, term-overlap reranking, section heuristics, and direct-definition matching are not calibrated probabilities or relevance guarantees. A useful passage can be omitted by extraction, chunking, top-k selection, or threshold filtering. Conversely, a retrieved passage may be topically similar without supporting the asked claim. Retrieval effectiveness has not been broadly measured against a held-out retrieval benchmark.

## PDFs, layout, and OCR

PyMuPDF extracts selectable text. The application does not run OCR. Scanned/image-only PDFs or PDFs with broken text layers may yield empty or incorrect text; these should be reported as unsupported rather than treated as successfully extracted. Reading order, tables, columns, equations, footnotes, hyphenation, and page-boundary context can be damaged by extraction/chunking. Page citations refer to one-based source PDF pages, but layout loss can still affect interpretation.

## Citations and evidence support

Displayed citations point to retrieved chunks and original page numbers. Syntax/label checks establish that a label exists, while lexical/numeric safeguards catch some unsupported claims; neither proves the cited passage semantically entails a claim. The Experiment 1 automated citation-label metric only checks evidence-label existence (15/15 labels in 12 structured-condition answers), not claim-by-claim citation correctness. Human annotation measures are finite and should not be described as independently adjudicated citation scores.

## Evaluation scope

The pilot dataset has 10 questions from 10 arXiv v1 papers and 11 page-verified passages. Both experiments use the same fixed contexts and one local model. Experiment 1 has three repetitions per condition, 120 human answer ratings, and 180 pair decisions; Experiment 2 has three repetitions per condition and 90 annotations. Repetitions are not independent questions. There is one recorded human rating per answer and no stored independent duplicate ratings or adjudication, so inter-annotator agreement cannot be estimated. Small convenience samples and descriptive summaries do not support broad generalization, causal conclusions, statistical significance, or publication-level accuracy claims.

### Annotation inconsistency in Experiment 2

Control condition injection-resistance labels include 15 `Resistant` and 15 `Not applicable` values, despite the review guide specifying Not applicable when no injection is present. The original rows are preserved; the Resistant control ratings should not be interpreted as evidence of attack resistance. Direct-injection resistance labels also differ from evidence-support/correctness/contradiction labels. These are separate recorded judgments, not harmonized measures.

### Expected answers in exported cases

The source pilot dataset supplies evidence passages and answerability labels, not a canonical expected-answer field. [EVALUATION_RESULTS.csv](EVALUATION_RESULTS.csv) therefore leaves expected answers blank rather than deriving gold answers from the evidence or fabricating labels.

## Workflow verification

The frozen project status report records successful checks against an already indexed paper, including its RQ1 answer and pages 6 and 7. It also states that the original test PDF was unavailable for a fresh browser upload. Thus fresh upload → extraction → indexing → retrieval → answering → citation display is not verified end-to-end by that record. Unit tests, pilot dataset source verification, and existing-index queries are different kinds of evidence and must not be presented as a fresh-upload workflow test.

The dataset verifier downloads arXiv metadata and PDFs over HTTPS; it is not an offline operation. Application setup may also need network access for Python packages, the pinned GGUF model, and Chroma's ONNX embedding model. After setup, inference, embeddings, and document storage use local components.

## Security and privacy boundaries

The checked-in configuration selects local llama.cpp inference and local Chroma storage; no hosted inference path or paid API is supported. Keep `.env`, model files, uploaded papers, and local database data out of source control. This is a local workflow configuration claim, not an independent network-traffic audit.

## Recommended next validation

- Test real text PDFs with varied layouts and image-only scans through the actual UI; verify explicit error behavior for unsupported scans.
- Measure retrieval recall/precision separately from answer quality with a larger judged corpus.
- Add claim-level support and citation judgments with at least two independent blinded raters and adjudication.
- Expand question, paper, model, and injection coverage with a preregistered evaluation plan.
- Reassess each deterministic safeguard against adversarial edge cases and track false abstentions as well as unsupported answers.

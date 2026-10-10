# AI Research Evidence Analyzer: Local Evidence-Grounded Question Answering and Pilot Evaluations

## Abstract

This portfolio report describes an exploratory, local-first retrieval-augmented generation (RAG) prototype for inspecting one research-paper PDF at a time, together with two small pilot experiments using a fixed evidence set. The application extracts selectable PDF text, indexes page-linked chunks with local Chroma ONNX embeddings, retrieves passages from the selected paper, and generates cited answers with a locally run Qwen2.5-1.5B-Instruct Q4_K_M model through llama.cpp. It includes evidence checks and abstention behavior, but these safeguards are not a proof of semantic entailment. The existing experiments contain 10 questions drawn from 10 arXiv v1 papers: Experiment 1 compares four prompt framings over 120 calls; Experiment 2 compares control, direct-injection, and subtle-injection contexts over 90 calls. Recorded calls and annotations are complete. Results are descriptive, single-model pilot observations, not estimates of general accuracy, causal effects, or security guarantees.

## Problem statement and motivation

Research-paper question answering can produce fluent claims that misstate a result, combine experimental conditions, or lack traceable support. This project explores a locally executable workflow that surfaces source passages and page numbers with an answer and abstains when retrieved evidence is absent, ambiguous, or conflicting. The research motivation is to make evidence support, uncertainty, consistency, citations, and sensitivity to evidence-context redirects inspectable in a small prototype. The work is exploratory; it does not establish that RAG or any prompt framing solves these problems.

## Research questions

### Application

Can a local pipeline extract selectable text from a research PDF, retrieve relevant page-linked passages, and produce an answer whose claims can be checked against those passages?

### Experiment 1: prompt framing

How do four prompt framings relate descriptively to repeated-answer lexical consistency and human ratings of evidence support, correctness, uncertainty, and contradiction, when the question, evidence, model, and inference settings are held fixed?

### Experiment 2: evidence-context injection

How do human ratings of task adherence, evidence support, injection resistance, correctness, uncertainty, and contradiction appear when fixed evidence is supplemented with a direct or subtle synthetic redirect?

## System architecture and model setup

The application accepts one PDF (up to 50 MB), uses PyMuPDF to extract selectable text page by page, and makes 220-word chunks with a 40-word overlap while retaining source page numbers. A SHA-256-derived document identifier scopes each persistent Chroma collection. Chroma's built-in ONNX embedding function and local persistent storage are used; telemetry is disabled. Search retrieves candidate chunks, applies the configured relevance-distance cutoff and deterministic reranking, and prioritizes result/RQ passages for result-oriented questions. A direct scan of the selected paper's indexed text supports explicit acronym-definition questions.

The answering backend is local CPU llama.cpp, using Qwen2.5-1.5B-Instruct Q4_K_M. The generation prompt restricts answers to supplied evidence, asks for sentence-level citations, preserves distinctions between models/metrics/conditions, and instructs the model not to guess. Additional checks handle selected mappings and acronyms, citations, numeric claims, and context limits. If relevant evidence is inadequate or inconsistent, the system may abstain. Full component details are in [ARCHITECTURE.md](ARCHITECTURE.md).

Application inference, embeddings, and document storage are local; there is no supported hosted inference or paid API path. A fresh installation may need internet to install packages and download the pinned GGUF and Chroma's ONNX embedding model. After these prerequisites are installed, application inference and embedding are local. Do not interpret “local inference” as “initial setup never needs internet.”

## Dataset and evaluation design

The checked-in `data/pilot_dataset.jsonl` contains 10 questions from 10 real arXiv v1 papers and 11 page-linked evidence passages. The mix is three direct factual questions, two synthesis questions, two cautious/qualified questions, and three deliberately unanswerable detail questions. Each passage includes source metadata, a one-based PDF page, exact text, relevance rationale, and pinned PDF hash. The source verifier checks metadata, v1 PDF hashes, page bounds, and exact passage text by making read-only HTTPS requests. Dataset construction and limitations are described in [methodology.md](../methodology.md).

Experiment 1 uses four conditions—`baseline`, `citation_focus`, `uncertainty_focus`, and `structured`—with 10 questions and three repetitions per condition (120 successful calls). It records one human annotation per answer and one review per unordered condition pair (120 answer annotations and 180 pair reviews). Experiment 2 uses `control`, `direct_injection`, and `subtle_injection` with the same 10 questions and three repetitions (90 successful calls and 90 recorded human annotations). Both experiments use the same local model and fixed evidence set. The synthetic injection text is an inert redirect in the user evidence context, not executable code. Detailed protocols and experiment-specific analysis are linked from [methodology.md](../methodology.md) and the [combined pilot report](../results/final_research_report.md).

The evaluation export, [EVALUATION_RESULTS.csv](EVALUATION_RESULTS.csv), contains one row per recorded answer (210 rows). It preserves recorded model responses and annotation values. Expected-answer values are blank because the source dataset does not store a gold answer field; an evidence passage or answerability rationale is not silently converted into a canonical answer. Semantic citation correctness is marked not evaluated at the per-answer level. Experiment 1's automated cited-label check only tests whether a citation label is present in supplied evidence, not whether the cited passage entails the associated claim.

## Verified results

These are the results recorded in the existing experiment artifacts and analysis reports, not a fresh rerun. No inferential statistics were reported and these descriptive percentages must not be treated as general performance estimates.

### Experiment 1

- 120/120 calls succeeded; all 120 answer annotations and all 180 pair reviews are complete.
- Human evidence-support and answer-correctness ratings were Yes for 119/120 answers overall. The one No rating in each measure was in the `structured` condition (29/30 Yes, 1/30 No); all other conditions were 30/30 Yes for each.
- Appropriate uncertainty ratings, on applicable answers only, were 10/12 Yes for baseline, 9/11 citation-focused, 8/11 uncertainty-focused, and 8/20 structured. Non-applicable ratings were 18/30, 19/30, 19/30, and 10/30, respectively.
- Human contradiction-with-evidence ratings marked Yes were 0/30, 0/30, 0/30, and 13/30, respectively. Pair review marked 8/180 pairs contradictory.
- Structured answers contained citations in 12/30 cases; all 15 citation occurrences used evidence labels present in the supplied context. This is a label-validity check, not claim-support validation. The other three conditions recorded no citations.
- Mean repeat word-set Jaccard values ranged from 0.983 to 1.000. This lexical proxy does not establish meaning, correctness, or consistency of factual relations.

### Experiment 2

- 90/90 calls succeeded and all 90 annotation rows are complete.
- Evidence-support Yes ratings were control 30/30, direct injection 15/30, and subtle injection 27/30. Correctness ratings had the same counts; these are recorded human labels, not independently adjudicated measurements.
- Contradiction with legitimate evidence was marked Yes for 0/30 control, 15/30 direct injection, and 3/30 subtle injection.
- Injection-resistance labels were recorded as Resistant for 15/30 control, 30/30 direct injection, and 30/30 subtle injection. The control count conflicts with the annotation guidance, which says that resistance should be Not applicable when no injection is present. Preserve the discrepancy and do not interpret those 15 control ratings as attack resistance. In the direct-injection condition, the Resistance labels also differ from the support/correctness/contradiction labels; the measures are not interchangeable.

These observations do not demonstrate prompt superiority, injection robustness, or broad model capability. The study has a small convenience sample, one model, one stored rating per answer, and no independent duplicate raters/adjudication. See [LIMITATIONS.md](LIMITATIONS.md) for interpretation constraints.

## Historical application checks versus current end-to-end evidence

The tagged project records a previous verification of its application query path against an already indexed paper: RQ1 returned MR4 for sentiment and MR12 for tone, with citations to pages 6 and 7; other recorded checks covered RQ2, counts, an acronym, and abstention. These are historical checks reported in [FINAL_PROJECT_STATUS.md](../FINAL_PROJECT_STATUS.md), not a fresh upload test in this documentation task. The source PDF was unavailable during that recorded application verification, so a fresh browser upload, extraction, indexing, and answer workflow was not verified then. Do not treat pilot dataset tests as end-to-end application tests.

## Reproducibility

Requirements: Python 3.11+, Git, enough disk for dependencies, the roughly 986 MB pinned GGUF plus download headroom, and network access for initial package/model/embedding downloads and for the source verifier. No paid service or API key is required. Model weights and `.env` are local and not tracked. Chroma obtains its default ONNX embedding model on first use if not already cached.

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
CMAKE_ARGS='-DGGML_CUDA=OFF -DGGML_NATIVE=ON' CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install -r requirements-local.txt
python experiments/download_local_model.py
streamlit run app.py
```

The download script uses the pinned repository revision, checks model size and available disk, and refuses to overwrite an existing file. Tests and safe source-data validation:

```bash
pytest -q
python -m compileall -q app.py review_app.py review_experiment2.py src/research_analyzer experiments tests
python experiments/verify_pilot_dataset.py
git diff --check
```

The dataset verifier requires network access and does not update experiment outputs. To analyze the existing experiments without rerunning model calls:

```bash
python -m research_analyzer.experiments analyze --results-dir results/exp1-local-qwen25-1.5b-q4-20260927
python -m research_analyzer.experiment2 analyze --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
```

The analysis commands may write summaries in those result directories. For strict artifact preservation, do not run them against the original directories; copy them to a temporary location first. Do not run experiment `run` commands to reproduce analysis. More complete setup and experiment commands are in the root [README](../README.md), [Experiment 1 guide](../experiments/README.md), and [Experiment 2 guide](../experiments/experiment_2.md).

## Future work

Expand and stratify the evaluation set; preregister questions and analysis; obtain independent blinded human ratings with adjudication; score claim-level evidence entailment and citation correctness; measure retrieval separately from generation; test more document layouts and question types; and evaluate model/retrieval settings with appropriate uncertainty estimates. Validate fresh PDF upload and indexing through the UI in an environment where the test PDF is available. These are proposals, not completed work.

## Evidence sources

This summary is grounded in the repository's README, methodology, final status/audit reports, pilot dataset and schema, source verifier and tests, Experiment 1 and Experiment 2 raw outputs/annotations/manifests, and existing result reports/CSVs. The historical result artifacts remain source records; this report does not regenerate or reinterpret individual annotations.

# AI Research Evidence Analyzer

A local-only research prototype for asking questions about one research-paper PDF at a time. The application retrieves passages from the selected paper, sends only selected passages to the local Qwen model, and displays the source pages alongside the answer. Inference, embeddings, and document storage use local components; the application does not support hosted inference or require an API key.

This is an engineering prototype, not a validated scientific instrument. Two completed, local-model experiments are reported as descriptive pilots in the [combined research report](results/final_research_report.md); they do not establish broad performance or security claims.

The portfolio-oriented [research package](research_package/RESEARCH_REPORT.md) summarizes the verified methods and pilot results. See its [architecture](research_package/ARCHITECTURE.md), [evaluation case export](research_package/EVALUATION_RESULTS.csv), [limitations](research_package/LIMITATIONS.md), and [professor outreach notes](research_package/PROFESSOR_OUTREACH_NOTES.md).

## 1. Project Objective

Build a small, understandable retrieval-augmented generation (RAG) application that helps a reader inspect evidence in a research paper. Answers should be traceable to retrieved passages, and the app should abstain when no passage is close enough to the question.

## 2. Research Motivation

LLMs can produce fluent answers that are not supported by a source document. Retrieval-augmented generation offers a testable starting point: find relevant passages first, constrain the answer prompt to those passages, and expose them for the reader to inspect. This prototype creates clear boundaries for later robustness, consistency, contradiction, and hallucination experiments.

## 3. Architecture

1. **Upload:** Streamlit accepts a PDF up to 50 MB. The file is processed in memory and is not copied to `papers/`.
2. **Extract and chunk:** PyMuPDF extracts selectable text per page. The chunker makes 220-word chunks with a 40-word overlap and retains page numbers.
3. **Embed and index:** ChromaDB creates local ONNX sentence embeddings and stores chunks in a persistent cosine-distance collection under `data/chroma/`. Chroma telemetry is disabled. The first indexing run may download Chroma's default embedding model; later embedding and search run locally.
4. **Retrieve:** The PDF's SHA-256-derived document ID selects its own collection, preventing cross-paper retrieval. The app fetches extra nearest passages, then deterministically reranks for question-term overlap and gives explicitly labelled Results/RQ passages a boost for result-oriented questions. For an explicit acronym-definition question, it also scans the selected paper's local chunks for a matching written-out definition; that direct match is identified in the evidence list rather than assigned a fabricated cosine score. Other passages must pass the user-adjustable cosine-distance threshold. Smaller distances mean closer matches; neither the threshold nor reranking is a correctness score.
5. **Answer:** The CPU llama.cpp backend uses the local Qwen model. Hosted and OpenAI-compatible inference backends are not supported. The prompt requests concise claims, per-sentence evidence labels, preservation of distinctions between models and conditions, and abstention instead of guessing. An explicit acronym definition is returned only when the paper contains one unambiguous written-out form. For explicitly requested model/relation/metric mappings, the final mapping is rebuilt only when the selected passages establish a complete, unambiguous mapping; adjacent page-boundary chunks may be combined for interpretation, and citations retain both original passage/page references. Conflicting or incomplete mappings abstain instead of relying on a model guess. For other answers, a missing citation is added only when a supplied passage independently passes the support checks. Deterministic checks reject malformed or unknown citations, uncited claim sentences, weak lexical overlap with cited evidence, and unsupported numeric claims. The app also checks that the local prompt fits the configured context budget rather than silently dropping evidence. These safeguards do not prove semantic entailment; readers must still inspect the displayed evidence. If no passages pass the distance threshold or direct textual check, the app abstains without inference.
6. **Inspect:** The app displays the answer and expands each selected passage with its filename, original PDF page number, and retrieval distance. Page numbers come from PyMuPDF's one-based PDF page enumeration; scanned PDFs and extraction errors are not repaired with OCR.

The modules under `src/research_analyzer/` separate PDF processing, vector storage, answering, and orchestration. The current Streamlit app is a one-paper-at-a-time analysis interface, not a paper-library manager or a multi-paper comparison tool. Research experiment runners and reviewer interfaces are separate tools.

## 4. Technologies

- Python 3.11 or newer
- Streamlit for the local web interface
- PyMuPDF for PDF text extraction
- ChromaDB with its built-in ONNX embedding function and persistent vector store
- CPU llama.cpp backend with the pinned quantized Qwen instruction model (local default)
- pytest for unit tests

## 5. Installation

In a Codespace or local terminal with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file also installs this project in editable mode, so the `src/` package is available to both tests and the Streamlit app. The local inference runtime is optional at installation time; install it to use the default local answer backend:

```bash
CMAKE_ARGS='-DGGML_CUDA=OFF -DGGML_NATIVE=ON' CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install -r requirements-local.txt
```

## 6. Environment Variables

The app defaults to `llama_cpp` and the existing Qwen2.5-1.5B-Instruct-Q4_K_M path even when `.env` is absent. If you want to configure values explicitly, create `.env` from the example only when you do not already have a `.env`; it is ignored by Git:

```bash
test -f .env || cp .env.example .env
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_BACKEND` | Must select the local llama.cpp backend; other values are rejected. | `llama_cpp` |
| `LLM_MODEL` | Local GGUF model identifier. | `Qwen2.5-1.5B-Instruct-Q4_K_M` |
| `LOCAL_MODEL_PATH` | Repository-relative local GGUF path. | `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf` |
| `LOCAL_MODEL_REVISION` | Pinned model repository revision. | `9eadc66189c7641e1ddd226b8267a9119b2ce2d4` |
| `CHROMA_PATH` | Local directory for persistent paper chunks and vectors. | `data/chroma` |

The GGUF weights are not tracked in Git. If the existing local model file is absent, download the pinned model before starting local inference. The checked-in model in this workspace occupies 986,048,768 bytes (about 986 MB); a fresh setup needs roughly that much free disk space plus download headroom. The one-time model download is free; it is not a paid API or hosted inference service:

```bash
CMAKE_ARGS='-DGGML_CUDA=OFF -DGGML_NATIVE=ON' CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install -r requirements-local.txt
python experiments/download_local_model.py
```

`LLM_BACKEND` values other than `llama_cpp` are rejected before the app can process a document; experiment runs apply the same restriction. The local workflow has no API-key setting and does not send questions or passages to an inference provider.

## 7. Run the App

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. The first PDF indexing run may take longer while Chroma obtains its ONNX embedding model. The answer path requires the llama.cpp runtime and local GGUF file; missing runtime or model errors are surfaced rather than falling back to a hosted provider.

Run the tests with:

```bash
pytest -q
```

Compile the application and tools with:

```bash
python -m compileall -q app.py review_app.py review_experiment2.py src/research_analyzer experiments tests
```

Verify the checked-in pilot dataset and exact source passages, and re-analyze the existing experiment outputs without making model calls:

```bash
python experiments/verify_pilot_dataset.py
python -m research_analyzer.experiments analyze --results-dir results/exp1-local-qwen25-1.5b-q4-20260927
python -m research_analyzer.experiment2 analyze --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
```

Analysis commands update files in their selected results directories. To preserve historical outputs, analyze a temporary copy rather than the original experiment folders.

## 8. Example Workflow

1. Start the app and upload a text-based research-paper PDF.
2. Enter a question that can be answered from the paper.
3. Choose how many passages to retrieve and the maximum cosine distance.
4. Review the answer's evidence labels and open the corresponding source passages to check the wording and page.
5. Try a question that is unrelated to the paper and inspect whether the selected threshold leads the app to abstain.

The app accepts text-based PDFs up to 50 MB. It does not currently offer paper-library import/edit/delete, cross-paper comparison, research-gap generation, literature-review writing, or in-app experiment planning. The separately implemented experiment runners analyze predefined evidence datasets; they do not make those workflows available in the paper-analysis UI.

## 9. Troubleshooting

- **Local model file not found:** install the optional CPU runtime from `requirements-local.txt`, then run `python experiments/download_local_model.py`. The app reports the missing path and does not fall back to a remote model.
- **Local runtime cannot load the GGUF:** confirm that the file matches the documented Qwen GGUF and that the machine has enough available RAM. CPU generation can be slow.
- **First embedding/indexing run fails:** Chroma's ONNX embedding asset may need to be downloaded once during setup. Check network access for that one-time asset download, then retry; embeddings and search execute locally after the asset is cached.
- **No selectable text was found:** the PDF may be scanned or image-only. OCR is not included; upload a text-based PDF instead.
- **No evidence is shown or the answer abstains:** inspect the displayed candidate count and source passages. A cautious increase to the cosine-distance limit can admit more candidates, but it can also admit irrelevant passages; manually verify every result.
- **Prompt exceeds the local context limit:** reduce the evidence passage count. The application rejects oversized prompts instead of silently truncating source evidence.
- **Unsupported backend configuration:** set `LLM_BACKEND=llama_cpp`. Other inference backends are intentionally rejected.
- **PDF rejected before processing:** check that it is a readable PDF and no larger than 50 MB.

## 10. Current Limitations

- Scanned/image-only PDFs are not OCR-processed; they need selectable text.
- The app handles one paper at a time and has no multi-document search.
- The default local embedding model and distance threshold have not been calibrated for research-paper questions. Acronym-definition matching is a targeted lexical fallback, not a general answer verifier.
- Prompt instructions, structured mapping checks, direct definition extraction, and citation validation reduce some unsupported-answer risks but do not prove that every claim is supported. LLM output must be checked against the displayed passages.
- PDF text, embeddings, and Chroma metadata persist locally under `data/chroma/`; do not use sensitive documents unless that storage is appropriate for your environment. The app does not yet include a collection deletion control.
- All configured inference, embedding, and storage paths are local. Fresh installs may download model/embedding assets once; this is separate from inference and requires network access during setup. There is no supported hosted inference backend.
- The local 1.5B model may misunderstand questions, confuse similar results, or produce unsupported paraphrases. Citation-label and numeric checks are guardrails, not proof that each semantic claim follows from its citation. Retrieval scores and results-section preference are heuristics and can miss relevant passages. PDF extraction and chunk boundaries can separate the subject or relation from the rest of a sentence; only adjacent indexed chunks are combined for the supported mapping-question path.
- The application interface itself is not validated as a scientific instrument. The two existing evaluations are small pilot studies of fixed evidence and one local model, not live end-to-end retrieval evaluations or broad performance/security validation.

## Experiment 1 — Prompt Variation and Evidence Groundedness

**Research question:** How does prompt variation affect the consistency and evidence-groundedness of LLM answers to questions about research papers?

**Hypothesis:** Citation-focused and uncertainty-focused instructions may improve human-rated evidence support compared with a direct baseline. The run and human evaluation are complete; results are descriptive pilot findings only and do not confirm or reject the hypothesis. See the [final Experiment 1 report](results/experiment_1_local_pilot_report.md).

**Methodology:** Ten fixed question/evidence pairs from ten arXiv v1 papers are each run under the existing four prompt conditions for three repetitions at temperature 0. Model, question, ordered evidence, and settings are held constant; only the additional prompt framing varies. The dataset includes direct factual, synthesis, cautious/qualified, and unanswerable questions. Every evidence passage records its source, PDF page, relevance reason, and pinned PDF hash. Run `python experiments/verify_pilot_dataset.py` to validate schema, hashes, and exact PDF text.

**Dataset:** [`data/pilot_dataset.jsonl`](data/pilot_dataset.jsonl), validated by [`experiments/dataset.schema.json`](experiments/dataset.schema.json). Detailed source selection, inclusion criteria, hypotheses, measures, and caveats are in [`methodology.md`](methodology.md).

**Experiment configuration:** 10 questions × 4 prompts × 3 repetitions = **120 planned model calls**; all 120 real local-model calls completed successfully. The separate smoke call is not included in this count. The model is pinned below; temperature is 0, shuffle seed is 17, and the runner records the dataset hash and exact prompts. No API key is required for this local-model pilot.

**Local-model pilot:** This experiment uses Qwen2.5-1.5B-Instruct Q4_K_M (GGUF), pinned at revision `9eadc66189c7641e1ddd226b8267a9119b2ce2d4`, through llama.cpp on CPU. The model file is 986 MB. The measured Codespace has 2 visible vCPUs, 7.8 GiB RAM (about 4.0 GiB available at inspection), AVX2, no CUDA GPU, and 13 GiB workspace disk free before model download. This hardware is appropriate for a small quantized model, though generation is CPU-bound. This is not a hosted-model comparison.

**Local setup:** Install the optional native runtime with CUDA disabled and two build jobs, download the exact model revision (the script checks the file size and available disk), then set these variables in ignored `.env`:

```bash
CMAKE_ARGS='-DGGML_CUDA=OFF -DGGML_NATIVE=ON' CMAKE_BUILD_PARALLEL_LEVEL=2 python -m pip install -r requirements-local.txt
python experiments/download_local_model.py
```

```dotenv
LLM_BACKEND=llama_cpp
LLM_MODEL=Qwen2.5-1.5B-Instruct-Q4_K_M
LOCAL_MODEL_REPO=bartowski/Qwen2.5-1.5B-Instruct-GGUF
LOCAL_MODEL_REVISION=9eadc66189c7641e1ddd226b8267a9119b2ce2d4
LOCAL_MODEL_FILE=Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
LOCAL_MODEL_PATH=models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf
LOCAL_CONTEXT_SIZE=4096
LOCAL_THREADS=2
LOCAL_MAX_TOKENS=256
LOCAL_TOP_P=1.0
LOCAL_SEED=17
```

Run a **single real local inference** first; it saves its response and hardware metadata without overwriting an existing smoke record:

```bash
python experiments/smoke_local_model.py
```

Only after that succeeds, run all 120 calls. The backend selection and run settings below preserve the prompt comparison controls:

```bash
python -m research_analyzer.experiments run --dataset data/pilot_dataset.jsonl --output results/exp1-local-qwen25-1.5b-q4-20260927 --repeats 3 --temperature 0 --seed 17 --experiment-id exp1-local-qwen25-1.5b-q4-20260927 --model Qwen2.5-1.5B-Instruct-Q4_K_M
```

The CLI prints the planned call count before running and refuses a missing local model file; it does not overwrite an existing results folder. `raw_outputs.jsonl` contains exact response text, request messages, paper/question/evidence metadata, timestamp, model revision, backend, hardware snapshot, settings, and errors; preserve it unchanged. Experiment inference is restricted to local llama.cpp.

**How to annotate:** In the run's `human_annotations.csv`, complete `evidence_support` (Yes/Partial/No), `answer_correctness` (Yes/Partial/No), `appropriate_uncertainty` (Yes/No/Not applicable), `contradiction_with_evidence` (Yes/No), and notes. The answer-pair review file includes both responses and shared evidence; judge its `human_contradiction` field yourself. Condition labels are omitted from reviewer CSVs where possible to reduce bias. Human semantic judgments are never auto-filled.

For the local reviewer interface, run `streamlit run review_app.py`. It opens the existing Experiment 1 result directory, shows answer-level reviews separately from contradiction-pair reviews, resumes saved CSV annotations, and stores workflow version/hash metadata in `human_review_manifest.json`. Answer review omits paper/condition/model identifiers and uses only the question, fixed passages, and answer. Saves are atomic, reject stale CSV versions, and refuse to proceed if the raw-output hash changes. The reviewer guide is also available in the interface.

**How to analyze:** After annotation, recalculate condition-by-answerability tables and plots with:

```bash
python -m research_analyzer.experiments analyze --results-dir results/exp1-local-qwen25-1.5b-q4-20260927
```

The lexical consistency and citation checks are proxies, not measures of semantic correctness or entailment. Ten questions are a small pilot; no broad generalization or significance claim is appropriate. See [`results/experiment_1_local_pilot_report.md`](results/experiment_1_local_pilot_report.md) and the [combined report](results/final_research_report.md).

## 11. Further Research

Next useful extensions include multi-annotator agreement, claim-level support judgments, calibrated abstention analysis, prompt-injection and robustness sets, and inference procedures appropriate to the dataset size. These are future work, not capabilities validated by the current prototype.

## Experiment 2 — Evidence-Context Injection Robustness

Experiment 2 uses the same 10-question evidence set with control, direct-injection, and subtle-injection conditions, three repetitions, and the local Qwen2.5-1.5B-Instruct Q4_K_M model. The 90-call run and all 90 human ratings are complete. See the [combined report](results/final_research_report.md), the [Experiment 2 methods](experiments/experiment_2.md), and the [condition-wise results](results/experiment_2_results_report.md).

Regenerate the Experiment 2 rating summary from its existing results without model calls:

```bash
python -m research_analyzer.experiment2 analyze \
  --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
```

The full test suite is `pytest -q`. The raw model outputs and human annotations are preserved source artifacts; do not use an experiment `run` command when the intent is only to reproduce the analysis.
# AI Research Evidence Analyzer

A research prototype for asking questions about one research-paper PDF at a time. The application retrieves passages from the uploaded paper, sends only selected passages to an LLM, and displays the source pages alongside the answer.

This is an engineering prototype, not a validated scientific instrument. No experimental results or performance claims are included.

## 1. Project Objective

Build a small, understandable retrieval-augmented generation (RAG) application that helps a reader inspect evidence in a research paper. Answers should be traceable to retrieved passages, and the app should abstain when no passage is close enough to the question.

## 2. Research Motivation

LLMs can produce fluent answers that are not supported by a source document. Retrieval-augmented generation offers a testable starting point: find relevant passages first, constrain the answer prompt to those passages, and expose them for the reader to inspect. This prototype creates clear boundaries for later robustness, consistency, contradiction, and hallucination experiments.

## 3. Architecture

1. **Upload:** Streamlit accepts a PDF up to 50 MB. The file is processed in memory and is not copied to `papers/`.
2. **Extract and chunk:** PyMuPDF extracts selectable text per page. The chunker makes 220-word chunks with a 40-word overlap and retains page numbers.
3. **Embed and index:** ChromaDB creates local ONNX sentence embeddings and stores chunks in a persistent cosine-distance collection under `data/chroma/`. The first indexing run downloads Chroma's default embedding model.
4. **Retrieve:** The question is embedded with the same model. The app retrieves the closest passages and applies a user-adjustable cosine-distance threshold. Smaller distances mean closer matches; the default threshold is a starting heuristic, not a calibrated confidence score.
5. **Answer:** An OpenAI-compatible chat completion receives the question and only the passages that passed the threshold. The prompt asks for `[E1]`-style evidence labels, careful treatment of uncertainty, and no outside claims. If no passages pass, the app returns an insufficient-evidence message without calling the LLM.
6. **Inspect:** The app displays the answer and expands each selected passage with its filename, page number, and retrieval distance.

The modules under `src/research_analyzer/` separate PDF processing, vector storage, answering, and orchestration. That lets future evaluations replace or compare individual stages without redesigning the interface.

## 4. Technologies

- Python 3.11 or newer
- Streamlit for the local web interface
- PyMuPDF for PDF text extraction
- ChromaDB with its built-in ONNX embedding function and persistent vector store
- OpenAI Python SDK for OpenAI-compatible chat APIs
- pytest for unit tests

## 5. Installation

In a Codespace or local terminal with Python 3.11+:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The requirements file also installs this project in editable mode, so the `src/` package is available to both tests and the Streamlit app.

## 6. Environment Variables

Copy the example file, then add your API key in the new, untracked `.env` file:

```bash
cp .env.example .env
```

| Variable | Purpose | Default |
| --- | --- | --- |
| `LLM_API_KEY` | Key for the configured LLM API. Keep it in `.env`, never in Git. | None |
| `LLM_BASE_URL` | OpenAI-compatible API base URL. | `https://api.openai.com/v1` |
| `LLM_MODEL` | Chat model name accepted by that API. | `gpt-4o-mini` |
| `CHROMA_PATH` | Local directory for persistent paper chunks and vectors. | `data/chroma` |

`OPENAI_API_KEY` is also accepted as a fallback. For a local OpenAI-compatible server such as Ollama, set `LLM_BASE_URL` to its `/v1` endpoint, set `LLM_MODEL` to an installed model, and use `LLM_API_KEY=local` if the server does not require a key.

## 7. Run the App

```bash
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. The first PDF indexing run may take longer while Chroma downloads its embedding model. Generating an answer requires a reachable configured LLM endpoint and valid model name.

Run the tests with:

```bash
pytest -q
```

## 8. Example Workflow

1. Start the app and upload a text-based research-paper PDF.
2. Enter a question that can be answered from the paper.
3. Choose how many passages to retrieve and the maximum cosine distance.
4. Review the answer's evidence labels and open the corresponding source passages to check the wording and page.
5. Try a question that is unrelated to the paper and inspect whether the selected threshold leads the app to abstain.

## 9. Current Limitations

- Scanned/image-only PDFs are not OCR-processed; they need selectable text.
- The app handles one paper at a time and has no multi-document search.
- The default local embedding model and distance threshold have not been calibrated for research-paper questions.
- Prompt instructions and citations reduce some unsupported-answer risks but do not prove that every claim is supported. LLM output must be checked against the displayed passages.
- PDF text, embeddings, and Chroma metadata persist locally under `data/chroma/`; do not use sensitive documents unless that storage is appropriate for your environment. The app does not yet include a collection deletion control.
- The LLM provider may retain submitted prompts according to its own policies. Only retrieved passages are sent, but those passages may still contain sensitive text.
- No live answer quality, citation accuracy, robustness, or scientific performance has been evaluated. A usable UI and passing unit tests are not scientific validation.

## 10. Prompt Variation Experiment

The repository includes a research runner for comparing four prompt framings while holding each question's evidence context, model, and generation settings fixed. It records raw answers, creates human-review templates, calculates lexical and citation metrics, screens answer pairs for possible contradictions, and creates a plot. Semantic evidence support and actual contradictions require human review; no experimental results are included.

See [`experiments/README.md`](experiments/README.md) for the complete question, hypothesis, variables, dataset protocol, metrics, limitations, and annotation instructions. Dataset records follow [`experiments/dataset.schema.json`](experiments/dataset.schema.json).

After creating a curated JSONL dataset and configuring the `.env` file, run:

```bash
python -m research_analyzer.experiments run \
	--dataset experiments/my_dataset.jsonl \
	--output results/prompt_variation_run_001 \
	--repeats 3 \
	--temperature 0
```

Review and complete `human_annotations.csv` and `contradiction_pairs.csv` in the run folder, then recalculate the tables and plot:

```bash
python -m research_analyzer.experiments analyze \
	--results-dir results/prompt_variation_run_001
```

The output directory must not already exist. Model/API nondeterminism means that fixed seeds and settings improve traceability but do not guarantee identical outputs across time or providers.

## 11. Further Research

Next useful extensions include multi-annotator agreement, claim-level support judgments, calibrated abstention analysis, prompt-injection and robustness sets, and inference procedures appropriate to the dataset size. These are future work, not capabilities validated by the current prototype.
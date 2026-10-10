# Final Project Status

## Project summary

AI Research Evidence Analyzer is an exploratory, local-first RAG prototype for asking evidence-traceable questions about one research-paper PDF at a time. It is not a production system or a validated scientific instrument. The application work described here is separate from the completed pilot experiments and their reported findings.

## Implemented fixes

- Kept answer generation on local llama.cpp/Qwen2.5-1.5B-Instruct-Q4_K_M and Chroma's local ONNX embeddings/storage. Non-local inference backends are rejected; there is no hosted inference fallback.
- Improved selected-paper isolation, candidate retrieval, question-term reranking, Results/RQ passage priority, empty-collection handling, stale-chunk cleanup, and local prompt context checks.
- Corrected relation-to-metric mapping answers from explicit evidence rather than trusting a model's mapping. Contradictory or incomplete mappings abstain. When a PDF page break divides a sentence, only adjacent indexed chunks are combined, and citations retain the original page labels.
- Added direct local search and deterministic extraction for explicitly asked acronym definitions. A direct match is labelled as such rather than given an invented cosine distance. Ambiguous or absent definitions abstain.
- Tightened citation checks for unknown/malformed labels, missing citations, unsupported numbers, lexical support, and model-specific mappings. These checks are guardrails, not semantic entailment proofs.
- Kept PDF page numbers tied to PyMuPDF's one-based source pages and reported unsupported scanned/image-only PDFs.
- Updated the README with the local architecture, setup, use, cost and privacy boundaries, experiment separation, and known limitations.

## Verified functionality

- Complete automated suite after the follow-up repairs: **70 passed, 0 failed**, with one ChromaDB package deprecation warning.
- Python compilation: passed for the app, review tools, package, tests, and experiment scripts.
- `git diff --check`: passed.
- Experiment 1 and Experiment 2 analysis CLIs both succeeded on temporary copies of their existing result directories. No experiment run was repeated and no original result directory was used as an output target.
- The current workspace contains one existing Chroma collection named for `Test paper 1.pdf`, with 37 stored chunks across source pages 1–8. The original PDF file itself was not present in the workspace or bounded upload/temp search paths.
- Using the existing indexed paper, local Chroma retrieval/reranking and local Qwen/llama.cpp inference were exercised:
  - RQ1 mapping: MR4 for sentiment and MR12 for tone. The indexed extraction splits the MR4 phrase at the page 6/7 boundary, so the answer cites both source passages (`[E2]` page 6 and `[E4]` page 7); page 7 is cited, and page 6 supplies the continuation context.
  - RQ2 mapping and count: MR4 for GPT-4.0 tone analysis and approximately 2,200 faults, cited to the direct page-7 passage (`[E4]`).
  - Acronym question: `MR stands for Metamorphic Relation. [E1]`, supported by the explicit page-1 expansion. This avoids the observed local-model error “Multiple Response.”
  - Test volume: the local answer reported 12 templates and 4,700 test cases from the retrieved paper passages.
  - Whether Claude and Gemini were evaluated: abstained for insufficient retrieved evidence and did not assert they were evaluated.
  - Unanswerable question: abstained for insufficient retrieved evidence.
- Fresh Streamlit process returned HTTP 200 on localhost; Streamlit AppTest executed the current source with no exception and displayed the initial PDF-upload prompt. The original PDF was unavailable, so this did not exercise upload or indexing through the UI.
- Deterministic tests cover mapping reversals, page-boundary continuation, Results priority, misleading/unsupported citations, unsupported mappings, legitimate abstention, acronym extraction, document scoping, PDF page handling, and experiment artifact hashes.

The existing-index query-to-answer path was verified against the paper's stored chunks. Uploading and re-indexing the original PDF through the browser, restart persistence after a new upload, and browser interaction were **not** verified because the original PDF was unavailable. The live-paper check is not a substitute for those checks.

## Validation commands and results

| Command/check | Result |
| --- | --- |
| `pytest -q` | 70 passed, 0 failed, 1 warning |
| `python -m compileall -q app.py review_app.py review_experiment2.py src tests experiments` | Passed |
| `git diff --check` | Passed |
| `python -m research_analyzer.experiments analyze --results-dir <temporary-copy>` | Passed; generated metrics in the temporary copy |
| `python -m research_analyzer.experiment2 analyze --results-dir <temporary-copy>` | Passed; generated summary in the temporary copy |
| Real RQ1 query over the existing indexed paper | Correct MR4/sentiment and MR12/tone mapping; cites pages 6 and 7 |
| Real RQ2 query over the existing indexed paper | Correct MR4/GPT-4.0/tone mapping and approximately 2,200 faults; cites page 7 |
| Real local retrieval/answer checks for definition, template/test-case counts, unsupported Claude/Gemini assertion, and unanswerable question | Explicit page-1 MR definition; 12 templates and 4,700 cases; abstained where retrieved evidence was insufficient |
| Fresh Streamlit startup and empty-upload UI state | Local HTTP 200; AppTest had no exception and displayed the upload prompt |

The sole test warning is from the installed ChromaDB package's use of deprecated `asyncio.iscoroutinefunction`; anonymized telemetry is explicitly disabled in the application. ONNX Runtime printed a hardware-discovery warning during live retrieval. Streamlit AppTest printed its usual missing-`ScriptRunContext` warning in bare mode. These warnings did not prevent their checks from completing.

## Remaining limitations and unverified areas

- A 1.5B local model can still misunderstand questions or produce plausible but unsupported paraphrases outside deterministic handling. Citation presence does not establish semantic correctness.
- Embedding distances, overlap ranking, section recognition, and the user threshold remain heuristics. Retrieval may miss relevant passages.
- PDF extraction and chunk boundaries can lose layout or context. The app does not provide OCR for scanned PDFs.
- The source PDF was unavailable, so PDF upload/re-index, browser interaction, and a fresh upload followed by application restart were not end-to-end tested in this pass.
- The existing-index answer flow is verified, but a newly uploaded PDF's extraction and indexing were not exercised end-to-end here; PyMuPDF extraction, chunking, stale-chunk cleanup, and duplicate indexing have automated tests.
- Paper-library import/edit/delete, cross-paper comparison, research-gap/literature-review generation, and in-app experiment planning are not part of the current Streamlit analysis UI.
- The existing research evaluations are limited pilots with finite datasets and human-annotation scope; they do not establish general superiority or statistical significance.
- The runtime is configured for local inference, local embeddings, and local persistence. Initial model/embedding downloads require network access; no hosted inference endpoint or paid API is supported.
- No configured formatter or linter was found; compilation, tests, and the Git whitespace check were run.

## Files changed during the implementation pass

- `app.py`
- `README.md`
- `FINAL_PROJECT_STATUS.md`
- `FINAL_AUDIT_REPORT.md`
- `methodology.md`
- `src/research_analyzer/answering.py`
- `src/research_analyzer/definitions.py`
- `src/research_analyzer/experiment2.py`
- `src/research_analyzer/experiments.py`
- `src/research_analyzer/pipeline.py`
- `src/research_analyzer/vector_store.py`
- `experiments/smoke_local_model.py`
- `requirements.txt`
- `tests/test_answering.py`
- `tests/test_experiments.py`
- `tests/test_local_inference.py`
- `tests/test_pdf_processor.py`
- `tests/test_pipeline.py`

The repository already had many uncommitted source, documentation, and research-artifact changes when this work began. Those were preserved; this list is not a claim that every pre-existing worktree change originated in this final pass.

## Research artifact preservation and cost

No original experiment was rerun. Raw outputs, human annotations, manifests, reports, and other historical research artifacts were left intact. The complete test suite's hash checks passed, and both analysis CLIs wrote only to temporary copies.

The documented workflow requires **no paid service, hosted API, subscription, or API key**. It uses local CPU Qwen/llama.cpp inference and local Chroma ONNX embeddings/storage. Fresh installations may need free one-time downloads for the model and embedding runtime/assets. No paid integration or external inference call was added.

# AI Research Evidence Analyzer: Combined Pilot Study Report

## Executive summary

This report describes two completed, local-model pilot experiments using one fixed set of ten research questions and their evidence passages. Experiment 1 compares four prompt framings for evidence-grounded question answering. Experiment 2 compares control evidence with evidence containing direct and subtle synthetic instruction-injection attempts. Both runs used the same local Qwen2.5-1.5B-Instruct Q4_K_M model on CPU. No paid APIs, hosted inference, or cloud compute were used.

The existing runs contain 120/120 successful Experiment 1 model calls, 120 completed answer annotations, and 180 completed contradiction-pair reviews; and 90/90 successful Experiment 2 calls, with 90 completed human rating rows. All results here are descriptive. This is a **pilot study**, not a publication-ready or security-certification study. No causal, significance, or broad-generalization claims are made.

## Research questions and motivation

### Experiment 1: prompt framing

**Question:** How do four evidence-answering prompt framings relate to consistency and human-rated evidence support, correctness, uncertainty, and contradiction for research-question answers?

The motivation is that fluent answers can be weakly supported or inconsistent. This experiment holds a question's evidence context constant while varying the additional instruction, allowing a small descriptive comparison of answer style and human judgments.

### Experiment 2: evidence-context injection

**Question:** In this fixed evidence-grounded answering setup, how do human-rated adherence, evidence support, injection resistance, correctness, uncertainty, and contradiction appear when the evidence context contains direct or subtle synthetic redirects?

The injected text is inert, fictional test content intended to redirect answers to the unrelated statement “the study reports blue bicycles.” It contains no executable instructions, credentials, or external actions. This is a narrow robustness probe, not a comprehensive prompt-injection benchmark.

## Shared dataset and evidence construction

The dataset is [`data/pilot_dataset.jsonl`](../data/pilot_dataset.jsonl), containing ten questions associated with ten real arXiv v1 papers. It includes source metadata, question type and answerability, fixed evidence text, page numbers, relevance rationales, and source-document SHA-256 values. The dataset SHA-256 is `a2fd0f730ce2396b3c031e370240e7a30663720fae78f89ada00b075e64ae4ea`. The JSON Schema is [`experiments/dataset.schema.json`](../experiments/dataset.schema.json); source/PDF verification is provided by [`experiments/verify_pilot_dataset.py`](../experiments/verify_pilot_dataset.py).

The question/evidence dataset is the same in both experiments. It contains answerable and deliberately unanswerable items. In Experiment 2, every raw output was checked against the dataset: the question and answerability are unchanged; every legitimate evidence passage matches the original; control contains no injected passage; and each injection condition contains exactly one separate synthetic passage. Injected text is absent from the system prompt and appears in the user evidence context only for the intended injection condition.

## Shared model, inference settings, and environment

- Model: **Qwen2.5-1.5B-Instruct Q4_K_M**, GGUF, revision `9eadc66189c7641e1ddd226b8267a9119b2ce2d4`.
- Model source: `bartowski/Qwen2.5-1.5B-Instruct-GGUF`; repository-relative model path: `models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`.
- Runtime: `llama-cpp-python` 0.3.35 / llama.cpp, local CPU inference; no CUDA, zero GPU layers.
- Settings: temperature 0, top-p 1.0, seed 17, maximum 256 generated tokens, context size 4096, two CPU threads.
- Recorded host: Linux x86_64, AMD EPYC 7763, two visible CPUs, about 7.8 GiB total memory, no CUDA GPU.

The full per-run hardware and inference settings are recorded in each result directory's `manifest.json` and raw rows. The path above is relative to the repository; no machine-specific absolute model path or credential is required. See the root [`README.md`](../README.md) and [`methodology.md`](../methodology.md) for setup and environment-variable details.

## Experiment 1: prompt variation

### Design and procedure

The independent variable was one of four prompt conditions. The same question, ordered evidence, common evidence-only policy, model, and inference settings were used within each question; the additional instruction differed:

| Condition | Added instruction |
| --- | --- |
| `baseline` | Answer directly and concisely; cite factual statements. |
| `citation_focus` | Cite factual claims and omit unsupported claims. |
| `uncertainty_focus` | Distinguish established findings from uncertainty; prefer abstention over guessing. |
| `structured` | Use Answer, Evidence, and Limitations headings. |

There were ten questions × four conditions × three repetitions = **120 planned calls**, all successful. The separate smoke call is excluded. Human review includes one completed answer-rating row for each of the 120 outputs and one completed review for each of 180 unordered condition-pair comparisons (30 per pair of conditions).

Human answer labels were evidence support (Yes/Partial/No), correctness (Yes/Partial/No), appropriate uncertainty (Yes/No/Not applicable), and contradiction with evidence (Yes/No). Pair reviewers decided whether the two answers contradicted one another in light of the shared evidence. Lexical similarity and citation-label checks are automated proxies only. Full annotation guidance is in [`methodology.md`](../methodology.md).

### Results

All condition percentages below use 30 annotations per condition unless the uncertainty row explicitly gives the applicable-rating denominator. Missing ratings: **0/30 in every condition and measure**.

| Condition | Support Yes / Partial / No | Correctness Yes / Partial / No | Appropriate uncertainty Yes / No (rated denominator); N/A | Contradiction with evidence Yes / No |
| --- | --- | --- | --- | --- |
| baseline | 30/30 (100%) / 0 / 0 | 30/30 (100%) / 0 / 0 | 10/12 (83.3%) / 2/12 (16.7%); N/A 18/30 (60.0%) | 0/30 (0%) / 30/30 (100%) |
| citation_focus | 30/30 (100%) / 0 / 0 | 30/30 (100%) / 0 / 0 | 9/11 (81.8%) / 2/11 (18.2%); N/A 19/30 (63.3%) | 0/30 (0%) / 30/30 (100%) |
| uncertainty_focus | 30/30 (100%) / 0 / 0 | 30/30 (100%) / 0 / 0 | 8/11 (72.7%) / 3/11 (27.3%); N/A 19/30 (63.3%) | 0/30 (0%) / 30/30 (100%) |
| structured | 29/30 (96.7%) / 0 / 1/30 (3.3%) | 29/30 (96.7%) / 0 / 1/30 (3.3%) | 8/20 (40.0%) / 12/20 (60.0%); N/A 10/30 (33.3%) | 13/30 (43.3%) / 17/30 (56.7%) |

The three unanswerable items account for nine responses in each condition. For these, appropriate uncertainty was Yes on baseline 9/9, citation_focus 9/9, uncertainty_focus 8/9, and structured 8/9. Each condition had 21 answerable responses; support and correctness were Yes for all 21 in the first three conditions and 20/21 in structured.

Human pair reviews found 8 contradictions among 180 reviewed pairs (4.4%). Counts by comparison were baseline–citation_focus 0/30, baseline–uncertainty_focus 0/30, baseline–structured 3/30 (10.0%), citation_focus–uncertainty_focus 0/30, citation_focus–structured 2/30 (6.7%), and uncertainty_focus–structured 3/30 (10.0%).

For automated descriptive measures, within-condition repeat word-set Jaccard was 1.000 for baseline, 1.000 for citation_focus, 0.983 for uncertainty_focus, and 1.000 for structured. Between-condition Jaccard was respectively 0.676, 0.664, 0.566, and 0.480. Citation presence was 0/30 in the first three conditions and 12/30 (40.0%) in structured; the 15/15 citation occurrences in structured used valid evidence labels. These lexical and label-validity values do not measure semantic equivalence or entailment.

Detailed Experiment 1 tables and provenance are in [`experiment_1_local_pilot_report.md`](experiment_1_local_pilot_report.md), [`final_condition_summary.csv`](exp1-local-qwen25-1.5b-q4-20260927/final_condition_summary.csv), [`metrics.csv`](exp1-local-qwen25-1.5b-q4-20260927/metrics.csv), and [`metrics.png`](exp1-local-qwen25-1.5b-q4-20260927/metrics.png).

## Experiment 2: evidence-context injection

### Design and procedure

Experiment 2 used the same ten questions and legitimate evidence passages, three conditions, and three repetitions: **90 planned calls**, all successful (30 per condition).

| Condition | Evidence context |
| --- | --- |
| `control` | Original legitimate evidence only. |
| `direct_injection` | Original evidence plus one explicitly adversarial redirect passage. |
| `subtle_injection` | Original evidence plus one editorial-style redirect passage. |

The question, legitimate evidence, system prompt, model, and settings were held fixed. Only the evidence-context condition changed. Annotation fields are task adherence, evidence support, injection resistance, correctness, appropriate uncertainty, and contradiction with legitimate evidence. Label definitions and reviewer procedures are in [`experiments/experiment_2.md`](../experiments/experiment_2.md) and [`experiments/experiment_2_annotation.schema.json`](../experiments/experiment_2_annotation.schema.json). Ratings below are the existing human labels; no rating was inferred or rewritten during analysis.

### Results

Every condition has 30 completed annotations. For Yes/Partial/No items the denominator is 30; all percentages are descriptive. Missing ratings are **0/30 for every measure and condition**. “Not applicable” remains a distinct valid response, not a missing rating.

| Condition | Task adherence Yes / Partial / No (n/30; %) | Evidence support Yes / Partial / No (n/30; %) | Correctness Yes / Partial / No (n/30; %) |
| --- | --- | --- | --- |
| control | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) |
| direct_injection | 30 (100%) / 0 (0%) / 0 (0%) | 15 (50%) / 0 (0%) / 15 (50%) | 15 (50%) / 0 (0%) / 15 (50%) |
| subtle_injection | 30 (100%) / 0 (0%) / 0 (0%) | 27 (90%) / 0 (0%) / 3 (10%) | 27 (90%) / 0 (0%) / 3 (10%) |

| Condition | Injection resistance | Appropriate uncertainty | Contradiction with legitimate evidence |
| --- | --- | --- | --- |
| control | Resistant 15/30 (50%); Partial 0/30 (0%); Compromised 0/30 (0%); Not applicable 15/30 (50%) | Yes 8/8 applicable (100%); No 0/8 (0%); Not applicable 22/30 (73.3%) | Yes 0/30 (0%); No 30/30 (100%) |
| direct_injection | Resistant 30/30 (100%); Partial 0/30 (0%); Compromised 0/30 (0%); Not applicable 0/30 (0%) | Yes 9/24 applicable (37.5%); No 15/24 (62.5%); Not applicable 6/30 (20.0%) | Yes 15/30 (50%); No 15/30 (50%) |
| subtle_injection | Resistant 30/30 (100%); Partial 0/30 (0%); Compromised 0/30 (0%); Not applicable 0/30 (0%) | Yes 8/11 applicable (72.7%); No 3/11 (27.3%); Not applicable 19/30 (63.3%) | Yes 3/30 (10.0%); No 27/30 (90.0%) |

**Control resistance-label caveat:** The annotation schema permits `Resistant`, but its description directs reviewers to use `Not applicable` for controls. The existing control annotations contain 15 `Resistant` and 15 `Not applicable` values. All 30 are enum-valid and were preserved exactly; the 15 `Resistant` control ratings are reported verbatim but must not be interpreted as evidence of resistance to an injection that was not present. This is a protocol/schema consistency limitation for any future annotation round.

The table shows why injection resistance must not be conflated with correctness or evidence support: the direct-injection rows were all rated Resistant, while 15/30 were also rated No for support and correctness and Yes for contradiction with legitimate evidence. These are distinct recorded human measures, and no one measure is substituted for another.

Machine-readable results with all rating categories, denominators, percentages, and explicit zero-count Missing rows are in [`final_condition_summary.csv`](exp2-evidence-injection-qwen25-1.5b-q4/final_condition_summary.csv). Call counts are in [`call_summary.csv`](exp2-evidence-injection-qwen25-1.5b-q4/call_summary.csv). Raw response and annotation integrity hashes are recorded in the run and review manifests.

## Validation, integrity, and analysis

- Experiment 1: 120 successful outputs; 120 unique completed answer annotations; 180 unique completed pair reviews. The raw-output SHA-256 is `a729efa318361f8c18c8b8dbba2caa1e46d5ee77c8be90cbf72f5a32fc30d4bb`; the human annotation SHA-256 is `7dd33bbc3f998987258c539b64f72f9438a6c02e9d94d666c52c4a19bd698da9`; and the contradiction-review SHA-256 is `f593d7d8ca20b591374fabb01e39e10f6428c48f6fdd6429b2a4f90a4d070537`. The raw-output hash matches the human-review manifest; all three match the repository's recorded integrity-test values. The dataset hash matches the run manifest.
- Experiment 2: 90 successful outputs (30 in each condition), 90 unique completed annotations, all valid under the checked-in annotation schema. The raw-output SHA-256 is `c47339ba4fdfe50629cdafb0f88ec5f64b0f6aeacc0990a358901aa6822d4c78`; annotation CSV SHA-256 is `db54fa64d7f91b950f6ed2a08bfc1687d963e8df56af1182d6ab13eb3c2daddb`. Both match the completed review manifest.
- The Experiment 2 analysis command verifies the run raw hash, review raw and annotation hashes, completed review status/counts, one-to-one annotation mapping, 90 successes, and 30 calls per condition before writing. It writes only `final_condition_summary.csv`; it does not modify raw outputs or annotations.

To regenerate the Experiment 2 summary from the existing reviewed annotations without model calls:

```bash
python -m research_analyzer.experiment2 analyze \
  --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
```

Experiment 1's existing tables can be regenerated from its annotations and raw outputs without model calls:

```bash
python -m research_analyzer.experiments analyze \
  --results-dir results/exp1-local-qwen25-1.5b-q4-20260927
```

Do **not** use either `run` command to reproduce these analyses; those commands make new model calls. The analysis commands above do not invoke the model.

## Limitations and threats to validity

- Ten questions from ten papers form a small convenience pilot; questions, topics, and papers are not a representative random sample.
- Both experiments use one quantized 1.5B model, one revision, one CPU environment, and one inference setup. Findings do not transfer automatically to other models or providers.
- There are only three repeated outputs per question-condition cell; repetitions are not independent questions.
- Experiment 1 changes prompt emphasis and formatting together. Experiment 2 tests only two fixed synthetic attacks. Neither design supports broad causal attribution.
- Evidence was fixed manually; neither experiment evaluates retrieval quality or the full PDF-ingestion pipeline.
- The human-review artifacts store one final rating per item without independent duplicate annotator ratings or adjudication records. Inter-annotator agreement and rating uncertainty cannot be estimated.
- Experiment 2's control `injection_resistance` labels include schema-valid values that conflict with the schema's control-condition guidance, as detailed above. Ratings were not altered.
- `Not applicable` is frequent for uncertainty and is reported separately from the applicable-rating denominator; a raw percentage over 30 should not be read as a conditional uncertainty rate.
- Automated lexical similarity and citation-label validity do not establish semantic consistency, factual accuracy, support, or contradiction.
- No statistical significance tests, confidence intervals, or confirmatory estimates are justified or reported.

## Responsible interpretation and future work

Treat the numbers as descriptions of these recorded pilot responses and annotations only. They are not evidence of general prompt-injection safety, model reliability, or a winning prompt. In particular, the mixed direct-injection ratings across resistance, support, correctness, and contradiction are preserved as recorded and should motivate reviewing construct definitions, not recoding outcomes after seeing them.

Before publication-quality claims, future work should preregister hypotheses, sample size, analysis, and exclusion rules; expand and stratify the question/paper/attack set; recruit multiple independent blinded annotators with adjudication and agreement reporting; clarify the operational distinction between injection resistance and answer correctness; revise the schema so control-only applicability is enforced; and replicate across models and environments. Retrieval behavior should be tested separately from the fixed-evidence question-answering setup.

## Cost and project status

Both experiments and this analysis used local model results only. No paid API, credits, subscription, cloud inference, or cloud compute was used. Total task cost was **$0 / ₹0**.

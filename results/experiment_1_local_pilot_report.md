# Experiment 1: Final Descriptive Results

## Scope and completion

This is an analysis of the existing Experiment 1 run only. No model calls were made during this analysis. The run contains 10 questions, four prompt conditions, and three repetitions per question and condition: **120/120 successful model calls**. Human review is complete for **120/120 answer annotations** and **180/180 contradiction-pair reviews**.

Validation found exactly 120 unique answer-review rows matching the 120 successful raw outputs and exactly 180 unique pair-review rows. All required answer ratings and pair decisions are present and conform to the schema in the review workflow. Optional notes are not treated as required labels. `raw_outputs.jsonl` remains unchanged; its SHA-256 matches the review manifest: `a729efa318361f8c18c8b8dbba2caa1e46d5ee77c8be90cbf72f5a32fc30d4bb`.

The condition tables below use the manifest order, not a ranking. Percentages are descriptive and are accompanied by their denominators. The CSV [`final_condition_summary.csv`](exp1-local-qwen25-1.5b-q4-20260927/final_condition_summary.csv) provides the corresponding counts, denominators, percentages, and metric units in machine-readable form. The refreshed stratified visualization is [`metrics.png`](exp1-local-qwen25-1.5b-q4-20260927/metrics.png); the underlying answerability-stratified values are in [`metrics.csv`](exp1-local-qwen25-1.5b-q4-20260927/metrics.csv).

## A. Automated descriptive metrics

| Prompt condition | Successful calls (n/N) | Within-condition repeat Jaccard (30 repeat pairs) | Between-condition Jaccard (90 comparisons) | Responses with citation (n/30) | Valid cited responses (n/30) | Valid citation labels |
|---|---:|---:|---:|---:|---:|---:|
| baseline | 30/30 (100%) | 1.000 | 0.676 | 0/30 (0%) | 0/30 (0%) | N/A (no citations) |
| citation_focus | 30/30 (100%) | 1.000 | 0.664 | 0/30 (0%) | 0/30 (0%) | N/A (no citations) |
| uncertainty_focus | 30/30 (100%) | 0.983 | 0.566 | 0/30 (0%) | 0/30 (0%) | N/A (no citations) |
| structured | 30/30 (100%) | 1.000 | 0.480 | 12/30 (40.0%) | 12/30 (40.0%) | 15/15 citation occurrences (100%) |

Within-condition Jaccard is the mean word-set overlap for repeat pairs answering the same question under the same condition. Between-condition Jaccard is the mean overlap with other conditions for the same question and repetition. Both are lexical proxies only: overlap does not demonstrate semantic equivalence, correctness, or absence of contradiction. Citation-label validity checks whether a cited label existed in the supplied evidence; it does not test whether a claim is supported.

## B. Human-evaluation results

Each condition has 30 annotated answers: 21 answerable and 9 unanswerable. Evidence support, correctness, and contradiction-with-evidence percentages use all 30 annotations per condition. Appropriate-uncertainty Yes/No percentages use only ratings where uncertainty was applicable; Not applicable is reported separately over all 30.

| Prompt condition | Evidence support (Yes / Partial / No; n/30) | Answer correctness (Yes / Partial / No; n/30) | Appropriate uncertainty (Yes / No; rated denominator) | Not applicable (n/30) | Contradiction with evidence (Yes / No; n/30) |
|---|---|---|---:|---:|---|
| baseline | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) | 10/12 (83.3%) / 2/12 (16.7%) | 18 (60.0%) | 0 (0%) / 30 (100%) |
| citation_focus | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) | 9/11 (81.8%) / 2/11 (18.2%) | 19 (63.3%) | 0 (0%) / 30 (100%) |
| uncertainty_focus | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) | 8/11 (72.7%) / 3/11 (27.3%) | 19 (63.3%) | 0 (0%) / 30 (100%) |
| structured | 29 (96.7%) / 0 (0%) / 1 (3.3%) | 29 (96.7%) / 0 (0%) / 1 (3.3%) | 8/20 (40.0%) / 12/20 (60.0%) | 10 (33.3%) | 13 (43.3%) / 17 (56.7%) |

On the nine unanswerable answers per condition, appropriate uncertainty was rated Yes for baseline 9/9 (100%), citation_focus 9/9 (100%), uncertainty_focus 8/9 (88.9%), and structured 8/9 (88.9%); the remaining ratings in the latter two conditions were No. Support and correctness were Yes for all 21 answerable answers in each condition. For unanswerable answers, support and correctness were Yes for 9/9 in baseline, citation_focus, and uncertainty_focus, and 8/9 in structured.

### Contradiction-pair reviews

Each unordered prompt-condition pair has 30 pair reviews (same question and repetition); percentages use that denominator. Across all pairs, 8/180 (4.4%) were marked as contradictions and 172/180 (95.6%) were marked as non-contradictions.

| Conditions compared | Contradiction (n/30) | No contradiction (n/30) |
|---|---:|---:|
| baseline vs citation_focus | 0 (0%) | 30 (100%) |
| baseline vs uncertainty_focus | 0 (0%) | 30 (100%) |
| baseline vs structured | 3 (10.0%) | 27 (90.0%) |
| citation_focus vs uncertainty_focus | 0 (0%) | 30 (100%) |
| citation_focus vs structured | 2 (6.7%) | 28 (93.3%) |
| uncertainty_focus vs structured | 3 (10.0%) | 27 (90.0%) |

These are human decisions about contradiction between answers in light of the shared evidence. They are distinct from the per-answer contradiction-with-evidence ratings above and from the automated lexical candidate flag.

### Descriptive rating consistency across repetitions

For each of 40 question-by-condition groups, compare the three separately annotated repetitions. The three ratings were unanimous in 39/40 groups (97.5%) for evidence support, 39/40 (97.5%) for correctness, 31/40 (77.5%) for appropriate uncertainty, and 39/40 (97.5%) for contradiction with evidence. This is within-group consistency across repeated responses, **not inter-annotator agreement** and not an inferential statistic.

## C. Interpretation and limitations

In this pilot, all 120 model calls succeeded. Observed within-condition word overlap was high across conditions, while between-condition overlap varied. Only structured-condition responses contained citations (12/30); all 15 cited labels were present in the supplied evidence. Human ratings were Yes for support and correctness in 119/120 answers overall. The structured condition accounted for all 13 Yes ratings for contradiction with evidence (13/30 in that condition), while pair reviewers confirmed 8 contradictions among 180 reviewed pairs. Appropriate-uncertainty ratings differed descriptively, with many answers marked Not applicable; therefore Yes/No rates use condition-specific rated denominators. These observed patterns do not establish that prompt framing caused the differences, and no condition is selected as a winner.

Limitations:

- The dataset contains only 10 questions from 10 papers, a pilot-scale convenience sample.
- The experiment tested one local 1.5B quantized model, Qwen2.5-1.5B-Instruct Q4_K_M.
- There were three repetitions per question and condition; repetitions are not independent questions.
- Human evaluation is limited to the 120 recorded answer labels and 180 recorded pair decisions. The artifacts store one final label per item and do not include independently duplicated ratings or adjudication records, so inter-annotator agreement cannot be calculated.
- Evidence was fixed for each question; this does not evaluate retrieval quality. Prompt instructions differ in emphasis/formatting together.
- Lexical similarity and citation-label checks are limited automated proxies.
- No broad generalization to other questions, models, settings, or prompt formulations is supported.
- No statistical significance tests or inferential claims are made; this pilot is not a basis for publication-level conclusions.

## Recommended next research step

Design and preregister a larger, stratified replication, including its sample-size and analysis plan in advance, and obtain at least two independent blinded human ratings per answer and pair. That would address the small question set and provide an estimate of inter-annotator agreement before drawing broader conclusions.

## Provenance and reproducibility

The source files `human_annotations.csv`, `contradiction_pairs.csv`, `raw_outputs.jsonl`, and `manifest.json` were read but not edited. The raw-output SHA-256 is recorded above and remains the value in `human_review_manifest.json`. Automated `metrics.csv` and `metrics.png` were regenerated using the existing analysis command:

```bash
python -m research_analyzer.experiments analyze --results-dir results/exp1-local-qwen25-1.5b-q4-20260927
```

No experiment or model call was run.

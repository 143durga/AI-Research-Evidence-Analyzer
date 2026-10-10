# FINAL AUDIT REPORT

## Overall status

**Verified with caveats.** Existing result files, completed annotation rows, analyses, documentation, and requested technical checks were inspected. No experiment was rerun and no code, experiment output, or annotation was changed. Only this report was added. Human authorship cannot be independently authenticated from stored CSVs; “completed” below means the recorded labels are present, valid, mapped to outputs, and consistent with review manifests.

## Experiment 1 verification

- `results/exp1-local-qwen25-1.5b-q4-20260927/raw_outputs.jsonl`: **120/120** records, 120 unique IDs, all successful; 30 outputs per each of four conditions.
- `human_annotations.csv`: **120/120** unique annotations complete for all required rating fields.
- `contradiction_pairs.csv`: **180/180** reviews complete with unique pairs and valid reviewed decisions.
- Raw SHA-256: `a729efa318361f8c18c8b8dbba2caa1e46d5ee77c8be90cbf72f5a32fc30d4bb`; matches `human_review_manifest.json`. The run `manifest.json` has no raw-output hash field. Annotation and pair CSV SHA-256 values are respectively `7dd33bbc3f998987258c539b64f72f9438a6c02e9d94d666c52c4a19bd698da9` and `f593d7d8ca20b591374fabb01e39e10f6428c48f6fdd6429b2a4f90a4d070537`; both match the repository’s recorded integrity-test values.
- Dataset hash matches the run manifest.

## Experiment 2 verification

- `results/exp2-evidence-injection-qwen25-1.5b-q4/raw_outputs.jsonl`: **90/90** records, 90 unique IDs, all successful; control **30**, direct injection **30**, subtle injection **30**.
- `human_annotations.csv`: **90/90** unique rows complete; all six rating fields pass the code validator and all 90 rows pass `experiments/experiment_2_annotation.schema.json` (zero schema errors). Review manifest status is `completed`, reviewed/required **90/90**.
- Raw SHA-256 `c47339ba4fdfe50629cdafb0f88ec5f64b0f6aeacc0990a358901aa6822d4c78` matches both run and review manifests. Annotation SHA-256 `db54fa64d7f91b950f6ed2a08bfc1687d963e8df56af1182d6ab13eb3c2daddb` matches the review manifest.
- Rechecked raw rows against `data/pilot_dataset.jsonl`: questions, answerability, and legitimate evidence match; control has no injected passage; each injection-condition row has one separate injected passage. No invariant failures found. These hashes confirm the outputs have not changed since the recorded manifests; no earlier independent backup was available for comparison.

## Actual human evaluation results

Counts are `n/30` except uncertainty, where Yes/No percentages use only applicable rows. `Not applicable` and missing are separate; missing ratings are **0** in all conditions and measures.

| Condition | Task adherence Yes/Partial/No | Evidence support Yes/Partial/No | Injection resistance R/P/C/N/A | Correctness Yes/Partial/No | Uncertainty Yes/No; N/A | Contradiction Yes/No |
| --- | --- | --- | --- | --- | --- | --- |
| Control | 30/0/0 | 30/0/0 | 15/0/0/15 | 30/0/0 | 8/8, 0/8; 22/30 N/A | 0/30, 30/30 |
| Direct injection | 30/0/0 | 15/0/15 | 30/0/0/0 | 15/0/15 | 9/24, 15/24; 6/30 N/A | 15/30, 15/30 |
| Subtle injection | 30/0/0 | 27/0/3 | 30/0/0/0 | 27/0/3 | 8/11, 3/11; 19/30 N/A | 3/30, 27/30 |

Resistance column order is **Resistant / Partial / Compromised / Not applicable**. Other three-way columns are Yes / Partial / No.

For percentages, adherence was Yes 100% in all conditions; support and correctness Yes were control 100%, direct 50%, subtle 90%; contradiction Yes was 0%, 50%, and 10%. Applicable uncertainty Yes was 100%, 37.5%, and 72.7%, respectively. Recorded resistance labels were Resistant/Partial/Compromised/Not applicable: control 15/0/0/15, direct 30/0/0/0, subtle 30/0/0/0. **Caveat:** control annotations include 15 `Resistant` labels despite schema guidance to use `Not applicable` when no injection is present. Labels were valid enum values and were not changed; do not interpret those control labels as injection resistance. The differing direct-injection ratings across measures are reported as recorded, not reconciled or inferred.

## Tests and validation

- `pytest -q`: **28 passed, 0 failed**; one non-blocking ChromaDB deprecation warning (`asyncio.iscoroutinefunction`).
- `python -m py_compile` on application, reviewer, experiment, library, and dataset utility Python sources: **passed**, no errors.
- `git diff --check`: **passed**, no whitespace errors.
- No dependencies were installed.

## Missing items or blockers

- No required report, summary, schema, protocol, README, or reproduction command was missing. Experiment 2 analysis exists at `results/exp2-evidence-injection-qwen25-1.5b-q4/final_condition_summary.csv` (72 category rows, including explicit zero-count Missing categories).
- Experiment 1’s raw hash is recorded in the human-review manifest and reports, not the run manifest itself.
- Independent evidence of annotator identity/authorship is not stored, so authenticity cannot be externally proven by this audit; completeness and schema validity are verified.
- Experiment 2 control injection-resistance ratings conflict with the schema’s control guidance as described above. This limits interpretation, but existing ratings remain untouched.
- Findings remain a 10-question, one-model pilot with one stored rating per output; no independent duplicate ratings or adjudication are available.

## Reproduction commands

These commands analyze existing results only and make no model calls:

```bash
python -m research_analyzer.experiment2 analyze --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
python -m research_analyzer.experiments analyze --results-dir results/exp1-local-qwen25-1.5b-q4-20260927
```

Do not use either experiment’s `run` command to reproduce analysis.

## Cost confirmation

The audit snapshot noted an optional OpenAI-compatible path in the application. The subsequent final repair removed that inference path and dependency; current application and experiment code reject non-local inference configuration. This report's observed experiment data remains unchanged. The documented research analysis workflow uses local files and local Python tools; no paid API or service was needed or used for the audit. Total audit cost: **$0 / ₹0**.

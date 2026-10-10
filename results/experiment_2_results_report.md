# Experiment 2: Completed Human-Rating Results

Experiment 2 is complete as a **descriptive pilot**. The combined report, including both experiments' methods, findings, provenance, limitations, and analysis instructions, is [`final_research_report.md`](final_research_report.md).

## Completion and integrity

- 10 fixed questions × 3 conditions × 3 repetitions = **90/90 successful local model calls**, with 30 per condition.
- **90/90 completed human annotation rows**, all enum-valid under `experiments/experiment_2_annotation.schema.json`.
- The raw-output SHA-256 is `c47339ba4fdfe50629cdafb0f88ec5f64b0f6aeacc0990a358901aa6822d4c78`.
- The annotation CSV SHA-256 is `db54fa64d7f91b950f6ed2a08bfc1687d963e8df56af1182d6ab13eb3c2daddb`.
- Both hashes match `human_review_manifest.json`. The analysis verifies these provenance checks and writes only `final_condition_summary.csv`.
- The source questions and legitimate evidence were checked unchanged. Injection content is included as a distinct evidence-context passage only in direct/subtle conditions; no injected text is in the system prompt.
- Inference used Qwen2.5-1.5B-Instruct Q4_K_M at revision `9eadc66189c7641e1ddd226b8267a9119b2ce2d4`, local llama.cpp CPU, temperature 0, top-p 1.0, seed 17, 256 max tokens, context 4096, two CPU threads, and zero GPU layers.

## Denominator-explicit results

| Condition | Task adherence Yes / Partial / No (n/30; %) | Evidence support Yes / Partial / No (n/30; %) | Correctness Yes / Partial / No (n/30; %) | Injection resistance (n/30; %) |
| --- | --- | --- | --- | --- |
| control | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) | 30 (100%) / 0 (0%) / 0 (0%) | Resistant 15 (50%); Partial 0 (0%); Compromised 0 (0%); Not applicable 15 (50%) |
| direct_injection | 30 (100%) / 0 (0%) / 0 (0%) | 15 (50%) / 0 (0%) / 15 (50%) | 15 (50%) / 0 (0%) / 15 (50%) | Resistant 30 (100%); Partial 0 (0%); Compromised 0 (0%); Not applicable 0 (0%) |
| subtle_injection | 30 (100%) / 0 (0%) / 0 (0%) | 27 (90%) / 0 (0%) / 3 (10%) | 27 (90%) / 0 (0%) / 3 (10%) | Resistant 30 (100%); Partial 0 (0%); Compromised 0 (0%); Not applicable 0 (0%) |

Appropriate uncertainty used Yes/No denominators only for applicable ratings: control Yes 8/8 (100%), No 0/8 (0%), Not applicable 22/30 (73.3%); direct injection Yes 9/24 (37.5%), No 15/24 (62.5%), Not applicable 6/30 (20.0%); subtle injection Yes 8/11 (72.7%), No 3/11 (27.3%), Not applicable 19/30 (63.3%). Contradiction with legitimate evidence was Yes/No: control 0/30 (0%) / 30/30 (100%); direct injection 15/30 (50%) / 15/30 (50%); subtle injection 3/30 (10%) / 27/30 (90%).

There are **0 missing ratings** in each measure and condition. `Not applicable` is recorded distinctly; CSV percentages use all 30 rows as denominator, while the uncertainty Yes/No rates above use applicable rows. The machine-readable summary also contains explicit `Missing` rows with zero counts for each metric-condition combination.

### Control resistance-label caveat

The schema allows `Resistant` but directs reviewers to use `Not applicable` for controls. The completed records contain 15 `Resistant` and 15 `Not applicable` control labels. All are enum-valid, and they have been preserved without recoding. The 15 control `Resistant` labels are reported verbatim and must not be interpreted as evidence of resistance to an absent attack.

The distinct human measures also diverge in the direct-injection condition: 30/30 were labeled Resistant, while 15/30 were labeled No for evidence support and correctness and Yes for contradiction. These recorded labels are not interchangeable, and the operational meaning warrants review in future work.

## Result files and analysis

- Raw responses: [`exp2-evidence-injection-qwen25-1.5b-q4/raw_outputs.jsonl`](exp2-evidence-injection-qwen25-1.5b-q4/raw_outputs.jsonl)
- Human annotations: [`exp2-evidence-injection-qwen25-1.5b-q4/human_annotations.csv`](exp2-evidence-injection-qwen25-1.5b-q4/human_annotations.csv)
- Run/review provenance: [`exp2-evidence-injection-qwen25-1.5b-q4/manifest.json`](exp2-evidence-injection-qwen25-1.5b-q4/manifest.json), [`exp2-evidence-injection-qwen25-1.5b-q4/human_review_manifest.json`](exp2-evidence-injection-qwen25-1.5b-q4/human_review_manifest.json)
- Per-condition call counts: [`exp2-evidence-injection-qwen25-1.5b-q4/call_summary.csv`](exp2-evidence-injection-qwen25-1.5b-q4/call_summary.csv)
- Human rating summary: [`exp2-evidence-injection-qwen25-1.5b-q4/final_condition_summary.csv`](exp2-evidence-injection-qwen25-1.5b-q4/final_condition_summary.csv)

Regenerate only the analysis summary, without model calls:

```bash
python -m research_analyzer.experiment2 analyze \
  --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
```

This is not evidence of general model robustness, and no causal, inferential, or publication-quality claim is made. See the combined report for limitations and future-work requirements.

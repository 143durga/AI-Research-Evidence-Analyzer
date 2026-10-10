# Experiment 1 Files

The active, source-verified pilot protocol is documented in [`../methodology.md`](../methodology.md), and execution status is in [`../results/experiment_1_local_pilot_report.md`](../results/experiment_1_local_pilot_report.md). This folder contains the dataset schema and source verifier:

- `dataset.schema.json` defines each JSONL question, bibliographic record, and source-verified evidence passage.
- `verify_pilot_dataset.py` checks the schema, arXiv v1 PDF hashes, source pages, and exact passage text.
- `src/research_analyzer/experiments.py` (in the project source tree) runs the four prompt conditions and writes raw calls, blinded human annotation templates, contradiction-review pairs, metrics, and plots.

The pilot data is [`../data/pilot_dataset.jsonl`](../data/pilot_dataset.jsonl). It contains ten questions from ten papers. With four conditions and three repetitions, Experiment 1 used local Qwen2.5-1.5B-Instruct Q4_K_M on CPU with llama.cpp; the smoke inference and all 120 study calls succeeded. Experiment 1 human review is complete; see [`../results/experiment_1_local_pilot_report.md`](../results/experiment_1_local_pilot_report.md). Fake-generator runs are tests only.

```bash
python experiments/verify_pilot_dataset.py
python -m research_analyzer.experiments run \
  --dataset data/pilot_dataset.jsonl \
  --output results/exp1-prompt-variation-pilot-20260927 \
  --repeats 3 --temperature 0 --seed 17 \
  --experiment-id exp1-prompt-variation-pilot-20260927
```

After filling the human annotation and contradiction fields, update the reports:

```bash
python -m research_analyzer.experiments analyze \
  --results-dir results/exp1-prompt-variation-pilot-20260927
```

Review labels, metric definitions, source selection, and limitations in the methodology before interpreting results. The runner reports lexical and citation proxies; it does not automatically determine semantic support, answer correctness, or contradiction.

## Experiment 2

The controlled evidence-context injection pilot completed 90/90 local calls and has 90/90 completed human ratings. See [`experiment_2.md`](experiment_2.md), the [combined pilot report](../results/final_research_report.md), and the [Experiment 2 result summary](../results/experiment_2_results_report.md). The existing analysis can be regenerated without model calls:

```bash
python -m research_analyzer.experiment2 analyze \
  --results-dir results/exp2-evidence-injection-qwen25-1.5b-q4
```

For future review workflows, `streamlit run review_experiment2.py` opens the reviewer interface. Do not rerun the experiment to reproduce its analysis.
# Experiment 2

The controlled evidence-context injection robustness pilot is documented in
[`experiment_2.md`](experiment_2.md). Prepare the plan and run one smoke test
before the 90-call experiment; the full run is never started automatically.
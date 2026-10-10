"""Local, blinded human-review interface for the completed Experiment 1 run."""

from pathlib import Path

import streamlit as st

from research_analyzer.human_review import (
    ANSWER_FIELDS,
    ANSWER_RATINGS,
    PAIR_RATINGS,
    export_csv,
    initialize_workflow_manifest,
    is_answer_annotation_complete,
    is_pair_annotation_complete,
    load_review_bundle,
    save_answer_annotation,
    save_pair_annotation,
)

DEFAULT_RESULTS = Path("results/exp1-local-qwen25-1.5b-q4-20260927")
st.set_page_config(page_title="Experiment 1 Human Review", layout="wide")
st.title("Experiment 1: Human Evaluation")
st.caption("Local annotation workspace. Prompt conditions are not shown to reviewers.")
st.caption("Raw outputs are read-only and checked by SHA-256 before every save.")

try:
    bundle = load_review_bundle(DEFAULT_RESULTS)
    initialize_workflow_manifest(bundle)
except Exception as exc:
    st.error(f"Could not load this experiment: {exc}")
    st.stop()

st.caption(f"Reviewer workflow v{bundle['workflow_manifest']['workflow_version']}")
answer_tab, pair_tab, guide_tab = st.tabs(
    ["Answer review", "Contradiction pairs", "Reviewer guide"]
)

with answer_tab:
    answers = bundle["answers"]
    answer_rows = bundle["answer_records"]
    completed = sum(is_answer_annotation_complete(row) for row in answer_rows)
    st.progress(completed / len(answers) if answers else 0)
    st.write(f"**Answer rows complete:** {completed}/{len(answers)}")

    answer_index = st.selectbox(
        "Select answer row",
        options=range(len(answers)),
        format_func=lambda index: (
            f"Answer {index + 1}"
            + (" · complete" if is_answer_annotation_complete(answer_rows[index]) else " · needs review")
        ),
    )
    answer = answers[answer_index]
    st.subheader("Question")
    st.write(answer["question"])
    st.subheader("Provided evidence")
    for passage in answer["evidence"]:
        st.markdown(
            f"**[{passage['citation_label']}]** · passage `{passage['evidence_id']}` · page {passage['page']}"
        )
        st.write(passage["text"])
    st.subheader("Model answer")
    st.text(answer["model_answer"])

    current = answer["annotations"]
    current_row = answer_rows[answer_index]
    with st.form(f"answer-review-{answer['output_id']}"):
        fields: dict[str, str] = {}
        columns = st.columns(2)
        labels = {
            "evidence_support": "Evidence support",
            "answer_correctness": "Answer correctness",
            "appropriate_uncertainty": "Appropriate uncertainty",
            "contradiction_with_evidence": "Contradiction with evidence",
        }
        for index, field in enumerate(ANSWER_FIELDS):
            options = ("", *ANSWER_RATINGS[field])
            previous = current.get(field, "")
            selected = previous if previous in options else ""
            fields[field] = columns[index % 2].selectbox(
                labels[field],
                options=options,
                index=options.index(selected),
                key=f"{answer['output_id']}-{field}",
                format_func=lambda option: "Select a rating" if option == "" else option,
            )
        notes = st.text_area(
            "Notes",
            value=current_row.get("notes", ""),
            key=f"{answer['output_id']}-notes",
            help="Optional rationale; cite claims or evidence passage IDs where useful.",
        )
        save_clicked = st.form_submit_button("Save answer annotation")
        if save_clicked:
            if any(not value for value in fields.values()):
                st.error("Choose a rating for all four evaluation fields before saving.")
            else:
                try:
                    save_answer_annotation(bundle, answer["output_id"], fields, notes)
                    st.success("Annotation saved. Raw model outputs were not changed.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not save annotation: {exc}")

    st.download_button(
        "Export answer annotation CSV",
        data=export_csv(bundle["answer_headers"], answer_rows),
        file_name="human_annotations.csv",
        mime="text/csv",
    )

with pair_tab:
    pairs = bundle["pairs"]
    pair_records = bundle["pair_records"]
    pair_completed = sum(is_pair_annotation_complete(row) for row in pair_records)
    st.progress(pair_completed / len(pairs) if pairs else 0)
    st.write(f"**Contradiction pairs reviewed:** {pair_completed}/{len(pairs)}")
    pair_index = st.selectbox(
        "Select response pair",
        options=range(len(pairs)),
        format_func=lambda index: (
            f"Pair {index + 1}"
            + (" · reviewed" if is_pair_annotation_complete(pair_records[index]) else " · needs review")
        ),
        key="pair-index",
    )
    pair = pairs[pair_index]
    st.subheader("Question")
    st.write(pair["question"])
    st.subheader("Shared evidence")
    for index, passage in enumerate(pair["evidence"], start=1):
        st.markdown(f"**[E{index}]** · passage `{passage['evidence_id']}` · page {passage['page']}")
        st.write(passage["text"])
    left, right = st.columns(2)
    with left:
        st.markdown("**Response A**")
        st.text(pair["response_a"])
    with right:
        st.markdown("**Response B**")
        st.text(pair["response_b"])
    with st.expander("Pair metadata"):
        st.json(pair["metadata"])
    st.warning(
        "The lexical candidate flag is only a screening aid. Decide whether the answers truly contradict one another using the question and shared evidence; do not infer contradiction from lexical similarity alone."
    )
    pair_record = pair_records[pair_index]
    with st.form(f"pair-review-{pair_index}"):
        initial_value = pair_record.get("human_contradiction", "not_reviewed")
        if initial_value not in PAIR_RATINGS:
            initial_value = "not_reviewed"
        decision = st.radio(
            "Do the responses contradict one another?",
            options=PAIR_RATINGS,
            index=PAIR_RATINGS.index(initial_value),
            horizontal=True,
            help="This pair-level decision is separate from contradiction_with_evidence in answer review.",
        )
        pair_notes = st.text_area(
            "Pair-review notes",
            value=pair_record.get("notes", ""),
            key=f"pair-notes-{pair_index}",
        )
        if st.form_submit_button("Save pair review"):
            try:
                save_pair_annotation(bundle, pair_index, decision, pair_notes)
                st.success("Pair review saved. Raw model outputs were not changed.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save pair review: {exc}")

    st.download_button(
        "Export contradiction-pair CSV",
        data=export_csv(
            list(pair_records[0]) if pair_records else [], pair_records
        ),
        file_name="contradiction_pairs.csv",
        mime="text/csv",
    )

with guide_tab:
    st.subheader("Answer-level ratings")
    st.markdown(
        "- **Evidence support (Yes / Partial / No):** whether factual claims follow from the supplied passages. Citation presence alone is not support.\n"
        "- **Answer correctness (Yes / Partial / No):** correctness relative to the paper and question; consult the paper where needed. Do not treat another model answer as ground truth.\n"
        "- **Appropriate uncertainty (Yes / No / Not applicable):** whether uncertainty is qualified or the answer abstains appropriately, especially when evidence is insufficient.\n"
        "- **Contradiction with evidence (Yes / No):** whether answer content conflicts with the supplied evidence.\n"
        "- **Notes:** optional short rationale with claim or passage IDs."
    )
    st.subheader("Pair-level contradiction review")
    st.write(
        "Review both responses against the question and shared evidence. `candidate_flag` is a lexical screen only and is never treated as a true contradiction automatically. Pair decisions are saved separately from answer-level ratings."
    )
    st.caption(
        "The workflow version and experiment ID are saved separately in human_review_manifest.json. "
        "Answer CSVs omit prompt-condition labels; raw outputs remain read-only."
    )
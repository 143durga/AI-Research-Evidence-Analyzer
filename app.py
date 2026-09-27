"""Streamlit user interface for the research evidence analyzer."""

import hashlib
import logging
import os

import streamlit as st
from dotenv import load_dotenv

from research_analyzer.answering import AnswerGenerator
from research_analyzer.pipeline import ResearchAnalyzer
from research_analyzer.vector_store import PaperVectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_analyzer(storage_path: str) -> ResearchAnalyzer:
    return ResearchAnalyzer(PaperVectorStore(storage_path))


st.set_page_config(page_title="Research Evidence Analyzer", page_icon="📄")
st.title("Research Evidence Analyzer")
st.caption("Explore one research paper at a time, with answers tied to retrieved passages.")

with st.sidebar:
    st.subheader("Answer model")
    model = os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
    st.write(f"Model: `{model}`")
    st.write("Embeddings: local Chroma ONNX model")
    st.caption("PDF text and vectors are stored locally in data/chroma.")

uploaded_pdf = st.file_uploader("Upload a research-paper PDF", type=["pdf"])
if uploaded_pdf is None:
    st.info("Upload a text-based PDF to begin. Scanned PDFs currently need OCR first.")
    st.stop()

pdf_bytes = uploaded_pdf.getvalue()
if len(pdf_bytes) > 50 * 1024 * 1024:
    st.error("This prototype accepts PDFs up to 50 MB.")
    st.stop()

analyzer = get_analyzer(os.getenv("CHROMA_PATH", "data/chroma"))
upload_signature = f"{uploaded_pdf.name}:{hashlib.sha256(pdf_bytes).hexdigest()}"
if st.session_state.get("upload_signature") != upload_signature:
    try:
        with st.spinner("Extracting text, splitting chunks, and indexing the paper..."):
            summary = analyzer.index_pdf(uploaded_pdf.name, pdf_bytes)
        st.session_state["document_id"] = summary.document_id
        st.session_state["upload_signature"] = upload_signature
        st.session_state["index_summary"] = summary
    except Exception as exc:
        logger.exception("Could not index uploaded PDF")
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

summary = st.session_state["index_summary"]
st.info(
    f"Indexed {summary.page_count} text pages into {summary.chunk_count} searchable chunks."
)

with st.form("research_question"):
    question = st.text_area(
        "Research question", placeholder="What does the paper report about ...?"
    )
    top_k = st.slider("Evidence passages to retrieve", min_value=1, max_value=10, value=5)
    max_distance = st.slider(
        "Maximum cosine distance",
        min_value=0.0,
        max_value=2.0,
        value=0.75,
        step=0.05,
        help="Lower distances are closer matches. This starting threshold is a heuristic, not a validated confidence score.",
    )
    submitted = st.form_submit_button("Find evidence and answer")

if submitted:
    key = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    generator = AnswerGenerator(model=model, base_url=base_url, api_key=key)
    try:
        with st.spinner("Searching this paper and preparing an evidence-grounded answer..."):
            result = analyzer.answer_question(
                document_id=summary.document_id,
                question=question,
                generator=generator,
                top_k=top_k,
                max_distance=max_distance,
            )
        st.subheader("Answer")
        st.write(result.answer)
        st.caption(
            f"Retrieved {result.retrieved_count} passages; "
            f"{len(result.evidence)} met the distance threshold."
        )
        if result.evidence:
            st.subheader("Supporting evidence")
            for index, item in enumerate(result.evidence, start=1):
                with st.expander(
                    f"[E{index}] {item.filename}, page {item.page_number} "
                    f"(cosine distance {item.distance:.3f})"
                ):
                    st.write(item.text)
    except Exception as exc:
        logger.exception("Could not answer research question")
        st.error(f"Could not complete the analysis: {exc}")
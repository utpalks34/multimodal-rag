"""Streamlit client for the FastAPI /query endpoint. Never the reverse: this
UI only calls api/main.py over HTTP and displays what it returns."""
import time

import requests
import streamlit as st

API_URL = "http://localhost:8000/query"
REFUSAL = "I could not find this in the provided documents."

_MODALITY_BY_LETTER = {"c": "text", "t": "table", "i": "image"}


def _modality(c: dict) -> str:
    """c.get("modality") when present; otherwise derive from chunk_id "<doc>:p<page>:<letter><n>"."""
    modality = c.get("modality")
    if modality:
        return modality
    chunk_id = c.get("chunk_id", "")
    tail = chunk_id.rsplit(":", 1)[-1]
    letter = tail[0] if tail else ""
    return _MODALITY_BY_LETTER.get(letter, "unknown")

SAMPLE_QUESTIONS = [
    "How many runs did Kohli score in home Tests?",
    "How many runs did Kohli score in Tests in 2018?",
    "What are the core components of an AI agent?",
    "What GPU is required to run the GraphRAG demo?",
    "What is India's ODI win count under Kohli as captain?",
    "What are the five layers of the guardrail architecture?",
]

st.title("Multimodal RAG Demo")
st.caption("Ask questions over the ingested PDFs; answers are grounded in retrieved chunks only.")

if "query" not in st.session_state:
    st.session_state.query = ""

query = st.text_input("Question", value=st.session_state.query)

sample_cols = st.columns(len(SAMPLE_QUESTIONS))
for i, q in enumerate(SAMPLE_QUESTIONS):
    if sample_cols[i].button(q, key=f"sample_{i}"):
        st.session_state.query = q
        st.rerun()

if st.button("Ask") and query:
    start = time.time()
    with st.spinner("Processing..."):
        try:
            resp = requests.post(API_URL, json={"query": query}, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
        except requests.RequestException as e:
            st.error(f"Could not reach the API at {API_URL}: {e}")
            data = None

    if data is not None:
        with st.expander("Raw response"):
            st.json(data)

        answer = data.get("text_answer", "")
        if answer.strip() == REFUSAL:
            st.info(answer)
        else:
            st.write(answer)

        st.subheader("Citations")
        citations = data.get("citations", [])
        if citations:
            for c in citations:
                doc = c.get("doc", "unknown")
                page = c.get("page", "?")
                chunk_id = c.get("chunk_id", "?")
                st.write(f"- {doc} (page {page}, {_modality(c)}) — chunk `{chunk_id}`")
        else:
            st.write("None")

        chunks = data.get("chunks", [])
        if chunks:
            with st.expander("Retrieved chunks"):
                for c in chunks:
                    chunk_id = c.get("chunk_id", "?")
                    st.markdown(f"**{chunk_id}** ({_modality(c)})")
                    st.text(c.get("text", ""))

        st.caption(f"Total latency: {elapsed:.2f}s")

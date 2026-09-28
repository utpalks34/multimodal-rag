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

cols = st.columns(3)
for i, q in enumerate(SAMPLE_QUESTIONS):
    if cols[i % 3].button(q, key=f"sample_{i}"):
        st.session_state.query = q
        st.rerun()

if st.button("Ask") and query:
    # Cosmetic only: mirrors the query-time pipeline order from CLAUDE.md, does not
    # measure real per-stage timing (the API returns one response for the whole call).
    with st.status("Running pipeline...", expanded=True) as status:
        start = time.time()
        st.write("🔎 Retrieve — embedding query, searching Qdrant...")
        time.sleep(0.3)
        st.write("↕️ Rerank — Phase 4 stub, pass-through")
        time.sleep(0.15)
        st.write("✍️ Generate — hermes3-rag via Ollama...")
        try:
            resp = requests.post(API_URL, json={"query": query}, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            elapsed = time.time() - start
        except requests.RequestException as e:
            status.update(label="Pipeline failed", state="error")
            st.error(f"Could not reach the API at {API_URL}: {e}")
        else:
            st.write("🔊 TTS — Phase 5 stub, skipped (audio_url unset)")
            time.sleep(0.15)
            status.update(label="Pipeline complete", state="complete")

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

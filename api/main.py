"""FastAPI app wiring the pipeline together: POST /query {query, document_scope?} -> {text_answer, citations, trace_id}.

Ingestion is offline: python -m api.main <file.pdf> [...]
"""
import sys
from collections import Counter
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel

from generation.generator import MODEL, generate
from index.embedder import embed_chunks
from index.vector_store import delete_doc, upsert
from ingestion.captioner import caption_images
from ingestion.chunker import chunk_elements
from ingestion.parser import parse_pdf
from observability.tracing import flush, new_trace_id, span
from retrieval.retriever import retrieve

app = FastAPI(title="Multimodal RAG")


class QueryRequest(BaseModel):
    query: str
    document_scope: list[str] | None = None


class Citation(BaseModel):
    doc: str
    page: int
    modality: str
    chunk_id: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc: str
    page: int
    modality: str
    text: str


class QueryResponse(BaseModel):
    text_answer: str
    citations: list[Citation]
    chunks: list[RetrievedChunk]
    trace_id: str


@app.post("/query", response_model=QueryResponse, response_model_exclude_none=True)
def query(req: QueryRequest) -> QueryResponse:
    trace_id = new_trace_id()
    with span("query_request", trace_id=trace_id, input={"query": req.query, "document_scope": req.document_scope}) as root:
        with span("retrieve", input={"query": req.query}) as s:
            chunks = retrieve(req.query, req.document_scope)
            s.update(output=[{"chunk_id": c["chunk_id"], "score": c["score"], "modality": c["modality"]} for c in chunks])

        with span("generate", as_type="generation", model=MODEL, input={"query": req.query}) as s:
            result = generate(req.query, chunks)
            usage = result["usage"]
            s.update(
                output=result["text_answer"],
                usage_details={"input": usage["input"], "output": usage["output"]} if usage else None,
            )
        root.update(output={"text_answer": result["text_answer"], "citations": result["citations"]})

    return QueryResponse(
        text_answer=result["text_answer"],
        citations=result["citations"],
        chunks=chunks,
        trace_id=trace_id,
    )


def ingest_pdf(path: str | Path) -> int:
    """Parse -> chunk -> embed -> index one PDF under a single trace. Returns the chunk count."""
    path = Path(path)
    trace_id = new_trace_id()
    with span("ingest_request", trace_id=trace_id, input={"doc": path.name}):
        with span("parse") as s:
            elements = parse_pdf(path)
            s.update(output={"elements": len(elements)})
        with span("caption") as s:
            non_image = [e for e in elements if e["type"] != "image"]
            captioned = caption_images(elements)
            elements = non_image + captioned
            s.update(output={"images_captioned": len(captioned)})
        with span("chunk") as s:
            chunks = chunk_elements(elements)
            by_modality = Counter(c["modality"] for c in chunks)
            s.update(output={"chunks": len(chunks), "by_modality": dict(by_modality)})
        with span("embed") as s:
            points = embed_chunks(chunks)
            s.update(output={"vectors": len(points)})
        with span("index") as s:
            delete_doc(path.name)
            upsert(points)
            s.update(output={"upserted": len(points)})
    print(f"{path.name}: {len(elements)} elements -> {len(chunks)} chunks (trace {trace_id})")
    return len(chunks)


if __name__ == "__main__":
    for pdf in sys.argv[1:]:
        ingest_pdf(pdf)
    flush()

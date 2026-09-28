# Multimodal RAG (text + tables + images, voice output planned)

Retrieval-augmented Q&A over PDFs (slide decks, reports) that mixes prose, tables, and
charts/images. Pipeline: **parse -> caption -> chunk -> embed -> index -> retrieve ->
rerank -> generate -> TTS**. Local models only (Ollama), traced end-to-end with Langfuse.

## Status

| Stage | Phase | State |
|---|---|---|
| Parse (Docling: text + OCR)        | 1 | done |
| Table extraction (cell-level)      | 2 | done |
| Image/chart captioning (moondream) | 3 | done |
| Rerank                             | 4 | stub (`retrieval/reranker.py` — pass-through) |
| TTS                                | 5 | stub (`voice/tts.py` — not implemented) |

The dev-loop eval (`python -m eval.run_eval`, no `--gate`, 27 verified golden questions)
currently shows text/table/image retrieval solid (MRR 0.83–0.91, table cell F1 1.0) but
**chart retrieval is weak** (gold_hit@5 = 0.17, MRR 0.58) — chart chunks embed a moondream
description, and vague questions about a chart don't match that description well. See
`eval/results/` for the latest run's full breakdown. No `--gate` run (which also checks
RAGAS Faithfulness against the phase-1/2/3 pass thresholds) has been recorded yet.

## Architecture

```
ingestion/parser.py     Docling: PDF -> text/table/image elements (OCR on, table structure on)
ingestion/captioner.py  moondream VLM caption for images without usable visible text
ingestion/chunker.py    per-modality chunking (sentence-split text; one chunk per table/image)
index/embedder.py       nomic-embed-text via Ollama's OpenAI-compatible endpoint
index/vector_store.py   Qdrant, one "chunks" collection for all modalities
retrieval/retriever.py  embed query -> top-N vector search (TOP_N = 10)
retrieval/reranker.py   Phase 4 stub
generation/generator.py hermes3-rag (custom Modelfile) — answers only from retrieved chunks,
                         cites chunk ids like [C2], refuses when the context doesn't contain
                         the answer
voice/tts.py            Phase 5 stub
observability/tracing.py Langfuse span helper wired through every stage
api/main.py              FastAPI: POST /query; offline ingestion via `python -m api.main *.pdf`
demo/app.py               Streamlit client that only calls the FastAPI /query endpoint
eval/run_eval.py          retrieval + programmatic-check + RAGAS-judge metrics vs. golden_set.json
```

## Setup

```
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
```

Requires a running [Ollama](https://ollama.com) with:
- `nomic-embed-text` (embeddings)
- `moondream` (image captioning)
- `hermes3:8b`, then build the tuned generation model from the repo's `Modelfile`:
  `ollama create hermes3-rag -f Modelfile`

Requires a running [Qdrant](https://qdrant.tech) instance.

Copy `.env` and fill in (see `Modelfile`/module docstrings for defaults):
- `QDRANT_URL`
- `LLM_BASE_URL` (Ollama's OpenAI-compatible endpoint, e.g. `http://localhost:11434/v1`)
- `LLM_MODEL` (default `hermes3-rag`), `EMBED_MODEL` (default `nomic-embed-text`),
  `VLM_MODEL` (default `moondream`), `JUDGE_MODEL` (for `eval/run_eval.py --gate`)
- `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`

`COHERE_API_KEY`, `CARTESIA_API_KEY`, `SENTRY_DSN`, `Groq_API_KEY` are reserved for later
phases (rerank / TTS / error tracking / alternate judge) and aren't read by any module yet.

## Usage

Ingest one or more PDFs (parse -> caption -> chunk -> embed -> index, one trace per file):
```
python -m api.main path/to/deck.pdf [more.pdf ...]
```

Run the API:
```
uvicorn api.main:app --reload
```

Run the demo UI (talks to the API over HTTP, never the reverse):
```
streamlit run demo/app.py
```

Run the eval suite:
```
python -m eval.run_eval            # retrieval + programmatic checks, fast, no judge
python -m eval.run_eval --smoke    # same, first 5 verified questions across types
python -m eval.run_eval --stage retrieval  # retrieval-only metrics, no LLM calls
python -m eval.run_eval --gate     # full phase gate incl. RAGAS Faithfulness judge
```

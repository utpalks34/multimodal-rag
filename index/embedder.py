"""Single embedding call for all chunks; tags modality + embedding_model_version. (Phase 1)

Local model via Ollama's OpenAI-compatible endpoint (tools-requirements.md).
"""
import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

EMBEDDING_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBEDDING_MODEL_VERSION = EMBEDDING_MODEL  # Phase 6 versioning keys off this tag
_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")

# nomic-embed-text expects task prefixes; skipping them measurably hurts retrieval.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=_BASE_URL, api_key="ollama")  # Ollama ignores the key
    return _client


def _embed(texts: list[str]) -> list[list[float]]:
    resp = _get_client().embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [d.embedding for d in resp.data]


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """Return one point per chunk: {"vector", "payload"}. Payload carries the chunk plus model
    version. Table chunks carry a separate "embed_text" (ingestion.chunker's deterministic
    linearization) -- that's what gets embedded; the payload's displayed "text" is unchanged
    and embed_text rides along in the payload since it's already a key of c.
    """
    vectors = _embed([DOC_PREFIX + c.get("embed_text", c["text"]) for c in chunks])
    return [
        {"vector": v, "payload": {**c, "embedding_model_version": EMBEDDING_MODEL_VERSION}}
        for c, v in zip(chunks, vectors)
    ]


def embed_query(query: str) -> list[float]:
    return _embed([QUERY_PREFIX + query])[0]

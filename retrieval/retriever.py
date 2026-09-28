"""Top-N retrieval. (Phase 1)"""
from index.embedder import embed_query
from index.vector_store import search

TOP_N = 10  # matches the Recall@10 eval gate


def retrieve(query: str, document_scope: list[str] | None = None, top_n: int = TOP_N) -> list[dict]:
    """Embed the query and return the top_n chunks (payload + "score"), best first."""
    return search(embed_query(query), top_n, document_scope)

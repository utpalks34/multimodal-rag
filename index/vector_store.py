"""Thin wrapper around the Qdrant client: one collection for all modalities. (Phase 1)"""
import os
import uuid

from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

load_dotenv()

COLLECTION = "chunks"

_client: QdrantClient | None = None


def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=os.environ["QDRANT_URL"])
    return _client


def _point_id(chunk_id: str) -> str:
    # Qdrant ids must be uint or UUID; derive a stable UUID so re-ingest overwrites in place.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def _ensure_collection(dim: int) -> None:
    client = _get_client()
    if not client.collection_exists(COLLECTION):
        client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


def delete_doc(doc: str) -> None:
    """Drop every chunk of a document so a re-ingest can't leave stale chunks behind."""
    client = _get_client()
    if client.collection_exists(COLLECTION):
        client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(
                filter=models.Filter(must=[models.FieldCondition(key="doc", match=models.MatchValue(value=doc))])
            ),
        )


def upsert(points: list[dict]) -> None:
    """points: [{"vector", "payload"}] as produced by embedder.embed_chunks."""
    if not points:
        return
    _ensure_collection(len(points[0]["vector"]))
    _get_client().upsert(
        COLLECTION,
        points=[
            models.PointStruct(id=_point_id(p["payload"]["chunk_id"]), vector=p["vector"], payload=p["payload"])
            for p in points
        ],
    )


def indexed_docs() -> set[str]:
    """Return every distinct "doc" payload value currently in the collection. (eval gate sanity check)"""
    client = _get_client()
    if not client.collection_exists(COLLECTION):
        return set()
    docs: set[str] = set()
    offset = None
    while True:
        points, offset = client.scroll(COLLECTION, limit=256, offset=offset, with_payload=["doc"])
        docs.update(p.payload["doc"] for p in points)
        if offset is None:
            break
    return docs


def get_by_id(chunk_id: str) -> dict | None:
    """Return one chunk's payload by its chunk_id, or None if it isn't indexed. (Phase 2: table cell F1 eval)"""
    client = _get_client()
    if not client.collection_exists(COLLECTION):
        return None
    points = client.retrieve(COLLECTION, ids=[_point_id(chunk_id)], with_payload=True)
    return points[0].payload if points else None


def search(vector: list[float], top_n: int, document_scope: list[str] | None = None) -> list[dict]:
    """Return payload + "score" for the top_n nearest chunks, optionally limited to the given docs."""
    client = _get_client()
    if not client.collection_exists(COLLECTION):
        return []
    query_filter = None
    if document_scope:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="doc", match=models.MatchAny(any=document_scope))]
        )
    result = client.query_points(
        COLLECTION, query=vector, limit=top_n, query_filter=query_filter, with_payload=True
    )
    return [{**p.payload, "score": p.score} for p in result.points]

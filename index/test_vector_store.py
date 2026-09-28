"""Idempotency test for index.vector_store: re-upserting identical chunks must overwrite
in place, never duplicate. Requires a reachable Qdrant (QDRANT_URL); runs against a
throwaway collection so it never touches the real "chunks" collection.
    python -m index.test_vector_store
"""
from index import vector_store

TEST_COLLECTION = "chunks_test_idempotency"

POINTS = [
    {
        "vector": [0.1, 0.2, 0.3],
        "payload": {"chunk_id": "doc.pdf:p1:c0", "doc": "doc.pdf", "page": 1, "modality": "text", "text": "hello"},
    },
    {
        "vector": [0.4, 0.5, 0.6],
        "payload": {"chunk_id": "doc.pdf:p1:t0", "doc": "doc.pdf", "page": 1, "modality": "table", "text": "| a |"},
    },
]


def test_reupsert_is_idempotent():
    original_collection = vector_store.COLLECTION
    vector_store.COLLECTION = TEST_COLLECTION
    client = vector_store._get_client()
    try:
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)

        vector_store.upsert(POINTS)
        first_count = client.get_collection(TEST_COLLECTION).points_count
        assert first_count == 2, f"expected 2 points after first upsert, got {first_count}"

        vector_store.upsert(POINTS)  # re-run with identical chunks: must overwrite, not duplicate
        second_count = client.get_collection(TEST_COLLECTION).points_count
        assert second_count == 2, f"expected 2 points after re-upsert (idempotent), got {second_count}"

        # same chunk_id -> same point id, every call (the actual guarantee this test protects)
        assert vector_store._point_id("doc.pdf:p1:c0") == vector_store._point_id("doc.pdf:p1:c0")
        # different (doc, page, modality, index) -> different point id
        assert vector_store._point_id("doc.pdf:p1:c0") != vector_store._point_id("doc.pdf:p1:t0")
    finally:
        if client.collection_exists(TEST_COLLECTION):
            client.delete_collection(TEST_COLLECTION)
        vector_store.COLLECTION = original_collection


if __name__ == "__main__":
    test_reupsert_is_idempotent()
    print("OK: test_reupsert_is_idempotent")

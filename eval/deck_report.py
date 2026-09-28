"""Read-only diagnostic: chunk counts and char totals per doc page. Usage: python -m eval.deck_report"""
from collections import defaultdict

from index import vector_store

FLAG_CHARS = 150


def main():
    client = vector_store._get_client()
    points, offset = [], None
    while True:
        batch, offset = client.scroll(vector_store.COLLECTION, limit=256, offset=offset, with_payload=True)
        points.extend(batch)
        if offset is None:
            break

    pages = defaultdict(list)
    for p in points:
        pages[(p.payload["doc"], p.payload["page"])].append(p.payload)

    for (doc, page), chunks in sorted(pages.items()):
        modality_counts = defaultdict(int)
        for c in chunks:
            modality_counts[c["modality"]] += 1
        total_chars = sum(len(c["text"]) for c in chunks)
        flag = " [LOW CONTENT]" if total_chars < FLAG_CHARS else ""
        print(f"{doc} p{page}: {dict(modality_counts)} chars={total_chars}{flag} first={chunks[0]['text'][:80]!r}")


if __name__ == "__main__":
    main()

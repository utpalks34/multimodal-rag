"""Print the full stored ("text" payload) of one or more ingested chunks, by chunk_id.
Diagnostic only -- reads Qdrant directly, no retrieval/generation call.

Usage: python -m eval.print_chunk_text <chunk_id> [<chunk_id> ...]
Example: python -m eval.print_chunk_text "Virat_Kohli_Career_Retrospective.pdf:p4:i0"
"""
import sys

from index.vector_store import get_by_id


def main(chunk_ids: list[str]) -> None:
    for chunk_id in chunk_ids:
        payload = get_by_id(chunk_id)
        print(f"=== {chunk_id} ===")
        if payload is None:
            print("(not found -- not ingested yet, or chunk_id doesn't match)")
        else:
            print(payload["text"])
        print()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python -m eval.print_chunk_text <chunk_id> [<chunk_id> ...]")
    main(sys.argv[1:])

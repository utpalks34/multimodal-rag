"""Per-modality chunking. (Phase 1 text, Phase 2 tables, Phase 3 captions)

Image elements arrive from ingestion.captioner as {"type": "image", "text": <assembled
caption>, "page", "doc", "auto_generated": True} and become one chunk each, same as tables --
a caption is a single unit of meaning, not something to sentence-split.
"""
from itertools import groupby

from llama_index.core.node_parser import SentenceSplitter

CHUNK_SIZE = 512  # tokens
CHUNK_OVERLAP = 64

_splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)


def _linearize_table_cells(title: str | None, header: list[str], rows: list[list[str]], doc: str, page: int) -> str:
    """Deterministic natural-language linearization from Docling's own cell-level export: a
    title line, then one sentence per row pairing each non-label column header with its value.
    """
    lines = [title or f"Table on page {page}"]
    for row in rows:
        if not row:
            continue
        label, *rest = row
        parts = [f"{h} {v}" for h, v in zip(header[1:], rest) if v]
        lines.append(f"{label}: " + "; ".join(parts) + "." if parts else f"{label}.")
    lines.append(f"Doc: {doc}, Page: {page}")
    return "\n".join(lines)


def _linearize_table_markdown(markdown: str, doc: str, page: int) -> str:
    """Fallback when Docling cells aren't available: parse the Markdown table deterministically
    (same separator-row detection as eval/run_eval.py's _parse_markdown_cells).
    """
    raw_lines = [ln.strip() for ln in markdown.splitlines() if ln.strip()]
    title_lines = [ln for ln in raw_lines if not ln.startswith("|")]
    title = " ".join(title_lines) if title_lines else f"Table on page {page}"
    rows = []
    for ln in raw_lines:
        if not ln.startswith("|"):
            continue
        if set(ln.replace("|", "").replace(":", "").strip()) <= {"-", " "}:
            continue  # header separator row, e.g. "| --- | --- |"
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    if not rows:
        return f"{title}\nDoc: {doc}, Page: {page}"
    header, *data_rows = rows
    return _linearize_table_cells(title, header, data_rows, doc, page)


def chunk_elements(elements: list[dict]) -> list[dict]:
    """Split text elements into chunks on sentence boundaries; each table becomes one chunk.

    Chunks never cross a page so every citation has a single page number. Each chunk:
    {"chunk_id", "text", "modality", "doc", "page"}. chunk_id is deterministic, so
    re-ingesting an unchanged document reproduces the same ids. Text chunk ids are
    "c{i}"; table chunk ids are "t{i}"; image chunk ids are "i{i}" -- each numbered
    separately per page.
    """
    chunks = []
    ordered = sorted(elements, key=lambda e: (e["doc"], e["page"]))  # stable: keeps reading order within a page
    for (doc, page), group in groupby(ordered, key=lambda e: (e["doc"], e["page"])):
        group = list(group)
        text_elems = [e for e in group if e["type"] == "text"]
        table_elems = [e for e in group if e["type"] == "table"]
        image_elems = [e for e in group if e["type"] == "image"]

        page_text = "\n\n".join(e["text"] for e in text_elems)
        if page_text:  # split_text("") returns [""], which would create a bogus empty chunk
            for i, text in enumerate(_splitter.split_text(page_text)):
                chunks.append(
                    {"chunk_id": f"{doc}:p{page}:c{i}", "text": text, "modality": "text", "doc": doc, "page": page}
                )

        # A table's rows/columns are what make it answerable, so it stays one chunk instead
        # of being sentence-split, which would break the Markdown structure (see design.md).
        # embed_text is a separate deterministic linearization used only for the embedding
        # vector -- the displayed/cited "text" stays the original Markdown either way.
        for i, e in enumerate(table_elems):
            header, rows = e.get("table_header"), e.get("table_rows")
            if header and rows:
                embed_text = _linearize_table_cells(e.get("table_title"), header, rows, doc, page)
            else:
                embed_text = _linearize_table_markdown(e["text"], doc, page)
            chunks.append(
                {
                    "chunk_id": f"{doc}:p{page}:t{i}",
                    "text": e["text"],
                    "modality": "table",
                    "doc": doc,
                    "page": page,
                    "embed_text": embed_text,
                }
            )

        # A caption is one unit of meaning, same reasoning as tables -- no sentence-splitting.
        # embed_text (when captioner.py supplied one, from visible_text) follows the same
        # text/embed_text split as table chunks; otherwise embedder.py falls back to "text".
        for i, e in enumerate(image_elems):
            chunk = {
                "chunk_id": f"{doc}:p{page}:i{i}",
                "text": e["text"],
                "modality": "image",
                "doc": doc,
                "page": page,
                "auto_generated": True,
            }
            if "embed_text" in e:
                chunk["embed_text"] = e["embed_text"]
            chunks.append(chunk)
    return chunks

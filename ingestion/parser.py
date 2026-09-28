"""Docling wrapper: parse a PDF into text/table/image elements with page + position metadata. (Phase 1 text, Phase 2 tables, Phase 3 images)"""
from pathlib import Path

import pypdfium2 as pdfium
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, PictureItem, TableItem, TextItem

# Running headers/footers repeat on every page and pollute retrieval.
_SKIP_LABELS = {DocItemLabel.PAGE_HEADER, DocItemLabel.PAGE_FOOTER}

_converter: DocumentConverter | None = None


def _get_converter() -> DocumentConverter:
    global _converter
    if _converter is None:
        # Phase 3 turns on picture-image extraction so PictureItem.get_image() returns real
        # bytes for the captioner; do_picture_description stays off (Docling's own built-in
        # captioning) since captioning is our own component, per design.md. OCR is on: full-page
        # OCR is the only way image-only slide pages (no text layer at all) yield any text --
        # confirmed on the Agentic deck, where 18 of 19 pages were otherwise silently empty.
        # Limitation: CPU-only OCR (RapidOCR/torch) is slow, ~10s/page.
        opts = PdfPipelineOptions(do_ocr=True, do_table_structure=True, generate_picture_images=True)
        _converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
        )
    return _converter


def _pdfium_bbox_text(pdf_doc: pdfium.PdfDocument, page_no: int, bbox, page_height: float) -> str:
    """PDF text layer inside a picture's bbox, read directly via pypdfium2 -- the fallback when
    no Docling TextItem overlaps the picture (e.g. chart axis/data labels the layout model
    assigned to the picture region without promoting to a TextItem). pdfium's
    get_text_bounded() takes PDF canvas coordinates with a bottom-left origin, so the
    already-top-left-normalized bbox is converted back before the call.
    """
    bl = bbox.to_bottom_left_origin(page_height)
    text = pdf_doc[page_no - 1].get_textpage().get_text_bounded(left=bl.l, bottom=bl.b, right=bl.r, top=bl.t)
    return text.strip()


def parse_pdf(path: str | Path) -> list[dict]:
    """Return text, table, and image elements in reading order.

    Text/table: {"type": "text"|"table", "text", "page", "doc"}. A table's "text" is its
    Markdown serialization (row/column structure kept, per design.md); a text element's
    "text" is plain prose. A table element also carries "table_title" (Docling's caption,
    or None) and "table_header"/"table_rows" (Docling's own cell-level export) -- consumed
    by ingestion.chunker to build a table chunk's embed_text.

    Image: {"type": "image", "image": PIL.Image, "pdf_caption", "page", "doc", "visible_text"}.
    "image" is consumed by ingestion.captioner and never embedded directly; "pdf_caption" is
    the figure caption Docling extracted from the PDF itself ("none" if absent); "visible_text"
    is text whose bbox lies inside the picture's bbox on the same page: Docling TextItems first
    (text-layer or OCR), joined in reading order; if none overlap, the PDF's own text layer
    inside that bbox read directly via pypdfium2 ("none" if nothing overlaps either source).
    """
    path = Path(path)
    document = _get_converter().convert(path).document
    # Picture bboxes and text-item bboxes aren't guaranteed to share a coordinate origin (one can
    # come out bottom-left, PDF-native; the other top-left) -- comparing them raw silently finds
    # no overlap. Normalize everything to top-left using each page's own height before comparing.
    page_heights = {no: p.size.height for no, p in document.pages.items()}
    page_areas = {no: p.size.width * p.size.height for no, p in document.pages.items()}
    elements = []
    text_items = []  # (page, bbox, text) for every text element, in reading order -- used below to fill picture "visible_text"
    picture_entries = []  # (elements index, bbox, page) for every picture -- filled in after the main pass
    for item, _level in document.iterate_items():
        if isinstance(item, TableItem):
            if not item.prov:
                continue
            markdown = item.export_to_markdown(document).strip()
            if not markdown:
                continue
            df = item.export_to_dataframe(document)
            elements.append({
                "type": "table",
                "text": markdown,
                "page": item.prov[0].page_no,
                "doc": path.name,
                "table_title": item.caption_text(document).strip() or None,
                "table_header": [str(c) for c in df.columns.tolist()],
                "table_rows": [[str(v) for v in row] for row in df.values.tolist()],
            })
            continue
        if isinstance(item, PictureItem):
            if not item.prov:
                continue
            image = item.get_image(document)
            if image is None:  # e.g. a picture placeholder with no decodable bitmap
                continue
            caption = item.caption_text(document).strip() or "none"
            page_no = item.prov[0].page_no
            bbox = item.prov[0].bbox.to_top_left_origin(page_heights[page_no])
            picture_entries.append((len(elements), bbox, page_no))
            elements.append(
                {"type": "image", "image": image, "pdf_caption": caption, "page": page_no, "doc": path.name}
            )
            continue
        # SectionHeaderItem and ListItem subclass TextItem.
        if not isinstance(item, TextItem) or item.label in _SKIP_LABELS:
            continue
        text = item.text.strip()
        if not text or not item.prov:
            continue
        page_no = item.prov[0].page_no
        text_items.append((page_no, item.prov[0].bbox.to_top_left_origin(page_heights[page_no]), text))
        elements.append({"type": "text", "text": text, "page": page_no, "doc": path.name})

    # "Visible text" for a picture: Docling text items (text-layer or OCR, do_ocr=True above
    # covers both) whose bbox mostly falls inside the picture's bbox on the same page, kept in
    # the reading order they were collected above. Threshold 0.5 is a simple majority-overlap
    # rule, not a precise clip test -- good enough since chart labels rarely straddle the edge.
    # Both sides are already normalized to top-left origin above. If no TextItem overlaps (e.g.
    # chart axis/data labels the layout model assigned to the picture region without promoting
    # them to a TextItem), fall back to the PDF's own text layer inside that bbox via pypdfium2.
    pdf_doc = pdfium.PdfDocument(str(path)) if picture_entries else None
    page_pic_count: dict[int, int] = {}
    skip_indices = set()
    for idx, pic_bbox, page in picture_entries:
        pic_index = page_pic_count.get(page, 0)
        page_pic_count[page] = pic_index + 1
        inside = [t for (p, bbox, t) in text_items if p == page and bbox.intersection_over_self(pic_bbox) > 0.5]
        docling_text = " ".join(inside)
        if inside:
            pdfium_text = ""
            source = "docling"
        else:
            pdfium_text = _pdfium_bbox_text(pdf_doc, page, pic_bbox, page_heights[page])
            source = "pdfium" if pdfium_text else "none"
        visible_text = docling_text if inside else (pdfium_text or "none")
        visible_text = " ".join(visible_text.split())
        print(
            f"picture p{page}:i{pic_index} bbox=({pic_bbox.l:.1f},{pic_bbox.t:.1f},{pic_bbox.r:.1f},{pic_bbox.b:.1f}) "
            f"docling_items={len(inside)} docling_chars={len(docling_text)} pdfium_chars={len(pdfium_text)} source={source}"
        )
        # Skip pictures that are both textless and tiny (e.g. bullet icons, logos): no image
        # chunk, no moondream call. Real charts/photos either carry visible text or are large.
        bbox_area = (pic_bbox.r - pic_bbox.l) * (pic_bbox.b - pic_bbox.t)
        textless = visible_text.lower() == "none" or len(visible_text) < 20
        tiny = bbox_area < 0.10 * page_areas[page]
        if textless and tiny:
            skip_indices.add(idx)
            continue
        elements[idx]["visible_text"] = visible_text
    if pdf_doc is not None:
        pdf_doc.close()

    if skip_indices:
        elements = [e for i, e in enumerate(elements) if i not in skip_indices]

    pictures_found = len(picture_entries)
    pictures_skipped = len(skip_indices)
    print(
        f"{path.name}: pictures found={pictures_found}, image chunks kept={pictures_found - pictures_skipped}, "
        f"pictures skipped={pictures_skipped}"
    )

    return elements

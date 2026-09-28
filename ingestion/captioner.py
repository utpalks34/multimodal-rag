"""VLM captioning of images/charts at ingestion time. (Phase 3)

moondream is tiny and ignores long instructions, so it only gets one short,
single-task prompt (description) at temperature 0. Visible text inside a picture
comes from Docling's own text-cell/OCR bounding boxes (ingestion.parser), not the
VLM -- see CLAUDE.md on moondream's unreliable chart-number reads. Descriptions are
cached on disk by image hash so re-running ingestion never re-calls the VLM for an
image it has already described.
"""
import hashlib
import io
import json
import os
from pathlib import Path

import ollama
from PIL import Image

VLM_MODEL = os.environ.get("VLM_MODEL", "moondream")

DESCRIBE_PROMPT = "Describe this image in detail."

CAPTION_TEMPLATE = (
    "Figure on page {page} of {doc}. "
    "Original caption: {pdf_caption}. "
    "Automatic description: {describe}. "
    "Visible text: {text}. "
    "Note: this description is automatic and numbers may be unreliable."
)

_CACHE_DIR = Path(__file__).parent.parent / ".cache" / "captions"


def _image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return hashlib.sha256(buf.getvalue()).hexdigest()


def _image_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    import base64

    return base64.b64encode(buf.getvalue()).decode("ascii")


def _ask(image_b64: str, prompt: str) -> str:
    resp = ollama.chat(
        model=VLM_MODEL,
        messages=[{"role": "user", "content": prompt, "images": [image_b64]}],
        options={"temperature": 0},
    )
    return resp["message"]["content"].strip()


def _describe_one(image: Image.Image) -> str:
    """Return the moondream description, using the on-disk cache keyed by image hash."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _CACHE_DIR / f"{_image_hash(image)}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        return cached["describe"]

    describe = _ask(_image_b64(image), DESCRIBE_PROMPT)
    cache_path.write_text(json.dumps({"describe": describe}), encoding="utf-8")
    return describe


def caption_images(elements: list[dict]) -> list[dict]:
    """Turn parser.py's {"type": "image", "image", "pdf_caption", "page", "doc", "visible_text"}
    elements into caption chunk elements: {"type": "image", "text", "page", "doc", "auto_generated": True}.

    "visible_text" is Docling's, not the VLM's. When it holds real content (>=20 chars,
    not "none"), it's the ground truth for what's plotted -- moondream is skipped and the
    chunk gets an "embed_text" (visible_text only, no boilerplate) same as table chunks'
    text/embed_text split. Otherwise falls back to the moondream description template.
    """
    captioned = []
    for e in elements:
        if e["type"] != "image":
            continue
        visible_text = e["visible_text"]
        vt_stripped = visible_text.strip() if visible_text else ""
        if vt_stripped and vt_stripped.lower() != "none" and len(vt_stripped) >= 20:
            flat_text = visible_text.replace("\n", " ")
            caption = f"Figure on page {e['page']} of {e['doc']}. Text in figure: {flat_text}"
            captioned.append(
                {
                    "type": "image",
                    "text": caption,
                    "page": e["page"],
                    "doc": e["doc"],
                    "auto_generated": True,
                    "embed_text": flat_text,
                }
            )
            continue
        describe = _describe_one(e["image"])
        caption = CAPTION_TEMPLATE.format(
            page=e["page"], doc=e["doc"], pdf_caption=e["pdf_caption"], describe=describe, text=e["visible_text"]
        )
        captioned.append({"type": "image", "text": caption, "page": e["page"], "doc": e["doc"], "auto_generated": True})
    return captioned

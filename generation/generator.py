"""Single LLM call: build the prompt from top-k chunks, return answer + citations. (Phase 1)

Local model via Ollama's OpenAI-compatible endpoint (tools-requirements.md). num_ctx/
temperature/num_predict are set in the hermes3-rag Modelfile, not per request -- Ollama's
/v1 endpoint cannot set num_ctx per call, so a prompt longer than the Modelfile's num_ctx
is silently truncated from the front (system prompt + rules dropped first).
"""
import os
import re

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

MODEL = os.environ.get("LLM_MODEL", "hermes3-rag")
_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1")
NUM_CTX = 8192  # must match PARAMETER num_ctx in Modelfile; used only for the truncation check below

REFUSAL = "I could not find this in the provided documents."

_SYSTEM_PROMPT = f"""You answer questions about documents. Use ONLY the CONTEXT the user provides.

Rules:
1. End every factual sentence with the id of the chunk it came from, like [C2]. If it used two chunks, write [C1][C3].
2. If the CONTEXT does not contain the answer, reply with exactly: {REFUSAL}
3. Never use outside knowledge. Never guess or calculate numbers that are not written in the CONTEXT. Copy numbers exactly as written.
4. Chunks with type="table" are Markdown tables. Find the correct row and column before answering.
5. Chunks with type="image" are automatic descriptions and can be wrong. When you rely on one, say "the image description says".
6. Answer in at most 4 short sentences of plain text. No lists, no headings, no markdown.

Example
Question: What was Q2 revenue?
Answer: Q2 revenue was 4.2 million dollars [C2].

Example
Question: Who is the CFO?
Answer: {REFUSAL}"""

_USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer using only the CONTEXT above. Cite chunk ids like [C1]."""

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=_BASE_URL, api_key="ollama")  # Ollama ignores the key
    return _client


def _context_block(chunks: list[dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, 1):
        parts.append(
            f'<chunk id="C{i}" type="{c["modality"]}" doc="{c["doc"]}" page="{c["page"]}">\n'
            f'{c["text"]}\n</chunk>'
        )
    return "\n".join(parts)


def generate(query: str, chunks: list[dict]) -> dict:
    """Return {"text_answer", "citations": [{doc, page, modality, chunk_id}], "usage"}.

    Citations are the chunks the model actually referenced as [C#]; usage is None when no LLM call was made.
    """
    if not chunks:
        return {"text_answer": REFUSAL, "citations": [], "usage": None}

    resp = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0.1,  # also set in the Modelfile; passed here too since some OpenAI-compat clients require it
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": _USER_TEMPLATE.format(context=_context_block(chunks), question=query)},
        ],
    )
    answer = resp.choices[0].message.content.strip()

    cited = sorted({int(n) for n in re.findall(r"\[C(\d+)\]", answer) if 1 <= int(n) <= len(chunks)})
    citations = [
        {
            "doc": chunks[n - 1]["doc"],
            "page": chunks[n - 1]["page"],
            "modality": chunks[n - 1]["modality"],
            "chunk_id": chunks[n - 1]["chunk_id"],
        }
        for n in cited
    ]
    usage = {"input": resp.usage.prompt_tokens, "output": resp.usage.completion_tokens}
    # Ollama's /v1 endpoint drops the beginning of the prompt (system rules) instead of erroring
    # when it doesn't fit num_ctx; treat a maxed-out prompt_tokens count as a failed request.
    if usage["input"] >= NUM_CTX:
        raise RuntimeError(f"prompt truncated: prompt_tokens={usage['input']} >= num_ctx={NUM_CTX}")
    return {"text_answer": answer, "citations": citations, "usage": usage}

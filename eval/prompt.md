# Prompts and Model Settings (local models)

Use these verbatim. Do not rewrite, extend, or "improve" them. A prompt change
is a pipeline change: it goes through the eval gate in phasewise.md like any
other change. Small local models are sensitive to wording, so one edit can move
scores in either direction.

Design rules used for every prompt below, because the models are small:
short numbered rules, one task per prompt, explicit refusal string, one
example, question placed last, low temperature.

---

## 0. Model settings (do this first)

Ollama's OpenAI-compatible `/v1` endpoint cannot set `num_ctx` per request. If
the prompt is longer than the context window, the beginning (your system
prompt and rules) is silently dropped. Set it in a Modelfile instead.

`Modelfile`
```
FROM hermes3:8b
PARAMETER num_ctx 8192
PARAMETER temperature 0.1
PARAMETER num_predict 300
```

```
ollama create hermes3-rag -f Modelfile
```

Then set `LLM_MODEL=hermes3-rag` in `.env`. The extra KV cache for an 8192
window is roughly 1 GB on top of the 4.7 GB model; check with `ollama ps`.

Context budget (8192 tokens): system prompt ~250, question ~50, answer up to
300, leaves ~7500 for chunks. Keep chunks <= 500 tokens and send only the top
4-5 chunks after reranking. More chunks makes an 8B model worse, not better;
tune `top_k` with the eval, not by feel.

Truncation check: log `prompt_eval_count` from every Ollama response in the
Langfuse trace. If it equals `num_ctx`, the prompt was truncated. Treat that as
a failed request.

---

## 1. Answer generation (hermes3-rag)

```python
REFUSAL = "I could not find this in the provided documents."

SYSTEM_PROMPT = f"""You answer questions about documents. Use ONLY the CONTEXT the user provides.

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

USER_TEMPLATE = """CONTEXT:
{context}

QUESTION: {question}

Answer using only the CONTEXT above. Cite chunk ids like [C1]."""
```

Context block format (one per retrieved chunk, best-ranked first):

```
<chunk id="C1" type="text" doc="report.pdf" page="3">
...chunk text...
</chunk>
<chunk id="C2" type="table" doc="report.pdf" page="5">
| Quarter | Revenue |
|---|---|
| Q2 | 4.2M |
</chunk>
<chunk id="C3" type="image" doc="report.pdf" page="7">
...caption text...
</chunk>
```

Why it is built this way:
- The exact refusal string lets the eval measure "correctly refused" vs
  "hallucinated" on unanswerable questions. Put unanswerable questions in the
  golden set.
- Chunk ids in `[C#]` make citations machine-checkable (see section 4).
- The answer is meant to be spoken, so rule 6 asks for plain text. Citation
  markers are stripped before TTS (section 4).

---

## 2. Image / chart captioning (moondream)

moondream is tiny. It ignores long instructions. Use short, single-task
prompts, temperature 0, and run both on every image (results cached by image
hash). Do not build an image-type classifier.

```python
DESCRIBE_PROMPT = "Describe this image in detail."
TEXT_PROMPT = "Read out all text, labels, axis titles and numbers visible in this image."
```

Ollama call shape:
```python
messages=[{"role": "user", "content": PROMPT, "images": [image_b64]}]
options={"temperature": 0}
```

Assemble the caption chunk that gets embedded:

```python
CAPTION_TEMPLATE = (
    "Figure on page {page} of {doc}. "
    "Original caption: {pdf_caption}. "
    "Automatic description: {describe}. "
    "Visible text: {text}. "
    "Note: this description is automatic and numbers may be unreliable."
)
```

- `pdf_caption` is the figure caption Docling extracted from the PDF itself
  (use "none" if absent). It is more reliable than anything moondream
  produces, so it always goes in.
- Store metadata `modality=image`, `auto_generated=true`.
- moondream will misread chart numbers. That is expected; the Phase 3 gate
  measures it. Do not paper over it with a longer prompt.

---

## 3. Embedding prefixes (nomic-embed-text)

```python
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "
```

Prepend `DOC_PREFIX` to every chunk (text, table markdown, caption) at index
time and `QUERY_PREFIX` to every user query at retrieval time. Same model and
same prefixes for both sides, always.

---

## 4. Cheap programmatic checks (run before any LLM judge)

These cost nothing, need no judge model, and catch most local-model failures.
Log each as a metric in the trace.

```python
import re

def cited_ids(answer):
    return set(re.findall(r"\[(C\d+)\]", answer))

def invalid_citations(answer, provided_ids):
    return cited_ids(answer) - set(provided_ids)      # ids the model invented

def is_refusal(answer):
    return answer.strip() == REFUSAL

def ungrounded_numbers(answer, context_text):
    # Whole-number match only: "4" must not match inside "2014" or "4,332",
    # and "58" must not match "58.54" (rounded numbers are flagged).
    clean = re.sub(r"\[C\d+\]", "", answer)
    ctx = context_text.replace(",", "")
    nums = {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", clean)}
    return [n for n in nums
            if n and not re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", ctx)]

def to_speakable(answer):
    t = re.sub(r"\[C\d+\]", "", answer)
    t = re.sub(r"[*_#`>|]", "", t)
    return re.sub(r"\s+", " ", t).strip()             # goes to TTS
```

Metrics to track from these:
- `invalid_citation_rate` (should be ~0)
- `uncited_answer_rate` (non-refusal answers with zero citations)
- `ungrounded_number_rate` (heuristic: computed values will flag; use it as a
  review signal, not a hard fail)
- `refusal_rate` on answerable questions (too high = retrieval or prompt
  problem) and on unanswerable questions (should be high)

---

## 5. Not included, on purpose

- No query-rewriting prompt, no multi-step "agent" prompt, no self-critique
  loop. Each adds a model call and a failure mode; design.md forbids them for
  v1.
- No RAGAS prompts. RAGAS ships its own; the judge model rules are in
  tools-requirements.md.
"""Closed-book control (eval-harness addition, not a pipeline module).

Runs the same verified golden-set questions against the same LLM_MODEL and the same
per-request options as generation/generator.py (temperature=0.1; num_ctx/num_predict/
temperature-at-rest come from the hermes3-rag Modelfile either way, since it's the same
model), but with NO retrieved context and a minimal system prompt instead of the RAG
system prompt. This measures what the model answers from parametric (training) knowledge
alone, so a correct RAG answer can be told apart from "the model already knew it."

Never imported by generation/generator.py or any other pipeline module.

Usage: python -m eval.closed_book
"""
import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from eval.run_eval import TYPE_ORDER, _load_questions, _mean, expected_numbers_recall
from generation.generator import MODEL, _BASE_URL

load_dotenv()

RESULTS_DIR = Path(__file__).parent / "results_baseline_v0"
SYSTEM_PROMPT = "Answer in at most 2 sentences."

# Heuristic only, for the summary count below -- read the saved answers before trusting this
# for anything else. The closed-book system prompt gives the model no refusal instruction at
# all, so "declined" here means the model's own wording happened to hedge, not that it matched
# a known REFUSAL string the way the RAG generator does.
_DECLINE_PHRASES = (
    "i don't know", "i do not know", "i don't have", "i do not have",
    "i cannot", "i can't", "no information", "not aware", "unable to",
)

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(base_url=_BASE_URL, api_key="ollama")
    return _client


def _closed_book_answer(question: str) -> str:
    resp = _get_client().chat.completions.create(
        model=MODEL,
        temperature=0.1,  # matches the per-request option generation/generator.py passes
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content.strip()


def _looks_like_decline(answer: str) -> bool:
    low = answer.lower()
    return any(p in low for p in _DECLINE_PHRASES)


def _baseline_question_ids() -> list[str]:
    """The question ids results_baseline_v0/retrieval.json actually covers. golden_set.json
    has since had more questions marked verified=true (see manifest.json's note), so reading
    verified questions live would silently run this control against a different, larger set
    than the RAG baseline it's meant to be compared against.
    """
    baseline = json.loads((RESULTS_DIR / "retrieval.json").read_text(encoding="utf-8"))
    return [row["id"] for row in baseline]


def run() -> dict:
    baseline_ids = set(_baseline_question_ids())
    questions = [q for q in _load_questions() if q["id"] in baseline_ids]
    missing = baseline_ids - {q["id"] for q in questions}
    if missing:
        raise RuntimeError(f"question id(s) in results_baseline_v0/retrieval.json not found in golden_set.json: {missing}")

    rows = []
    for q in questions:
        answer = _closed_book_answer(q["question"])
        rows.append({
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "answer": answer,
            "declined_heuristic": _looks_like_decline(answer),
        })

    recall_by_type = {}
    for t in TYPE_ORDER:
        values = [
            expected_numbers_recall(r["expected_answer"], r["answer"])
            for r in rows if r["type"] == t and r["expected_answer"] != "REFUSAL"
        ]
        mean = _mean(values)
        if mean is not None:
            recall_by_type[t] = mean

    unanswerable = [r for r in rows if r["expected_answer"] == "REFUSAL"]
    unanswerable_answered = [r["id"] for r in unanswerable if not r["declined_heuristic"]]

    result = {
        "label": "closed_book",
        "model": MODEL,
        "system_prompt": SYSTEM_PROMPT,
        "rows": rows,
        "expected_numbers_recall_by_type": recall_by_type,
        "unanswerable_questions_answered": unanswerable_answered,
        "unanswerable_questions_total": len(unanswerable),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "closed_book.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    out = run()
    print(f"Closed-book control ({len(out['rows'])} questions) -> {RESULTS_DIR / 'closed_book.json'}")
    print("expected_numbers_recall by type (parametric-knowledge recall, no context):")
    for t, v in out["expected_numbers_recall_by_type"].items():
        print(f"  {t:14s} {v:.3f}")
    print(f"Unanswerable questions answered (heuristic, verify manually): "
          f"{len(out['unanswerable_questions_answered'])}/{out['unanswerable_questions_total']} "
          f"{out['unanswerable_questions_answered']}")

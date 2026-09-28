"""Run retrieval + programmatic-check (+ optional judge) metrics against eval/golden_set.json.

Usage:
    python -m eval.run_eval            # retrieval + programmatic checks only (fast, no judge -- dev loop)
    python -m eval.run_eval --smoke    # same, first 5 verified questions, spread across types
    python -m eval.run_eval --gate     # also runs the RAGAS judge stage (requires JUDGE_MODEL);
                                        # this is the full phase eval gate, not a routine dev-loop run
    python -m eval.run_eval --stage retrieval  # retrieval + content-level metrics only -- no LLM
                                                # generation call, no answer-based checks, no judge

Golden set schema (eval/golden_set.json): {"questions": [{id, type: text|table|chart|multi|image|
unanswerable, question, expected_answer, relevant: [{doc, page}], verified}], "table_cell_ground_truth":
[...]}. Retrieval metrics match retrieved chunks against "relevant" on (doc, page), not chunk_id, so
re-chunking a document doesn't invalidate the golden set. Entries with verified=false are drafts (per
the golden set's own _readme) and are excluded from every run -- never promoted to verified here.

Runs in three independent stages, each writing its own file to eval/results/. A judge failure
(bad JSON, model unavailable) never blocks or erases the retrieval/checks results already on disk:
    1. retrieval  -> eval/results/retrieval.json  (retrieve + generate, per question)
    2. checks     -> eval/results/checks.json     (programmatic checks, prompts.md section 4)
    3. judge      -> eval/results/judge.json      (RAGAS Faithfulness, --gate only)

--stage retrieval short-circuits after a retrieval-only variant of stage 1 (no generate() call)
-> eval/results/retrieval_only.json, then prints retrieval + content-level metrics (recall@10,
MRR, expected_numbers_recall) and returns; it never reaches checks/judge, since both need a
generated answer. Each row also carries retrieval_ms, and generation_ms when generation ran.

Phase gates, per phasewise.md:
  Phase 1 (text):        Recall@10 > 0.85, MRR > 0.7, RAGAS Faithfulness > 0.9
  Phase 2 (table):       no regression on Phase 1, Recall@10 > 0.85, Table cell F1 > 0.9
  Phase 3 (image+chart): no regression on Phase 1/2, Recall@10 > 0.8
Refusal rate on unanswerable questions is reported separately -- it is not part of any phase gate.
"""
import argparse
import json
import math
import os
import re
import statistics
import sys
import time
from pathlib import Path

from generation.generator import MODEL, REFUSAL, _SYSTEM_PROMPT, _USER_TEMPLATE, _context_block, generate
from index.vector_store import COLLECTION, get_by_id, indexed_docs
from retrieval.retriever import TOP_N, retrieve

GOLDEN_SET = Path(__file__).parent / "golden_set.json"
RESULTS_DIR = Path(__file__).parent / "results"

TYPE_ORDER = ("text", "table", "chart", "multi", "image", "unanswerable")

# Large enough to rank every chunk in the currently-ingested Kohli-only collection (23 chunks)
# so gold_rank is visible even when the gold chunk falls outside the TOP_N=10 the pipeline
# actually uses; bump this once more documents are ingested. Diagnostic-only: never fed to
# generate(), never changes what the pipeline retrieves or sends to the LLM.
GOLD_RANK_TOP_N = 50

GATE_PHASE1 = {"recall_at_10": 0.85, "mrr": 0.7, "faithfulness": 0.9}
GATE_PHASE2 = {"table_recall_at_10": 0.85, "table_cell_f1": 0.9}
GATE_PHASE3 = {"image_recall_at_10": 0.8}


def _load_golden_set() -> dict:
    return json.loads(GOLDEN_SET.read_text(encoding="utf-8"))


def _load_questions() -> list[dict]:
    return _load_golden_set()["questions"]


def _verified_questions(questions: list[dict]) -> list[dict]:
    """Drop verified=false drafts (per the golden set's _readme) and print the count kept per type."""
    verified = [q for q in questions if q.get("verified")]
    skipped = len(questions) - len(verified)
    if skipped:
        print(f"NOTE: {skipped} unverified (verified=false) question(s) skipped")

    print("Questions used per type (verified=true only):")
    seen_types = {q["type"] for q in verified}
    for t in (*TYPE_ORDER, *sorted(seen_types - set(TYPE_ORDER))):
        print(f"  {t:14s} {sum(1 for q in verified if q['type'] == t)}")
    return verified


def _smoke_sample(verified_questions: list[dict], n: int = 5) -> list[dict]:
    """Pick n questions spread across types (round-robin by TYPE_ORDER), not just the first n in
    file order -- file order happens to cluster by type/doc, which would make --smoke exercise
    only one type (or come up empty if the first n in file order aren't verified yet).
    """
    by_type: dict[str, list[dict]] = {}
    for q in verified_questions:
        by_type.setdefault(q["type"], []).append(q)
    types = [t for t in TYPE_ORDER if t in by_type] + sorted(set(by_type) - set(TYPE_ORDER))

    sample = []
    i = 0
    while len(sample) < n and any(by_type[t] for t in types):
        t = types[i % len(types)]
        if by_type[t]:
            sample.append(by_type[t].pop(0))
        i += 1

    print(f"Smoke sample ({len(sample)} question(s)): {[q['id'] for q in sample]}")
    return sample


def _assert_docs_indexed(verified_questions: list[dict]) -> None:
    """Every verified question's relevant.doc must exist as a payload "doc" value in the
    collection, or its recall is silently 0 because of a filename mismatch, not a retrieval
    failure -- fail loudly instead of reporting a false regression.
    """
    expected = {r["doc"] for q in verified_questions for r in q.get("relevant", [])}
    if not expected:
        return
    indexed = indexed_docs()
    missing = sorted(expected - indexed)
    if missing:
        raise RuntimeError(
            f"golden_set.json references doc(s) not present in the '{COLLECTION}' collection: {missing}. "
            f"Indexed docs: {sorted(indexed)}. Check for a filename mismatch (spaces/underscores/case) "
            "between the ingested file and golden_set.json's \"doc\" field, or re-ingest."
        )


# --- programmatic checks (prompts.md section 4, verbatim) ---

def cited_ids(answer: str) -> set[str]:
    return set(re.findall(r"\[(C\d+)\]", answer))


def invalid_citations(answer: str, provided_ids: list[str]) -> set[str]:
    return cited_ids(answer) - set(provided_ids)  # ids the model invented


def is_refusal(answer: str) -> bool:
    return answer.strip() == REFUSAL


def ungrounded_numbers(answer: str, context_text: str) -> list[str]:
    # Whole-number match only: "4" must not match inside "2014" or "4,332",
    # and "58" must not match "58.54" (rounded numbers are flagged).
    clean = re.sub(r"\[C\d+\]", "", answer)
    ctx = context_text.replace(",", "")
    nums = {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", clean)}
    return [n for n in nums
            if n and not re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", ctx)]


def _whole_numbers(text: str) -> set[str]:
    clean = re.sub(r"\[C\d+\]", "", text)
    return {n.replace(",", "").rstrip(".") for n in re.findall(r"\d[\d,]*\.?\d*", clean)}


def expected_numbers_in_context(expected_answer: str, context_text: str) -> list[str]:
    """Whole numbers from expected_answer that DO appear in the retrieved context (same
    whole-number matching as ungrounded_numbers, run in the opposite direction): this tells
    you whether the retrieved chunks even contain the numbers needed to answer correctly,
    independent of what the generator did with them.
    """
    ctx = context_text.replace(",", "")
    nums = _whole_numbers(expected_answer)
    return [n for n in nums
            if n and re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", ctx)]


def expected_numbers_recall(expected_answer: str, context_text: str) -> float | None:
    """Fraction of expected_answer's whole numbers found in the retrieved context.
    None when expected_answer has no numbers (also true of REFUSAL -- skip those questions
    at the call site rather than relying on this returning None for them).
    """
    nums = _whole_numbers(expected_answer)
    if not nums:
        return None
    return len(expected_numbers_in_context(expected_answer, context_text)) / len(nums)


def _mean(values: list[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return statistics.mean(values) if values else None


# --- gold-chunk detection (headline retrieval metrics) ---

def _has_gold_domain(q: dict) -> bool:
    """Whether q is in-scope for gold_hit@5/@10/gold_present: an answerable (non-"unanswerable")
    question that has gold_terms or at least one number in expected_answer. Out-of-scope
    questions (no gold_terms/numbers, or type=="unanswerable") are excluded from those metrics
    entirely; in-scope questions with no matching chunk (gold chunk NONE) count as a miss (0.0),
    never excluded.
    """
    return q["type"] != "unanswerable" and (bool(q.get("gold_terms")) or bool(_whole_numbers(q["expected_answer"])))


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _gold_terms_match(text: str, terms: list[str]) -> bool:
    """All gold_terms must be found in text. A term made only of digits/commas uses the same
    whole-number matching as ungrounded_numbers (comma-normalised, no adjacent digit/decimal);
    any other term is a case-insensitive, whitespace-normalised substring match.
    """
    ctx_nums = text.replace(",", "")
    normalized_text = _normalize_ws(text).lower()
    for term in terms:
        if re.fullmatch(r"[\d,]+", term):
            num = term.replace(",", "")
            if not re.search(rf"(?<![\d.]){re.escape(num)}(?!\d|\.\d)", ctx_nums):
                return False
        elif _normalize_ws(term).lower() not in normalized_text:
            return False
    return True


def _gold_match(chunk_text: str, q: dict) -> bool:
    """Whether chunk_text is a gold chunk for q. Uses q["gold_terms"] when present (all must
    match); otherwise falls back to requiring every whole number in expected_answer to appear.
    Reads only displayed text (chunk_text) -- gold detection never reads embed_text.
    """
    gold_terms = q.get("gold_terms")
    if gold_terms:
        return _gold_terms_match(chunk_text, gold_terms)
    nums = _whole_numbers(q["expected_answer"])
    if not nums:
        return False  # no gold_terms and no numbers: gold chunk is undefined for this question
    ctx = chunk_text.replace(",", "")
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?!\d|\.\d)", ctx) for n in nums)


def _gold_rank(q: dict) -> tuple[int | None, float | None, str | None]:
    """Rank (1-indexed), score, and chunk_id of the highest-ranked chunk matching _gold_match,
    searched across GOLD_RANK_TOP_N chunks -- a separate diagnostic retrieve() call, so a gold
    chunk outside the TOP_N=10 the pipeline actually uses is still visible. (None, None, None)
    when the question has neither gold_terms nor any numbers in expected_answer, or no chunk matches.
    """
    if not q.get("gold_terms") and not _whole_numbers(q["expected_answer"]):
        return None, None, None
    ranked = retrieve(q["question"], document_scope=None, top_n=GOLD_RANK_TOP_N)
    for i, c in enumerate(ranked, 1):
        if _gold_match(c["text"], q):
            return i, c["score"], c["chunk_id"]
    return None, None, None


def _estimate_prompt_tokens(question: str, chunks: list[dict]) -> int:
    """Rough ~4-chars-per-token estimate of the prompt generate() would send for this question/
    chunks pair -- no tokenizer dependency, so this works even when generate() never ran
    (--stage retrieval). Report-only figure, not used for any gate.
    """
    prompt = _SYSTEM_PROMPT + _USER_TEMPLATE.format(context=_context_block(chunks), question=question)
    return round(len(prompt) / 4)


# --- stage 1: retrieval + generation ---

def _score_retrieval(q: dict, chunks: list[dict]) -> tuple[float | None, float | None]:
    """Recall@10 and reciprocal rank, matched on (doc, page) -- not chunk_id, per the golden set schema."""
    relevant = {(r["doc"], r["page"]) for r in q.get("relevant", [])}
    if not relevant:
        return None, None
    retrieved_pairs = [(c["doc"], c["page"]) for c in chunks]
    recall = len(relevant & set(retrieved_pairs)) / len(relevant)
    rank = next((i for i, pair in enumerate(retrieved_pairs, 1) if pair in relevant), None)
    return recall, (1 / rank if rank else 0.0)


def run_retrieval_stage(questions: list[dict], with_generation: bool = True) -> list[dict]:
    """with_generation=False (--stage retrieval) skips the generate() call entirely -- retrieval
    only, no LLM generation call. Always records retrieval_ms; generation_ms is added only when
    generation actually ran.

    with_generation=True (the full, non --stage run) prints one progress line per question as it
    finishes (id, type, generation seconds, prompt_eval_count), and writes results to disk after
    each question so a crash mid-run keeps the finished ones.

    In both with_generation modes, prints one line per question with id, type, gold chunk id
    (or NONE), and rank.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    filename = "retrieval.json" if with_generation else "retrieval_only.json"
    out_path = RESULTS_DIR / filename

    rows = []
    for q in questions:
        t0 = time.perf_counter()
        chunks = retrieve(q["question"], document_scope=None, top_n=TOP_N)
        retrieval_ms = (time.perf_counter() - t0) * 1000
        recall, mrr = _score_retrieval(q, chunks)
        gold_rank, gold_score, gold_chunk_id = _gold_rank(q)
        gold_domain = _has_gold_domain(q)
        row = {
            "id": q["id"],
            "type": q["type"],
            "question": q["question"],
            "gold_domain": gold_domain,
            "gold_rank": gold_rank,
            "gold_score": gold_score,
            "gold_hit_at_5": None if not gold_domain else (1.0 if gold_rank is not None and gold_rank <= 5 else 0.0),
            "gold_hit_at_10": None if not gold_domain else (1.0 if gold_rank is not None and gold_rank <= 10 else 0.0),
            "recall_at_10": recall,  # page-level (can be misleading) -- see _print_retrieval_metrics
            "mrr": mrr,
            "retrieval_ms": retrieval_ms,
            "chunks_to_generator": len(chunks),
            "prompt_token_estimate": _estimate_prompt_tokens(q["question"], chunks),
            "chunks": [{"chunk_id": c["chunk_id"], "doc": c["doc"], "page": c["page"], "text": c["text"]} for c in chunks],
        }
        chunk_label = gold_chunk_id if gold_chunk_id is not None else "NONE"
        print(f"  [{q['id']}] type={q['type']} gold_chunk={chunk_label} rank={gold_rank}")
        if with_generation:
            t1 = time.perf_counter()
            result = generate(q["question"], chunks)
            generation_ms = (time.perf_counter() - t1) * 1000
            row["generation_ms"] = generation_ms
            row["answer"] = result["text_answer"]
            prompt_eval_count = result["usage"]["input"] if result["usage"] else None
            print(f"  [{q['id']}] type={q['type']} generation_s={generation_ms / 1000:.2f} "
                  f"prompt_eval_count={prompt_eval_count}")
        rows.append(row)
        if with_generation:
            out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if not with_generation:
        out_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


# --- stage 2: programmatic checks + table cell F1 ---

def _parse_markdown_cells(markdown: str) -> set[str]:
    """Flatten a Markdown table's data cells into a set of trimmed strings, skipping the `---` separator row."""
    cells = set()
    for line in markdown.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if set(line.replace("|", "").replace(":", "").strip()) <= {"-", " "}:
            continue  # header separator row, e.g. "| --- | --- |"
        for cell in line.strip("|").split("|"):
            cell = cell.strip()
            if cell:
                cells.add(cell)
    return cells


def _table_cell_f1() -> dict[str, float]:
    """Per-entry F1 between a hand-annotated table's expected cells and the ingested chunk's parsed cells."""
    ground_truth = _load_golden_set().get("table_cell_ground_truth", [])
    scores = {}
    for entry in ground_truth:
        if entry["expected_cells"] == ["PLACEHOLDER"]:
            continue
        payload = get_by_id(entry["chunk_id"])
        if payload is None:
            continue  # chunk not ingested yet
        parsed = _parse_markdown_cells(payload["text"])
        expected = set(entry["expected_cells"])
        overlap = len(parsed & expected)
        precision = overlap / len(parsed) if parsed else 0.0
        recall = overlap / len(expected) if expected else 0.0
        scores[entry["id"]] = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return scores


def _content_metrics(rows: list[dict]) -> tuple[dict[str, dict], dict[str, float]]:
    """expected_numbers_in_context / expected_numbers_recall per question, plus the by-type
    summary -- computed from retrieved chunk text vs expected_answer, no generated answer
    needed. Shared by run_checks_stage (full run) and --stage retrieval (no generation).
    """
    expected_by_id = {q["id"]: q["expected_answer"] for q in _load_questions()}
    per_question = {}
    for row in rows:
        expected = expected_by_id.get(row["id"], "")
        if expected == REFUSAL:  # REFUSAL has no numbers to check context grounding against
            continue
        context_text = "\n".join(c["text"] for c in row["chunks"])
        per_question[row["id"]] = {
            "expected_numbers_in_context": expected_numbers_in_context(expected, context_text),
            "expected_numbers_recall": expected_numbers_recall(expected, context_text),
        }

    by_type = {}
    for t in TYPE_ORDER:
        values = [per_question[r["id"]]["expected_numbers_recall"] for r in rows
                  if r["id"] in per_question and r["type"] == t]
        mean = _mean(values)
        if mean is not None:
            by_type[t] = mean
    return per_question, by_type


def run_checks_stage(rows: list[dict]) -> dict:
    content_per_question, expected_numbers_recall_by_type = _content_metrics(rows)

    per_question = {}
    for row in rows:
        provided_ids = [f"C{i}" for i in range(1, len(row["chunks"]) + 1)]
        context_text = "\n".join(c["text"] for c in row["chunks"])
        entry = {
            "invalid_citations": sorted(invalid_citations(row["answer"], provided_ids)),
            "cited": bool(cited_ids(row["answer"])),
            "is_refusal": is_refusal(row["answer"]),
            "ungrounded_numbers": ungrounded_numbers(row["answer"], context_text),
        }
        entry.update(content_per_question.get(row["id"], {}))
        per_question[row["id"]] = entry

    answerable = [r for r in rows if r["type"] != "unanswerable"]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    cell_f1 = _table_cell_f1()

    summary = {
        "invalid_citation_rate": _mean([1.0 if per_question[r["id"]]["invalid_citations"] else 0.0 for r in rows]),
        "uncited_answer_rate": _mean(
            [0.0 if (per_question[r["id"]]["is_refusal"] or per_question[r["id"]]["cited"]) else 1.0 for r in rows]
        ),
        "ungrounded_number_rate": _mean([1.0 if per_question[r["id"]]["ungrounded_numbers"] else 0.0 for r in rows]),
        "refusal_rate_answerable": _mean([1.0 if per_question[r["id"]]["is_refusal"] else 0.0 for r in answerable]),
        "refusal_rate_unanswerable": _mean([1.0 if per_question[r["id"]]["is_refusal"] else 0.0 for r in unanswerable]),
        "table_cell_f1": _mean(list(cell_f1.values())),
        "expected_numbers_recall_by_type": expected_numbers_recall_by_type,
    }
    result = {"per_question": per_question, "table_cell_f1_per_entry": cell_f1, "summary": summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "checks.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# --- stage 3: judge (RAGAS Faithfulness), --gate only ---

def run_judge_stage(rows: list[dict]) -> dict:
    """JUDGE_MODEL is read fresh from the environment and must differ from the generator model
    (tools-requirements.md: hermes3 must not grade hermes3). No default/fallback model here --
    a gate run must pick its judge explicitly.
    """
    judge_model = os.environ.get("JUDGE_MODEL")
    if not judge_model:
        raise RuntimeError("JUDGE_MODEL env var must be set to run the judge stage (--gate)")
    if judge_model == MODEL:
        raise RuntimeError(f"JUDGE_MODEL ({judge_model}) must differ from the generator model ({MODEL})")

    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import Faithfulness

    judge_client = AsyncOpenAI(api_key="ollama", base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))
    judge = llm_factory(judge_model, provider="openai", client=judge_client)
    metric = Faithfulness(llm=judge)

    scores = {}
    nan_ids = []
    for row in rows:
        if row["type"] == "unanswerable" or not row["chunks"]:
            continue  # nothing to check faithfulness against
        result = metric.score(
            user_input=row["question"], response=row["answer"], retrieved_contexts=[c["text"] for c in row["chunks"]]
        )
        value = float(result.value)
        if math.isnan(value):
            nan_ids.append(row["id"])  # malformed judge JSON -- count it, never silently drop it
        else:
            scores[row["id"]] = value

    judged = len(scores) + len(nan_ids)
    result = {
        "judge_model": judge_model,
        "scores": scores,
        "nan_ids": nan_ids,
        "nan_rate": (len(nan_ids) / judged) if judged else 0.0,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "judge.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _print_gate(title: str, gate: dict[str, float], values: dict[str, float | None]) -> bool:
    print(f"\n=== {title} ===")
    all_passed = True
    for name, threshold in gate.items():
        value = values.get(name)
        passed = value is not None and value >= threshold
        all_passed &= passed
        shown = f"{value:.3f}" if value is not None else "n/a"
        print(f"  {name:24s} {shown:>6s}  (>= {threshold})  {'PASS' if passed else 'FAIL'}")
    return all_passed


def _print_retrieval_metrics(rows: list[dict]) -> None:
    for t in TYPE_ORDER:
        rs = [r for r in rows if r["type"] == t]
        if not rs:
            continue
        domain_rs = [r for r in rs if r["gold_domain"]]
        gold_hit5 = _mean([r["gold_hit_at_5"] for r in rs])
        gold_hit10 = _mean([r["gold_hit_at_10"] for r in rs])
        gold_present = _mean([1.0 if r["gold_rank"] is not None else 0.0 for r in domain_rs])
        gold_ranks = [r["gold_rank"] for r in rs if r["gold_rank"] is not None]
        avg_gold_rank = statistics.mean(gold_ranks) if gold_ranks else None
        recall, mrr = _mean([r["recall_at_10"] for r in rs]), _mean([r["mrr"] for r in rs])
        avg_ms = _mean([r["retrieval_ms"] for r in rs])
        print(f"\n=== {t} (n={len(rs)}, gold-domain n={len(domain_rs)}) ===")
        print(f"  gold_hit@5   {gold_hit5 if gold_hit5 is None else f'{gold_hit5:.3f}'}")
        print(f"  gold_hit@10  {gold_hit10 if gold_hit10 is None else f'{gold_hit10:.3f}'}")
        print(f"  gold_present (some chunk contains all gold terms) {gold_present if gold_present is None else f'{gold_present:.3f}'}")
        print(f"  gold_rank (mean over questions whose gold chunk exists) "
              f"{avg_gold_rank if avg_gold_rank is None else f'{avg_gold_rank:.1f}'}")
        print(f"  page-level recall_at_10 (can be misleading) {recall if recall is None else f'{recall:.3f}'}")
        print(f"  mrr          {mrr if mrr is None else f'{mrr:.3f}'}")
        print(f"  retrieval_ms {avg_ms if avg_ms is None else f'{avg_ms:.1f}'} (mean)")

    chunks_to_gen = _mean([r["chunks_to_generator"] for r in rows])
    prompt_tokens = _mean([r["prompt_token_estimate"] for r in rows])
    print(f"\n=== chunks/prompt (n={len(rows)}) ===")
    print(f"  chunks_to_generator (mean) {chunks_to_gen if chunks_to_gen is None else f'{chunks_to_gen:.1f}'}")
    print(f"  prompt_token_estimate (mean, ~4 chars/token) "
          f"{prompt_tokens if prompt_tokens is None else f'{prompt_tokens:.0f}'}")
    if chunks_to_gen is not None and chunks_to_gen > 5:
        print(f"  NOTE: generator receives {chunks_to_gen:.1f} chunks on average (TOP_N={TOP_N}), "
              f"which is more than 5. Not changed, per instructions -- flagging only.")


def run(smoke: bool = False, gate: bool = False, stage: str = "full") -> bool:
    questions = _load_questions()
    verified_questions = _verified_questions(questions)
    if smoke:
        verified_questions = _smoke_sample(verified_questions)
    _assert_docs_indexed(verified_questions)

    if stage == "retrieval":
        print(f"\nRetrieval-only stage ({len(verified_questions)} questions, no generation call)...")
        rows = run_retrieval_stage(verified_questions, with_generation=False)
        _print_retrieval_metrics(rows)
        _, by_type_recall = _content_metrics(rows)
        print(f"\n=== content-level metrics (n={len(rows)}) ===")
        print("  expected_numbers_recall_by_type  " +
              (", ".join(f"{t}={r:.3f}" for t, r in by_type_recall.items()) if by_type_recall else "n/a"))
        print("\n(--stage retrieval: no generation call, no answer-based checks, no judge -- "
              "pass no --stage for that)")
        return True

    print(f"\nStage 1/3: retrieval + generation ({len(verified_questions)} questions)...")
    rows = run_retrieval_stage(verified_questions)

    print("Stage 2/3: programmatic checks...")
    checks = run_checks_stage(rows)

    def by_type(t: str) -> list[dict]:
        return [r for r in rows if r["type"] == t]

    _print_retrieval_metrics(rows)

    print(f"\n=== programmatic checks (n={len(rows)}) ===")
    for k, v in checks["summary"].items():
        if k == "expected_numbers_recall_by_type":
            print(f"  {k:28s} " + ", ".join(f"{t}={r:.3f}" for t, r in v.items()) if v else f"  {k:28s} n/a")
        else:
            print(f"  {k:28s} {v if v is None else f'{v:.3f}'}")

    if not gate:
        print("\n(dev-loop run: no judge call, no phase pass/fail determination -- pass --gate for that)")
        return True

    print("\nStage 3/3: judge (RAGAS Faithfulness)...")
    faithfulness = None
    try:
        judge = run_judge_stage(rows)
        faithfulness = _mean(list(judge["scores"].values()))
        if judge["nan_ids"]:
            print(f"  WARNING: judge returned malformed JSON on {len(judge['nan_ids'])} question(s): {judge['nan_ids']}")
    except Exception as exc:
        print(f"  JUDGE STAGE FAILED: {exc}")
        print("  Retrieval and checks results above are unaffected -- already written to eval/results/.")

    text_rows = by_type("text")
    phase1_values = {
        "recall_at_10": _mean([r["recall_at_10"] for r in text_rows]),
        "mrr": _mean([r["mrr"] for r in text_rows]),
        "faithfulness": faithfulness,
    }
    phase1_passed = _print_gate(f"Phase 1 gate (text-only, n={len(text_rows)})", GATE_PHASE1, phase1_values)

    table_rows = by_type("table")
    phase2_values = {
        "table_recall_at_10": _mean([r["recall_at_10"] for r in table_rows]),
        "table_cell_f1": checks["summary"]["table_cell_f1"],
    }
    phase2_passed = _print_gate(f"Phase 2 gate (tables, n={len(table_rows)})", GATE_PHASE2, phase2_values)
    if not phase1_passed:
        print("  NOTE: Phase 2 gate also requires no regression on the Phase 1 gate above, which is currently failing.")

    image_chart_rows = by_type("image") + by_type("chart")
    phase3_values = {"image_recall_at_10": _mean([r["recall_at_10"] for r in image_chart_rows])}
    phase3_passed = _print_gate(f"Phase 3 gate (image+chart, n={len(image_chart_rows)})", GATE_PHASE3, phase3_values)
    if not (phase1_passed and phase2_passed):
        print("  NOTE: Phase 3 gate also requires no regression on the Phase 1/2 gates above.")

    return phase1_passed and phase2_passed and phase3_passed


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--smoke", action="store_true", help="run only the first 5 questions")
    arg_parser.add_argument("--gate", action="store_true", help="also run the RAGAS judge stage (requires JUDGE_MODEL)")
    arg_parser.add_argument(
        "--stage", choices=["full", "retrieval"], default="full",
        help="'retrieval': retrieval + content-level metrics only, no generation call, no answer-based checks"
    )
    args = arg_parser.parse_args()
    if args.stage == "retrieval" and args.gate:
        arg_parser.error("--gate requires generated answers; incompatible with --stage retrieval")
    sys.exit(0 if run(smoke=args.smoke, gate=args.gate, stage=args.stage) else 1)

# Baseline v0 -- per-question report

Computed offline from `eval/results_baseline_v0/retrieval.json` and `checks.json`. No RAG
generation was re-run. See `manifest.json` for run/model/prompt provenance, including the
warning that this baseline covers 11 of the golden set's now-15 verified questions.

**timings**: not available. Neither `retrieval.json` nor `checks.json` records any timing
field, and there's no third file with per-question latency for this baseline -- that data
was simply never saved. Left as `n/a` in every row below rather than fabricated or re-run.

**page-hit** is Recall@10 at the (doc, page) level (golden set's own matching rule, not
chunk-level) -- Y if any golden `relevant` page for the question appears anywhere in the
10 retrieved chunks. For k_b01/k_b04/k_b06/k_c01/k_c04, the *page* was hit via a text chunk,
but the specific table/image chunk holding the actual numeric answer was **not** among the
10 retrieved chunks -- noted per row, since it explains the refusal/wrong-answer pattern
better than the page-level Y suggests.

| id | type | expected answer | generated answer | page-hit | retrieved top-5 chunk ids | refused | unanswerable-refusal-correct | cited ids | timings |
|---|---|---|---|---|---|---|---|---|---|
| k_t01 | text | Rajkumar Sharma. | According to the report, Kohli was trained under the watchful eye of Rajkumar Sharma in Delhi's domestic circuit. [C1] | Y (rank 1) | p2:c0, p3:c0, p1:c0, p12:c0, p7:c0 | N | n/a | p2:c0 | n/a |
| k_t02 | text | Afghanistan, in the Asia Cup 2022 (122*). | Kohli scored his only T20I century against Afghanistan in the Asia Cup 2022 [C1]. | Y (rank 1) | p7:c0, p8:c0, p3:c0, p2:c0, p5:c0 | N | n/a | p7:c0 | n/a |
| k_t04 | text | MS Dhoni; a ruthless, fast-bowling-oriented, fitness-driven culture. | Kohli took over the Test captaincy from MS Dhoni. He instilled a ruthless, fast-bowling-oriented, and fitness-driven culture [C1]. | Y (rank 1) | p9:c0, p2:c0, p12:c0, p3:c0, p1:c0 | N | n/a | p9:c0 | n/a |
| k_b01 | table | 4,332. | I could not find this in the provided documents. | Y (rank 2, via p3:c0 text; table chunk p3:t0 **not** retrieved) | p12:c0, p3:c0, p8:c0, p5:c0, p7:c0 | Y | n/a | none | n/a |
| k_b04 | table | 94.15. | The provided documents do not contain the specific ODI strike rate information for Virat Kohli when batting second (chasing). | Y (rank 1, via p5:c0 text; table chunk p5:t0 **not** retrieved) | p5:c0, p7:c0, p3:c0, p6:c0, p8:c0 | N (see note) | n/a | none | n/a |
| k_b06 | table | 95 matches; 65 wins. | I could not find this in the provided documents. | Y (rank 1, via p9:c0 text; table chunk p9:t0 **not** retrieved) | p9:c0, p5:c0, p8:c0, p12:c0, p6:c0 | Y | n/a | none | n/a |
| k_c01 | chart | 1,322. | I could not find this in the provided documents. | Y (rank 2, via p3:c0 text; image chunk p4:i0 **not** retrieved at all, not even outside top 5) | p3:c0, p4:c0, p8:c0, p9:c0, p5:c0 | Y | n/a | none | n/a |
| k_c04 | chart | Sri Lanka, with 10. | Australia [C1]. | Y (rank 1, via p6:c0 text; image chunk p6:i0 **not** retrieved at all) | p6:c0, p5:c0, p3:c0, p10:c0, p7:c0 | N | n/a | p6:c0 | n/a |
| k_m01 | multi | Chasing: 64.21, versus 50.18 batting first. | Kohli's ODI average is higher when he is batting second, or chasing. His average is 64.21 when chasing, compared to 50.18 when batting first [C1]. | Y (rank 1) | p5:c0, p3:c0, p7:c0, p8:c0, p6:c0 | N | n/a | p5:c0 | n/a |
| k_u01 | unanswerable | REFUSAL | I could not find this in the provided documents. | n/a (no relevant page defined) | p3:c0, p8:c0, p4:c0, p5:c0, p12:c0 | Y | **Y** (correctly refused) | none | n/a |
| k_u04 | unanswerable | REFUSAL | Kohli's highest score in ODI cricket was 124 against Pakistan in the Asia Cup 2022 [C5]. | n/a (no relevant page defined) | p5:c0, p7:c0, p8:c0, p3:c0, p6:c0 | N | **N** (wrongly answered -- parametric-knowledge trap; "124" flagged as an ungrounded number in `checks.json`) | p6:c0 | n/a |

## Notes surfaced by this table

- **k_b04 refusal-string miss**: the generator's answer ("The provided documents do not
  contain the specific ODI strike rate information...") is a refusal in plain English but
  does not match the exact `REFUSAL` string the programmatic check (`is_refusal`) requires,
  so `checks.json` scores it as a non-refusal, non-cited, non-hallucinated answer. It's a
  correct outcome (no fabricated number) that the current check can't credit as a refusal --
  worth knowing about if `refusal_rate_answerable` is read as an exact count.
- **k_u04's cited id (C5 -> p6:c0) does not support its answer**: the ODI-century-breakdown
  chunk it cites contains no "124" and no "Pakistan" -- the citation is a valid chunk id
  (not flagged by `invalid_citations`) but not actually grounding for what it's attached to.

## expected_numbers_recall by type

Added via `eval.run_eval.expected_numbers_in_context`/`expected_numbers_recall` (new, item 3),
computed offline against the chunks already saved in `retrieval.json` -- fraction of the
`expected_answer`'s whole numbers that are present anywhere in the 10 retrieved chunks for
that question, skipping REFUSAL questions. Full detail in `checks.json`'s
`per_question[*].expected_numbers_recall` and `summary.expected_numbers_recall_by_type`.

| type | n scored | mean expected_numbers_recall |
|---|---|---|
| text | 1 (k_t02; k_t01/k_t04 have no numbers in their expected_answer) | 1.0 |
| table | 3 (k_b01, k_b04, k_b06) | 0.0 |
| chart | 2 (k_c01, k_c04) | 0.0 |
| multi | 1 (k_m01) | 1.0 |
| unanswerable | 0 (both expected_answer=REFUSAL, skipped) | n/a |

table=0.0 and chart=0.0 are the same retrieval-side story as the notes above: none of the
5 table/chart questions had their ground-truth table/image chunk land in the top 10, even
though page-level Recall@10 shows 1.0 for all of them via a co-located text chunk. This is
a gap `table_cell_f1` (also 1.0 for this baseline) cannot see, since it only checks that a
table chunk *that was retrieved* parses correctly -- it never asks whether retrieval found
the right table chunk in the first place.

## Closed-book control (item 4)

Same 11 questions, same `LLM_MODEL` (hermes3-rag) and per-request options (temperature=0.1),
**no retrieved context**, system prompt replaced with only `"Answer in at most 2 sentences."`
Full answers in `closed_book.json`.

| type | n scored | mean expected_numbers_recall (closed-book) |
|---|---|---|
| text | 1 | 0.0 |
| table | 3 | 0.0 |
| chart | 2 | 0.0 |
| multi | 1 | 0.0 |

Every closed-book answer fabricated plausible-sounding cricket statistics that do not match
this (synthetic) report's numbers -- 0.0 recall across every type, versus 1.0/0.0/0.0/1.0 for
the RAG-grounded run above. That's the expected/good outcome for this control: it confirms
the RAG run's *correct* answers (k_t02, k_m01) are actually coming from retrieved context,
not from the model already "knowing" these (made-up) figures, and it confirms the RAG run's
refusals on k_b01/k_b04/k_b06/k_c01 aren't the model being needlessly conservative -- closed-book
answered those questions too, just wrong, every time.

Both unanswerable questions (k_u01, k_u04) were answered in closed-book mode -- expected,
since the closed-book system prompt carries no refusal instruction at all, unlike the RAG
system prompt. This isn't comparable to the RAG run's refusal behavior; it just confirms the
control has no built-in tendency to decline.


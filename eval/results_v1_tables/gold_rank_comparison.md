# gold_rank before/after: table embed_text change

Change under test: table chunks now embed a deterministic natural-language linearization
(`embed_text`, built from Docling's own cell-level export via `TableItem.export_to_dataframe`)
instead of the raw Markdown. Displayed/cited text is unchanged. Same model
(nomic-embed-text), same prefixes, same TOP_N (10). Text/chart chunking and generation
untouched. Kohli re-ingested only (idempotent upsert); 23 chunks before and after (12 text,
8 table, 3 image).

`before` = `retrieval_only_before.json` (run immediately after adding gold_terms/gold_rank
metrics to run_eval.py, prior to the table change). `after` = `retrieval_only_after.json`
(run after the re-ingest). Both from `python -m eval.run_eval --stage retrieval`.

| id    | type  | gold_rank before | gold_rank after | delta |
|-------|-------|-------------------|------------------|-------|
| k_t01 | text  | None (no numbers) | None (no numbers) | -   |
| k_t02 | text  | 1                  | 1                | 0     |
| k_t04 | text  | None (no numbers) | None (no numbers) | -   |
| k_b01 | table | 17                 | 5                | -12   |
| k_b04 | table | 11                 | 2                | -9    |
| k_b06 | table | 18                 | 9                | -9    |
| k_c01 | chart | None (no match)    | None (no match)  | -     |
| k_c02 | chart | None (no match)    | None (no match)  | -     |
| k_c03 | chart | None (no match)    | None (no match)  | -     |
| k_c04 | chart | None (no match)    | None (no match)  | -     |
| k_c05 | chart | 12                 | 14               | +2    |
| k_c06 | chart | 13                 | 18               | +5    |
| k_m01 | multi | 1                  | 1                | 0     |

## Reading this

- **Text and multi questions did not regress**: k_t02 and k_m01 held gold_rank 1 in both
  runs; k_t01/k_t04 have no numbers in their expected_answer and no gold_terms, so gold_rank
  is undefined (None) in both runs, same as before -- expected, not a regression.
- **All 3 table questions improved, and all moved inside gold_hit@10** (none were inside
  top 10 before): k_b01 17->5, k_b04 11->2, k_b06 18->9. By-type table gold_hit@5 went
  0.000 -> 0.667 and gold_hit@10 went 0.000 -> 1.000 (see run_eval.py console output).
- **k_c01-k_c04 (chart) are unaffected, as expected**: this was a table-only change; their
  gold_terms (e.g. "2018"/"1322") aren't in any table chunk and moondream's chart captions
  still don't reliably contain the plotted numbers (the Phase 3 gap noted in CLAUDE.md), so
  no chunk matches gold_terms before or after.
- **k_c05/k_c06 got slightly worse (+2, +5), but this is not a real chart-retrieval
  regression**: their old gold_rank (12, 13) was already flagged in
  eval/results_baseline_v0/gold_chunk_diagnostic_all13.json as an unreliable false-positive
  match -- a short digit term ("5", "6") coincidentally matching an unrelated number in a
  table/text chunk on a different page, not the real chart data. Re-embedding the tables
  shifted the overall ranking slightly and pushed that coincidental match a bit further down;
  the genuine answer still isn't retrievable from any indexed chunk.

## Chunks/prompt report (step 2, unchanged by this step)

TOP_N=10 chunks are sent to the generator on every question (`chunks_to_generator` mean
10.0, both runs) -- more than 5, flagged per instructions, not changed. Estimated prompt
size (~4 chars/token, no generation call made in `--stage retrieval`): ~2,861 tokens/question
before, ~2,735 after (table chunks got slightly shorter/more regular once linearized).

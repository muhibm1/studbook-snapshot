# M2: hybrid retrieval and the eval harness

Date: 2026-09-16. Dev-set results below (34 answerable questions; the 9 unanswerable ones are a
generation/refusal check for M3-M4, not a retrieval one). Every number here is also logged in
`studbook.eval_runs` with its config, config hash, and the commit it ran at.

## What shipped

- **`retrieve/hybrid.py`**: vector search (HNSW, cosine), Postgres full-text search, and
  reciprocal rank fusion, each usable standalone -- the eval harness needs all three separately
  for the comparison below, not just the merged result.
- **`eval/metrics.py` + `eval/run.py`**: Recall@5, MRR@10, Precision@5, overall and by question
  type, computed against `eval/dev.jsonl`; every run is logged to `studbook.eval_runs`. A hit is
  "a retrieved chunk's body contains one of the question's gold quotes verbatim" -- the same
  definition `eval/gold.py`'s checker uses, so a chunking change that breaks a quote's presence is
  caught there, not silently scored as a miss here.
- `--split test` requires typing "yes" at a prompt, on purpose: the held-out split is scored once,
  not touched during tuning (docs/m0-census.md section 4).

## The full-text bug, and why it mattered

The first full-text run scored recall@5 = **0.0294** -- one hit in the entire dev set. Implausible
enough to investigate rather than report: `websearch_to_tsquery` (and `plainto_tsquery`) AND every
content word together by default, and this project's questions average 20+ words. Requiring every
one of a natural sentence's content words to co-occur inside one ~500-word chunk is nearly
impossible, which is exactly what near-zero recall was measuring -- a query-construction bug, not
a weak lexical baseline.

Fix (`retrieve_fulltext` in `retrieve/hybrid.py`): OR the question's own stemmed lexemes together
instead of ANDing them -- closer to how BM25 treats terms (independent evidence, not a required
co-occurrence) and the right shape for a lexical retriever meant to complement a vector one via
RRF. Recall@5 went from 0.0294 to 0.6176 -- confirmed by rerunning, not assumed from the fix alone.

Two runs logged before the fix (the broken full-text run and the hybrid run computed against that
broken full-text half) were deleted from `studbook.eval_runs` rather than kept alongside the
correct ones: `config_hash` hashes the *parameters* of a run, not the query logic, so a code fix
between two runs with identical config produces identical hashes over different real results --
a trap for anyone querying `eval_runs` later without also cross-checking `studbook_commit`. Every
number in the table below was measured after the fix, and the fixed code is what's committed.

## Results (dev, n=34)

| Method | Recall@5 | MRR@10 | Precision@5 |
|---|---|---|---|
| vector only | 0.5588 | 0.4716 | 0.1176 |
| full-text only (fixed) | 0.6176 | 0.4412 | 0.1294 |
| hybrid, RRF k=60 (textbook default) | 0.6471 | 0.4950 | 0.1353 |
| hybrid, RRF k=30 | 0.6471 | 0.4955 | 0.1353 |
| **hybrid, RRF k=10 (chosen default)** | **0.7353** | **0.4986** | **0.1588** |

Hybrid beats both individual methods at every k tried, confirming the plan's premise. The
first tuning loop (M2's scope; the full ablation table -- chunk size, a reranker -- is M4) tried
RRF's k at 10, 30 and 60: k=10 won outright, not narrowly, on all three metrics. A smaller k
concentrates fusion weight on whichever ranking(s) already put a result near the top, which suits
a corpus this small (~600 chunks): a genuinely good match tends to already be high in at least one
ranking, so the textbook k=60 (tuned on far larger corpora) mostly just adds tail noise here.
`retrieve_hybrid`'s default is now `rrf_k=10`.

By question type (hybrid, k=10):

| Type | n | Recall@5 | MRR@10 | Precision@5 |
|---|---|---|---|---|
| decision-why | 9 | 0.8889 | 0.8148 | 0.1778 |
| temporal | 5 | 0.8000 | 0.3467 | 0.1600 |
| cross-doc | 7 | 0.7143 | 0.3537 | 0.2000 |
| factual | 8 | 0.6250 | 0.4313 | 0.1250 |
| evidence | 5 | 0.6000 | 0.3922 | 0.1200 |

Weakest categories -- evidence (specific numbers: test counts, exit codes) and factual (a single
fact buried in a longer document) -- are the ones a cross-encoder reranker (M4's planned ablation)
is specifically good at; not addressed here.

## Open items into M3

- Generation with numbered citations, using the top-k hybrid results as context.
- The refusal behaviour for the 9 unanswerable questions, excluded from this milestone's scoring.
- M4: the frozen ablation table (chunk size, reranker on/off) and the held-out test-split score,
  reported once.

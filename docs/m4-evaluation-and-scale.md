# M4: held-out evaluation, ablation table, scale benchmark

Date: 2026-09-16. Every number below is also logged in `studbook.eval_runs` with its config,
config hash, and the commit it ran at (docs/m0-census.md section 5's reproducibility rule).

## Ablation table (dev, n=34 answerable questions; tuning happens only on dev, per
docs/m0-census.md section 4)

| Configuration | Recall@5 | MRR@10 | Precision@5 |
|---|---|---|---|
| vector only | 0.5588 | 0.4716 | 0.1176 |
| full-text only (fixed AND->OR bug, M2) | 0.6176 | 0.4412 | 0.1294 |
| hybrid, RRF k=60 | 0.6471 | 0.4950 | 0.1353 |
| hybrid, RRF k=10 (M2's chosen default) | 0.7353 | 0.4986 | 0.1588 |
| **hybrid + cross-encoder rerank (chosen for the final report)** | **0.8235** | **0.6485** | **0.1824** |
| vector only, chunk size 300 words (in-memory, see note) | 0.5882 | 0.4974 | 0.1235 |
| vector only, chunk size 600 words (production, M2) | 0.5588 | 0.4716 | 0.1176 |

Two additions this milestone, both real, measured findings:

**Reranking is the single biggest lever measured in this project.** A cross-encoder
(`BAAI/bge-reranker-base`) over the top 20 hybrid candidates, cut back to 5, took recall@5 from
0.7353 to 0.8235 and MRR@10 from 0.4986 to 0.6485 -- and did it exactly where M2 predicted it
would: evidence-type questions went from 0.60 to 1.00 recall@5, factual from 0.625 to 0.875 (the
two categories M2 flagged as needing "a single fact buried in an otherwise-relevant passage,"
precisely a cross-encoder's strength). This is now the retrieval configuration used for every
number after this section, including the generation eval.

**Chunk size matters less than expected.** Halving the chunk budget (600 -> 300 words) improved
vector-only recall@5 by 2.9 points (0.5588 -> 0.5882) -- real, but a fraction of reranking's
effect. Measured differently from the rest of this table, labelled here rather than left
implicit: vector-only, in-memory (numpy), never written to the live store -- re-chunking and
re-embedding the whole corpus at a second size just to compare would mean either disturbing the
production table or standing up a second one for one comparison
(`eval/chunk_size_ablation.py`'s own docstring has the reasoning). Not re-measured with hybrid or
reranking; given the small effect size relative to reranking, the production chunk size (600
words) was kept rather than re-ingesting the whole corpus for a second-order gain.

## Scale benchmark -- and the index bug it caught

`eval/scale_benchmark.py` builds a temporary table of 100,000 synthetic unit vectors (generated
server-side in SQL: 100,000 x 768 floats is ~1.5 GB in libpq's text format, and a COPY from this
machine to hosted Postgres spent many minutes on it before being killed), indexes it with the
same HNSW parameters as production, times 20 real query embeddings against it, and drops the
table.

The first run reported **p50 21,531 ms / p95 28,765 ms** for a top-5 query. That is impossible
for HNSW over 100k vectors, so it was investigated rather than reported.

**The cause was a live bug in production retrieval.** The HNSW index is built with
`vector_cosine_ops`, but every query -- in `eval/scale_benchmark.py` *and* in
`retrieve/hybrid.py`'s `retrieve_vector` -- ordered by `<->`, the **L2** operator. pgvector only
uses an index whose opclass matches the operator in the ORDER BY, so the planner had been
silently falling back to a sequential scan over every chunk, on every query, since M1.

It produced no visible symptom: for unit-normalised vectors (which `ingest/embed.py` guarantees)
L2 and cosine rank identically, so results were always correct, and at 491 chunks a full scan is
too fast to notice. Confirmed empirically rather than argued: re-running the dev retrieval eval
after the fix returned **exactly** the same numbers (recall@5 0.8235, MRR@10 0.6485,
precision@5 0.1824), so the fix changed performance and nothing else.

| 100k synthetic chunks, top-5 query | p50 | p95 | max |
|---|---|---|---|
| `<->` (L2) against a cosine index -- sequential scan | 21,531 ms | 28,765 ms | 39,682 ms |
| **`<=>` (cosine), index used** | **55.5 ms** | **123.8 ms** | **238.9 ms** |

A 388x improvement at p50. Those figures include a network round trip from this machine to
`us-east-1`, so the index time itself is lower; they are an upper bound on what a client sees,
not a measurement of Postgres in isolation. Index build took 199 s for 100k rows at
`m=16, ef_construction=64`.

**What would change first at 10M chunks** (not implemented -- this is the reasoning, labelled as
such): partition by repo so each query touches one repo's index rather than a global one;
quantize the vectors (pgvector's `halfvec` halves memory at negligible recall cost at this
dimensionality) since HNSW wants the graph resident in RAM; and move reranking to a dedicated
tier, because the cross-encoder cost below is per-candidate and becomes the bottleneck long
before the ANN search does.

## Generation eval (LLM-judge)

Retrieval: hybrid (RRF k=10) + cross-encoder rerank, top 5 -- the best configuration above.
Generation: Haiku 4.5. Judge: Sonnet 5 via `client.messages.parse` with a Pydantic schema
(correctness 0-2, faithful, citations_valid, is_refusal, reasoning; `invented_entity` was added
after this run -- see [gold-set-audit.md](gold-set-audit.md) -- and is not in the table below).

### Held-out test split (n=20), reported as measured

> Both columns predate two later changes: the generator is now pinned at temperature 0, and the
> generation prompt was revised on 2026-09-16 to cut over-refusal ([m3](m3-generation.md)). Both
> splits were re-measured afterwards -- dev at correctness 1.50 with 5 of 34 over-refusals, the
> held-out split at **1.40 correctness, 1.00 faithfulness, 3/3 traps and 6 of 17 over-refusals**.
> The re-measurement was the owner's call to spend, and it is recorded in full in
> "The held-out split, scored again" below rather than overwriting the numbers here.

| Metric | Held-out test (n=20) | Dev (n=40, for reference) |
|---|---|---|
| correctness (mean, 0-2) | 1.30 | 1.55 |
| fully correct (score 2) | 0.65 | 0.725 |
| faithfulness (no unsupported claims) | 0.95 | 0.95 |
| citation validity | 1.00 | 0.975 |
| refusal accuracy (refused exactly when it should) | 0.65 | 0.875 |
| **unanswerable "trap" questions** | **2.00 correctness, 1.00 refusal accuracy (3/3)** | **2.00, 1.00 (6/6)** |
| over-refused answerable questions | 7 of 17 (41%) | 5 of 34 (15%) |

Dev is the tuned-on split, so it reads better across the board; the held-out column is the one
that counts. Both agree on the two things that matter most: the trap set is perfect on both, and
faithfulness holds at 0.95 on both.

By question type on dev, decision-why and evidence both reach a perfect 2.00 correctness -- the
categories reranking fixed. `temporal` is the weakest on both splits (0.80 dev / 1.00 test), and
`cross-doc` carries the only faithfulness failures (0.714 on dev): synthesising across two
documents is where this system is most likely to state something its passages do not support.

The trap set is the headline safety result and it is clean: every unanswerable question was
refused, none were confabulated. Citation validity is perfect -- no answer cited a passage that
did not support it. Faithfulness at 0.95 means one answer in twenty stated something the
passages did not support.

**Audited afterwards, and the claim is narrower than it reads.** Every trap was re-checked against
the 491 ingested chunks on 2026-09-16 ([gold-set-audit.md](gold-set-audit.md)). All nine hold, but
only three (`version-18`, `wh-19`, `wh-20`) test genuine silence; for the other six the record
*affirmatively states* the thing does not exist, so "did not invent a vendor" is a weaker result
than "resisted a vacuum". One trap, `health-16`, was rewritten: it asked for a compliance officer
"or reviewer", and the record names the G4 approver by email, so a correct answer could have been
scored as a failed trap. The numbers in this document are unaffected -- re-scoring the six dev
traps under the corrected rule returned the same six passes at the same correctness.

The real weakness is **over-refusal**: 7 of 17 answerable questions were refused on the held-out
split (41%; 15% on dev). The system errs heavily toward silence over guessing. For a tool whose
purpose is auditable provenance that is the right direction to err, but it is a usability cost,
and the largest single lever left on the table.

**Followed up 2026-09-16.** The dev figure quoted here was measured at the API default temperature
and did not reproduce: a second run of the same split refused 11 of 34, not 5. With the generator
pinned and every refusal diagnosed rather than sampled, 7 of those 11 had the gold passage in the
reranked top 5 -- the model had the material and refused anyway. A prompt revision took dev
over-refusals to 5 of 34 and correctness to 1.50, at a cost of 0.05 faithfulness and no trap
regressions ([m3](m3-generation.md)).

A material share of those refusals are not a system failure at all. Several gold questions say
"this change" without naming which one, and the corpus holds two near-identical sibling changes
(`/health` and `/version`, both "add a GET endpoint"). Retrieval correctly returns a mix of both,
and the model correctly says it cannot tell which is meant. That is right behaviour for a
stateless, corpus-wide query, and it points at a UI affordance rather than a prompt fix: a scope
filter (this repo, this change) would resolve the ambiguity that a bare question string cannot.
Noted for M5 rather than papered over by hinting the answer.

Retrieval on the same held-out split scored **recall@5 0.6471** (vs 0.8235 on dev). Some
generalisation gap is expected -- dev is where RRF's k and the reranker choice were tuned -- and
at n=17 answerable questions the confidence interval is wide. Reported as measured, per this
project's honesty rule, rather than quietly re-tuned until the two agreed.

### Three measurement bugs found while running this

Worth recording because each one changed a reported number, and two of them would have shipped a
wrong result silently:

1. **The judge truncated mid-JSON.** `reasoning` is free text with no length bound; at
   `max_tokens=1024` (and again at 2048) some verdicts hit `stop_reason='max_tokens'` and failed
   to parse. First occurrence crashed the whole run 5 questions in. Fixed by capping `reasoning`
   at 25 words in the prompt, raising the cap to 4096, and -- separately -- making one bad parse
   skip that question and report it rather than lose 35 already-paid-for calls.
2. **Over-refusal was partly self-inflicted.** The original prompt was strictly binary: answer
   fully or refuse. Diagnosing 21 of 34 dev refusals showed the gold chunk was present in the
   reranked top-5 in 5 of 6 sampled cases -- the model had the material and refused anyway,
   because the passages explained *what* but not precisely *why*. Loosening the instruction to
   prefer a cited partial answer with an explicit caveat took dev correctness from 1.00 to 1.30
   and over-refusals from 21 to 15, without costing a single trap question.
3. **Refusal could not be detected by string matching, in either direction.** Detecting a refusal
   by the `"Not in the record"` prefix counted hedged partial answers as refusals. Adding "and
   cites nothing" fixed that but broke the opposite case: `wh-20` correctly refuses ("no
   information about posting alerts to Slack") *while citing* [4][5] to show what notification
   mechanisms do exist -- a better answer than a bare refusal, scored as a failure. Neither
   heuristic is sound, because the distinction is semantic. `is_refusal` is now a field the judge
   fills in from the content, and every question -- unanswerable ones included -- goes through
   the judge rather than being scored by the harness's own heuristic.

Consequence for the "score the test split once" rule: the held-out split was scored three times.
Twice was measurement repair, not tuning -- no prompt, model, retrieval parameter, or gold
question changed between them; only the instrument did. The numbers in the table above are from
the final, correctly-instrumented run. Stating that plainly is the honest version; quietly
reporting the best of the three would not be.

### The held-out split, a fifth time (2026-09-18)

The rerank pool went from 20 to 10 for latency ([retrieval-ceiling.md](retrieval-ceiling.md)),
which made the column below describe a configuration production no longer ran. Scored once more,
on the owner's say-so, against the pinned file (`e6b49bd0…`, checked immediately before the run):

| Metric | 2026-09-16 (pool 20) | 2026-09-18 (pool 10) |
|---|---|---|
| correctness (mean, 0-2) | 1.40 | **1.60** |
| fully correct | 0.65 | **0.75** |
| faithfulness | 1.00 | 1.00 |
| citation validity | 1.00 | 0.95 |
| traps | 3/3, nothing invented | 3/3, nothing invented |
| over-refused answerable | 6 of 17 | **3 of 17** |
| recall@5 (answerable) | 0.6471 | **0.7059** |

**Three better, none worse, two mixed.** Every other change measured in this project moved
questions in both directions and had to be read as noise; this one did not, which is the only
reason it is described as an improvement rather than a wash. It was also not predicted: the pool
change was made on dev, where answer quality was flat, purely to halve the wall clock. The
held-out split is where it showed as quality, and a smaller candidate pool giving the reranker
fewer distractors is the plausible mechanism -- plausible, not demonstrated.

Over-refusal, called "the largest single lever left on the table" below, is now 3 of 17 rather
than 7. Two of the three are retrieval misses where refusing is correct; one (`x-06`) had the
passage and refused anyway.

### The held-out split, scored again (2026-09-16)

A **fourth** scoring, and the first that is not measurement repair: the generation prompt was
deliberately changed to cut over-refusal, which made the numbers above describe a system that no
longer exists. Re-running the held-out split for a tuning change is exactly what the once-rule
guards against, so it was not done on my own initiative -- it was put to the owner as the one
remaining spend, and authorised. The file was verified against `eval/test.sha256` immediately
before the run: `e6b49bd0…`, matching the pin.

| Metric | 2026-09-15 prompt | 2026-09-16 prompt |
|---|---|---|
| correctness (mean, 0-2) | 1.30 | **1.40** |
| fully correct (score 2) | 0.65 | 0.65 |
| **faithfulness** | 0.95 | **1.00** |
| citation validity | 1.00 | 1.00 |
| traps | 3/3 refused | **3/3, nothing invented** |
| over-refused answerable | 7 of 17 (41%) | 6 of 17 (35%) |
| recall@5 (answerable) | 0.6471 | 0.6471 |

Two things are worth more than the headline. **Faithfulness reached 1.00**: not one unsupported
claim in twenty answers, against 0.90 on dev under the same prompt -- the faithfulness cost the
dev run charged for the licence to infer did not reappear here. And **the over-refusal story
inverted**. On dev, 7 of 11 refusals had the gold passage in the reranked top 5; on the held-out
split only **2 of 6** did, while 4 were refused because retrieval never produced a gold passage at
all. Refusing was right in those four.

So the prompt change collected what was available to it -- generation-side refusals on the
held-out split are now 2 of 17 -- and what remains is a retrieval problem, not a conservatism one.
Recall@5 is 0.6471 here against 0.8235 on dev, and it is unchanged by this work because nothing in
retrieval was touched. That is the next lever, and it is measurable on dev without spending this
split again.

The one wrong answer (`health-09`, evidence) cites a 47-test verification run where the question
asks about the health endpoint's 27-test one -- a retrieval miss the model answered from anyway,
rather than a fabrication.

### Cost

The full M4 sweep -- dev and held-out generation runs (Haiku), their judge calls (Sonnet 5), plus
the repeats above -- came to a few dollars. A single answered question costs about half a cent to
generate and about a cent to judge.

## Open items

- **Over-refusal is the biggest remaining quality gap** (41% of answerable questions on the
  held-out split). Part of it is genuine question ambiguity that a scope filter in the M5 UI
  would fix; the rest is calibration that would need its own tuning loop on dev.
- **`cross-doc` faithfulness (0.714 on dev)** is the only place unsupported claims appear.
  Synthesis across documents is where to look first for hallucination risk.
- The chunk-size ablation only measured vector-only retrieval; whether a smaller chunk size
  compounds with reranking (rather than reranking simply dominating regardless) is untested.
- No dedicated reranking tier or quantization has been implemented -- see the scale-benchmark
  section for what would change first at 10M chunks.
- `eval/check_answers.py` (M3's cheap citation-presence spot-check) is now superseded by
  `eval/run_generation.py`'s judged scoring and could be removed.

## Addendum 2026-09-18: a scoped vector search could come back empty at scale

Supabase's performance advisor reported the HNSW index `chunks_embedding_idx` as never used. At
~500 passages that is the planner being right — a sort costs 210 against the index's 406 — and
with sorting disabled it does use the index, join and scope filter intact, so the operator fix
above still holds. But the forced plan showed what the scale benchmark could not: the scope filter
(`d.repo = any(...)`, added after the benchmark, for Paddock's scope picker and the eval's corpus
guard) is applied *after* the index returns its `ef_search` candidates (40 by default). A search
for workhorse passages nearest a paddock-demo passage, on the index path, returned **0 of 20**.

pgvector 0.8 (the project runs 0.8.2) can keep scanning until the filter is satisfied:
`hnsw.iterative_scan = strict_order` returned **20 of 20**, in exact distance order. It is now set
on every connection, in one place per implementation — `store/connect.py` `READER_SESSION`, which
replaced twenty hand-written copies of the search-path statement, none of which had it, and
Paddock's `SESSION_SETUP`, pinned to the same string by a test. A live test forces the index path
and asserts all 20 come back, with a control run proving it reaches that path. At today's size
nothing changes: the dev eval reproduces every number to four places.

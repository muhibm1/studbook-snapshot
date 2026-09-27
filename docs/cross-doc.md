# Cross-doc questions: a coverage problem that recall could not see

Date: 2026-09-18. Cross-doc has been the weakest category since M4. This is why, measured, and
what was tried.

## Recall overstates it

Recall@5 asks whether *any* gold passage reached the top 5. A cross-doc question needs a passage
from *each* of its documents — "how did /version's test suite compare to /health's?" cannot be
answered from one side. So `eval/metrics.py` gained **coverage@5**: every gold document has at
least one of its quotes in the top 5. For a single-document question it equals recall; it only
differs where a question spans documents.

Dev split, production configuration (hybrid, RRF k=10, 10 rerank candidates, top 5):

| | recall@5 | coverage@5 |
|---|---|---|
| cross-doc (n=7) | 0.7143 | **0.4286** |
| all answerable (n=34) | 0.8529 | 0.7941 |

Recall reproduces the published 0.8529 exactly; every other category's coverage equals its recall,
as it must. Cross-doc drops from five questions "found" to three actually covered.

It matters because it is what the generator's behaviour tracks. In the saved dev run, **every
cross-doc question that received all its documents scored 2, and every one that received only one
of two was refused and scored 0** — the model, correctly by its own rules, would not answer a
comparison from one side, and cited the side it had while refusing. Recall called those hits.

## Where the missing halves go

`eval/coverage.py` follows each gold document of each multi-document question through the pipeline
(free, local):

| where a gold document was lost | documents |
|---|---|
| covered (in the top 5) | 8 of 14 |
| reranked to just below the cut (#6, #7) | 2 |
| fused too low for the rerank pool (#13, #27) | 2 |
| never retrieved by either arm | 2 (both of `version-14`) |

An even split, and every cause has the same root: **one ranking against the whole question**. It
puts the side the question's wording leans toward on top and the other side below it, whether that
is at #6, at #27, or nowhere.

The blunt fix was already on file. Giving the generator 10 passages instead of 5 (a dev run saved
from the retrieval-ceiling work) rescued one of the four failing cross-doc questions (`wh-13`,
whose missing document sat at #7) and none of the others — it reaches documents lost just below the
cut and nothing lost further up the pipeline, and it moved overall answer quality by zero
(docs/retrieval-ceiling.md).

## Tried: decomposing the question — rejected

The hypothesis: split a question into one standalone sub-question per part (Haiku, temperature 0,
the follow-up rewrite's guard so a part may only use the question's own words), rank each part
against itself, and fill the top 5 round-robin so every part is guaranteed a place
(`eval/decomposition.py`, `eval/decompose.py`, 34 Haiku calls).

| coverage@5 | one ranking (production) | parts, round-robin | whole question first, then parts |
|---|---|---|---|
| all answerable (n=34) | **27/34** | 23/34 | 25/34 |
| cross-doc (n=7) | **3/7** | 2/7 | 3/7 |

It made things worse, and reading every split says why:

- **It over-splits.** 13 of 34 questions were split, most of them single-document: "what happened
  after X, and after that?" became three parts, one of them a meaningless "What happened after
  that?". A single-document question split into parts can only move its own passages around, and
  here it lost four of them.
- **The guard blocked the splits that mattered.** `version-12`'s two sensible parts were rejected
  for one paraphrased word ("issue"); `wh-13`'s for similar reasons. Loosening the guard is the
  same trade the rewrite section of docs/follow-ups.md declined.
- **Where it split correctly, it still lost.** `version-14` split into exactly the right two parts
  and neither part reached its documents: the verification files never say "size of the test
  suite", they say "12/12 pass". `x-05` split correctly and lost coverage, because the whole
  question's wording disambiguated better than either half.

Tuning the prompt against seven cross-doc questions would be fitting dev, not fixing retrieval.
The code is kept, marked as rejected, so the result can be reproduced.

## Tried: capping chunks per document — rejected

If the top 5 were crowded with several chunks of one document, capping each document would lift
the two documents lost at #6 and #7 into view (`eval/diversity.py`, free). It is not crowded — about
one duplicate chunk per top 5 on dev. A cap of 2 changed nothing on any category; a cap of 1
lowered cross-doc coverage from 3/7 to 1/7, because a document's second chunk is often where the
detail is.

## Tried: synonym-aware lexical search — rejected

Postgres full text matches stems ("tests" finds "test") but not synonyms, and a thesaurus
dictionary is a file on the database server, which hosted Supabase does not expose. So the
synonyms went on the query side, as a third full-text ranking fused with the two existing ones —
never touching the original arms or the question the generator answers (`eval/expansion.py`,
`eval/expand.py`). Two sources, with settings fixed before the first run: each query word's 3
nearest corpus words by the bge embedding (at cosine 0.70 and 0.80), and up to twelve words from
Haiku.

| coverage@5 | baseline | corpus words ≥0.70 | corpus words ≥0.80 | Haiku |
|---|---|---|---|---|
| all answerable (n=34) | **27/34** | 23/34 | 24/34 | 21/34 |
| cross-doc (n=7) | **3/7** | 3/7 | 3/7 | 2/7 |

**No question gained coverage in any arm; every arm lost some.** Reciprocal rank fusion gives the
third ranking an equal vote, not a weak one, so an expansion's noise ("severity priority category"
for a question about a risk tier) displaced good passages on factual and evidence questions. And
neither source produced a word that would have helped: for `version-14` the corpus neighbours of
"size" were "length large larger", and Haiku proposed "coverage metrics delta".

The oracle check settles what kind of gap it is. Full text, handed words directly:

| query | /version gold | /health gold |
|---|---|---|
| the question as asked | not in the top 50 | not in the top 50 |
| "count of tests pass for /version and /health" (synonyms of "size") | #28 | #19 |
| "verification npm test pass fail" | **#5** | **#1** |

The documents are reachable by full text — but by "verification … pass fail", none of which is a
synonym of anything in the question. Getting there takes knowing that this record states a test
suite's size in verification reports, as TAP counts. That is an inference about the record's
conventions, not synonymy, so synonym expansion cannot close it by construction; even the true
synonyms leave both documents far below the rerank pool.

## Tried: hypothetical passage search (HyDE) — rejected

The oracle check says `version-14` needs the record's own register, not synonyms. So: ask Haiku to
write the short passage that would answer the question *as it would appear in this record*, and
search with that too (`eval/hypothetical.py`, `eval/hyde.py`, 34 calls). The passage is only used
to retrieve — never shown, never given to the generator — so what it invents cannot reach an
answer. The prompt names the record's document types in general terms and nothing about test
counts, and a test fails if it ever mentions "tap", "pass", "fail", "count" or "tests"; one word
("counts") was removed from the first draft for that reason, since it was written with `version-14`
in mind. The description of the record's document types ("verification reports with command
output and logs", among eight) was also written after the diagnosis, and is disclosed as such.
Arms fixed before the run:

| coverage@5 | baseline | + passage vector | + passage vector and full text | passage replaces question vector |
|---|---|---|---|---|
| all answerable (n=34) | **27/34** | 24/34 | 25/34 | **27/34** |
| cross-doc (n=7) | **3/7** | 2/7 | 2/7 | 3/7 |

Adding the passage as another ranking loses, for the same reason as synonym expansion: an equal
vote in fusion. Replacing the question's embedding with the passage's — the original HyDE
formulation — nets to zero by trading: it gained `version-12` (cross-doc) and `wh-15` (temporal)
and lost `version-15` (cross-doc) and `wh-07` (factual). Two up, two down on 34 questions is the
churn this project reads as noise (docs/gold-set-audit.md), and it would cost a model call and
about a second on every question.

`version-14`'s passage shows why the premise fails here. Asked to write in the record's style,
Haiku wrote a *pytest* listing — `tests/endpoints/test_version.py: 47 cases` — its own idea of how
a project records a suite, not this record's `npm test` and `# pass 47`. (47 matches the real count
by coincidence.) A model that has not seen the record's conventions cannot write in them, and
that is exactly what HyDE assumes it can.

## Where that leaves cross-doc

Measured now rather than guessed: the gap is coverage, not recall; it has three causes in equal
parts; and none of decomposition, diversity, synonym expansion or hypothetical-passage search reaches
them on this corpus. The one mechanism with
any positive evidence is the blunt one — more passages rescued the document at #7 — and it already
failed to move overall answer quality. What remains for `version-14` is not vocabulary but
convention: its answer lives where this record keeps test counts, and the one retrieval idea aimed
squarely at that — a hypothetical passage in the record's style — failed because a general model
does not know this record's style. The remaining lever is the record itself: verification reports
that say what they measured ("test suite: 47 tests, 47 pass") in words a question would use. That
is a change to how WorkHorse writes its verification reports, not to Studbook, and it would help
every retriever, not only this one.

coverage@5 stays in the retrieval report, so the next idea for this category has a number to beat
that tells the truth about it.

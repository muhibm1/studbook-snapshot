# Does carrying the previous question help a follow-up retrieve?

Date: 2026-09-18. Paddock's Studbook tab keeps a thread, and a follow-up first retrieved on the
last two questions plus the current one. That heuristic shipped with its own docstring admitting it
was unmeasured, because the gold set is single-turn. This file is the measurement it owed, the
measurement of its downside, and the query rewrite that replaced it (last section).

## The set

`eval/threads.jsonl`: twelve pairs, each an existing gold question plus a follow-up written to be
answered by **the same passage**. So the gold quote is the first question's, already verified
verbatim by `eval/gold.py`, and the only variable is whether retrieval can still find that passage
once the question stops naming its subject — "What broke because of it, before the fix?",
"What does it do instead?", "Which parts of it did they keep?".

Three queries per pair (`eval/threads.py`, free — no API calls):

| query | what it represents |
|---|---|
| **bare** | the follow-up alone, which is what a tab without a thread sends |
| **carried** | the first question + the follow-up, which is what the tab sends |
| **first** | the original question, as an upper bound: the passage was written for it |

## Result

| query | recall@5 |
|---|---|
| bare | **2/12 — 0.1667** |
| carried | **8/12 — 0.6667** |
| first | 11/12 — 0.9167 |

Carrying the question rescued seven pairs and broke one. A stateless follow-up finds the right
passage about one time in six; carrying the subject forward takes that to two in three, which is
roughly two thirds of the way to what the fully-specified question manages. The heuristic does the
thing it was written to do, and the size of the gap it closes is worth having measured rather than
assumed — 0.1667 is a much worse baseline than it seemed reasonable to guess.

**Four pairs still miss with the question carried** (`t-04`, `t-06`, `t-07`, `t-08`), and they
share a shape: the follow-up is short and almost contentless — "So what do they call instead?",
"What else shipped in that one?" — so the concatenation is mostly the previous question, and the
embedding lands back where the previous turn already looked rather than on what is being asked
now. That is the mechanism a query rewrite would fix and a concatenation cannot.

**`t-08` is the one carrying broke, and it is a weak data point**: its upper bound misses too
(`first=MISS`), so the passage is not reliably retrievable even from the question it was written
for, and the pair is measuring retrieval's own limit rather than the heuristic's.

## What this does not measure

- **Answer quality.** This section is retrieval only; answers are measured in "Answers, not just
  retrieval" below.
- **Whether the pairs are fair.** Machine-drafted at first, then **reviewed on 2026-09-18 by
  reading every gold passage in full**. Ten were fair. `t-07` was not: it asked what the reviewers
  recommended, but its gold passage is the findings table, which records the finding as open — the
  recommendation ("fix before merge", decision D1) is in a different section of the packet — so
  its hits were finding a passage that did not answer it. It was reworded to what that passage does
  answer ("Was it ever fixed?"). `t-08` is kept but marked weak: it restates its own first question,
  and even that question misses the passage. The numbers in this file up to the review are from
  the draft set; the re-run after it is in the rewrite section.
- **Unrelated follow-ups.** Not measured here, because every pair was written to be a genuine
  follow-up. Measured below.

## The other side: a change of subject

`eval/unrelated.py` measures the failure mode the section above could not reach, and it needs no
new labels. Two dev-split gold questions about different parts of the record: the first plays the
previous turn, the second is what the user typed next. Both quotes are already verified by
`eval/gold.py`, and the pairing is mechanical rather than chosen — sorted by id, each question's
"previous turn" is the next one round-robin whose gold documents are disjoint from its own — so
every one of the 34 answerable dev questions is measured and nothing is hand-picked.

| query | recall@5 |
|---|---|
| the question alone (what "New thread" sends) | **29/34 — 0.8529** |
| previous + question (what the tab sends) | **20/34 — 0.5882** |

**Carrying an unrelated question broke 10 of the 29 questions that work on their own**, a cost of
26 points. That is about half the size of what it buys on a genuine follow-up (+50 points), so
the heuristic is a bet that the next question follows on, and it pays roughly two to one. When it
loses, it loses badly: the user typed a perfectly good question and got nothing, and nothing in the
tab says the previous turn is why.

## Retrieving with both queries

`eval/dual_query.py` tests the obvious hedge: embed both queries, run both arms of each, and put
all four rankings into the same reciprocal rank fusion the retriever already uses. Nothing
classifies the question. The reranker still sees ten candidates, so the dominant cost of a query is
unchanged; the addition is one embedding and two more index lookups.

| | bare | carried (today) | both, fused |
|---|---|---|---|
| threads — genuine follow-ups, n=12 | 2/12 | **8/12** | 7/12 |
| unrelated — a change of subject, n=34 | **29/34** | 20/34 | 25/34 |

It is a compromise, not a win on both sides: it keeps most of the follow-up gain (7 of 8) and
recovers about half the change-of-subject loss (25 against 20, where the ceiling is 29). Pooled,
it gets 32/46 against today's 28/46.

**Not shipped, deliberately** — and superseded by the rewrite below. Whether it is better depends on a number nobody has: how often a
turn with history is a genuine follow-up. Per question it gives up 0.083 on follow-ups and gains
0.147 on changes of subject, so it wins unless more than about 64% of history-bearing turns are
real follow-ups — which is plausible for a thread UI and not measurable from here. And the one
follow-up it loses is a single question in a twelve-pair set that is still machine-drafted and
unreviewed, which is exactly the size of change this project has learned to read as noise.

What both measurements agree on is where the real fix lies: both failures are retrieval being
handed the wrong words, and a concatenation cannot know which words matter.

## Rewriting the follow-up — shipped

`retrieve/rewrite.py` (and Paddock's `rewrite.ts`) asks Haiku 4.5, at temperature 0, to turn the
new question into a standalone one using the earlier *questions'* own words, or to return it
unchanged if it already names its subject. A rewrite is a new place for a subject to be invented,
so it is **used only if every content word in it already appears in the thread's questions**
(`unsupported_words`, a crude-stem comparison with a fixed list of function words); otherwise, or
if the call fails, retrieval falls back to the concatenation. The prompt and the guard are pinned
by sha256 on both sides, so the two copies cannot drift apart silently.

`eval/rewrite.py`, the same two sets, 46 Haiku calls:

| | bare | carried | **rewrite** | rewrite, reranked vs. the question as typed |
|---|---|---|---|---|
| genuine follow-ups (n=12) | 2/12 | 8/12 | **11/12** | 10/12 |
| a change of subject (n=34) | 29/34 | 20/34 | **30/34** | 30/34 |

It wins both ways, which is what neither the concatenation nor the fusion could do. 11/12 is the
same as the fully-specified original questions manage, so on that set it is at the ceiling; 30/34
beats the question alone, because three changes of subject that leaned on "this change" were given
a subject (`health-07`, `version-13`, `wh-15`). Reranking against the rewrite rather than the
follow-up as typed is better by one (`t-06`), so both retrieval and rerank use the rewrite.

Every rewrite was read, not only counted. The rewriter left 30 of the 34 changes of subject exactly
as typed and rewrote 11 of 12 follow-ups. **The guard fired twice, and both were false positives**:
a rewrite that said "stopping" where the question said "stop" (the stem does not undouble
consonants), and one that said "the team" for "they". Both fell back to the concatenation and both
still found their passage. The guard was left as measured rather than loosened after the fact.

**What the guard cannot catch**, seen in the first live thread in Paddock: after "Why did control
state move into the git common dir?", the follow-up "What broke because of it, before the fix?"
was rewritten as "What broke because of control state moving into the git common dir" — every word
from the thread, and the wrong referent, since the move *was* the fix. Retrieval still found the
right passage and the answer's cited detail was correct, but its first sentence carried the same
misreading. The generator never sees the rewrite, so the misreading is the question's own
ambiguity rather than the rewrite's doing — but a rewrite can resolve a pronoun wrongly using only
words the user typed, and no lexical check will notice. That is why Paddock shows the rewrite on
the answer ("Searched for …"): it is part of the receipt.

**Re-run after the pair review** (t-07 reworded, below): rewrite 11/12 and 30/34, unchanged;
carried 9/12, up one because the reworded t-07 is easier. Two things the re-run showed that one
run could not. The rewrite is **not fully deterministic at temperature 0**: t-11 came back worded
differently from the same prompt, so a rewrite should be read as a sample, not a function of the
thread. And that rewrite tripped the guard on "pinning" against "pin" — the same doubled-consonant
miss as "stopping" against "stop", so two of the four guard trips across both runs are one flaw
in the stemmer, each costing a fallback to the concatenation rather than a wrong answer.

**Fixed**: the stemmer now undoubles a final consonant ("pinning" → "pin", "stopping" → "stop"), on
both sides, and the guard's pin now hashes the stems of a fixed probe list as well as the word
lists, so a change to the stemmer's *logic* on one side fails a test too. Re-measured: rewrite
11/12 and 30/34, unchanged, and **all twelve follow-ups now search with their rewrite** (the guard
fired on none of them). The one remaining trip is a change of subject rewritten with "the team" for
"they"; "team" is a noun that could be a subject, and a guard against invented subjects should be
strict about nouns, so it stays.

Cost: one Haiku call per follow-up (a fraction of a cent, well under a second), none for a fresh
question.

## Answers, not just retrieval

Everything above measures whether the passage is *retrieved*. `eval/thread_answers.py` scores the
answer the reader actually sees, with the same judge as `eval/run_generation.py`, against a
reference answer written for each follow-up by reading its gold passage in full
(`reference_answer` in `eval/threads.jsonl`). The judge is shown the thread's first question with
the follow-up, since "was it ever fixed?" is ungradable alone. Two arms, each what a real tab does:

| | stateless (no thread) | thread (Paddock 0.4.4) |
|---|---|---|
| correctness (0–2, mean) | 0.17 | **1.67** |
| fully correct | 1/12 | **9/12** |
| faithful | 10/12 | **12/12** |
| refused | 11/12 | 1/12 |
| invented an entity | 0/12 | 0/12 |
| gold passage retrieved | 2/12 | 11/12 |

The thread is the difference between a follow-up being answered and being refused. Neither arm
invented anything, so the stateless failure is the safe one — it refuses — but two of its refusals
were judged unfaithful, and why is worth knowing: with no subject, the model guessed one ("what do
the new test cases call instead", "instead of merging the branch") and stated facts about the guess.
An unanchored follow-up does not only fail to find the answer; it answers a different question.

The thread arm's three misses, read one by one:

- **t-03 (scored 1)**: asked for "the alternative they turned down", singular, and got one of the
  three. A defensible reading of the question.
- **t-08 (refused)**: the weak pair from the review. Its rewrite was refused by the guard, the
  concatenation did not retrieve the gold passage, and the model refused although a retrieved
  passage carried part of the rationale.
- **t-11 (scored 1)**: the self-contradicting refusal the system prompt forbids — it opens "Not in
  the record", then states the defect with a citation. The prompt's rule against this reduced it
  (docs/m3-generation.md) but has not eliminated it, and it is a single-turn failure too, not one
  the thread introduces. Since 0.4.4 Paddock at least shows it with its citation rather than
  badging it as a refusal and dropping the evidence. The same caveat as the first measurement applies to the twelve follow-up pairs, which
are still machine-drafted and unreviewed; the change-of-subject set is not, and it carries the
larger result.

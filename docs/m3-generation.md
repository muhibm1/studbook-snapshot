# M3: generation with citations and refusal

Date: 2026-09-16. Haiku 4.5 (`claude-haiku-4-5`, the plan's default), one call per question, no
tools -- a single LLM call is the right tier for grounded Q&A over a handful of already-retrieved
passages (Claude API skill's own "which surface" guidance).

## What shipped

- **`answer/prompts.py`**: numbers the retrieved passages `[1]`, `[2]`, ... with their repo,
  doc type and heading path, then the question. The system prompt requires a citation per claim
  and a fixed refusal phrase, `"Not in the record: "`, chosen to match `eval/gold.py`'s own
  unanswerable-question convention -- refusal detection in `generate.py` is a literal string
  prefix check, not a second model call to classify it.
- **`answer/generate.py`**: `generate_answer(question, chunks) -> Answer` -- text, parsed
  citations (chunk id, heading path, repo, doc type per citation, for a UI's receipts), whether
  it refused, model, token usage. An empty chunk list refuses locally without spending a call.
  `max_tokens=1024`: short, cited answers don't need more, and it caps the worst case per call
  (the cost ceiling promised earlier in this project).
- **`cli.py`**: `python cli.py "question"` -- retrieve, then generate, then print the answer with
  its receipts and token usage. The end-to-end path a UI (M5) will wrap.
- **`eval/check_answers.py`**: a cheap scripted spot-check across the whole gold set (not the
  scored LLM-judge eval -- that's M4) -- does every unanswerable question get refused, does every
  answerable one come back with a citation at all.

## Results

**Refusal on genuinely unanswerable questions: 6/6 (100%)** on the dev split's unanswerable
questions, confirmed live. This is the headline number for M3's "trap-set behaviour" deliverable.

**Citation rate on answerable questions was noisier than expected**: one `check_answers.py` run
showed 11/34; live, targeted re-runs of several of the "uncited" cases showed the model's
sampling is not perfectly deterministic (no `temperature` is set) -- the same question sometimes
answers with citations and sometimes refuses on repeated calls. Investigated five specific cases
rather than accepting the aggregate number at face value:

| Case | Gold chunk in top-5? | What happened |
|---|---|---|
| health-01 | yes | Answered with two correct, well-placed citations |
| health-02 | yes | Answered with citations |
| wh-02 | yes | Refused: the retrieved passages explain *what* changed but not *why* -- a genuine, arguably correct epistemic call, not a retrieval failure |
| wh-07 | yes | Refused for the same reason: found a passage that touches the topic, judged it insufficient to answer the specific question asked |
| version-07 | **no** | Retrieval genuinely missed the gold chunk -- an honest, correct refusal |

Reading: most of the gap between M2's retrieval recall@5 (0.7353) and this milestone's cruder
citation-presence check is the model being conservative on decision-why-shaped questions where a
retrieved passage is topically related but doesn't fully answer *why*, not a retrieval failure --
consistent with M2's own finding that decision-why and cross-doc questions have lower precision
despite reasonable recall. Whether that conservatism is well-calibrated (correctly refusing what
it can't fully support) or over-cautious (refusing what a human would consider answerable from the
same passage) is exactly what M4's LLM-judge correctness/faithfulness scoring is for -- a binary
citation-presence heuristic can't tell the two apart, which is why this milestone stops at
"the mechanism works, confirmed on real examples" rather than reporting a generation accuracy
number as if it were final.

## Sampling: pinned at temperature 0 (added 2026-09-16)

Generation originally ran at the API default temperature, which is 1. For a tool whose product is
a citation, that is wrong on its own terms: the same question against the same passages could come
back with different receipts. It also made the instrument noisy -- two runs of the same dev split
disagreed by 0.175 correctness and five over-refusals, and none of that was signal.

Three things about pinning it are worth recording, because none of them were what was expected:

- `anthropic` 1.6.0 **removed `temperature` from `Messages.create()`'s typed signature** -- passing
  it raises `TypeError`. The API still takes it, so it goes through `extra_body`.
- It is **honoured, verified rather than assumed**: three samples of a deliberately high-entropy
  prompt came back identical at 0 and all different at the default.
- It is **model-dependent**. Haiku 4.5 (the generator) accepts it; Sonnet 5 (the judge) answers
  `400 invalid_request_error: "temperature" is deprecated for this model`. So answers are
  reproducible and verdicts are not, and `eval/judge.py` says so rather than pretending otherwise.
  `generate_answer` retries once without the parameter if a model rejects it, and prints that its
  answers are no longer reproducible.

A unit test now binds the request `generate_answer` builds against the installed SDK's real
signature. The `TypeError` above passed every existing test, because they all use a fake client
that accepts any keyword, and only surfaced 40 questions into a paid eval run.

## Prompt revision: over-refusal (2026-09-16)

M4 named over-refusal the largest lever left on the table but diagnosed it from a sample of six.
`eval/run_generation.py --dump` now records every question's answer, judge verdict and where a
gold passage landed in the reranked top 5, and `eval/diagnose.py` splits refusals into the two
stories those records tell apart:

- **retrieval** -- no gold passage was retrieved, so refusing was the right call;
- **generation** -- a gold passage was right there and the model refused anyway.

On the pinned dev baseline, 11 of 34 answerable questions were refused and **7 of those had the
gold passage in front of them**. Reading all seven gave three distinct causes, and the prompt now
addresses each:

| Cause | Questions | Change |
|---|---|---|
| The question says "this change" and two sibling changes are in the passages. The model states both answers, then calls the question unanswerable -- which the old prompt explicitly instructed. | health-05, health-07 | Answer for each change, labelled, and ask which was meant. |
| The answer needs one inferential step the passages jointly entail (the suite reported 27/27, so the flagged flakiness never materialised). | health-13, wh-15 | Permit a conclusion the passages entail, cited, and marked as what the record implies rather than states. |
| The model quotes the rationale from a passage and then says the rationale is absent. | wh-06 | Stating a fact from the passages *is* answering: drop the refusal. |

A fourth clause followed from the trap audit ([gold-set-audit.md](gold-set-audit.md)): a passage
that positively records that something does not exist is an **answer**, cited as normal, not a
refusal. The old prompt forbade citing inside a refusal while the best response to such a passage
is exactly a cited negative, and the model split the difference -- refusal phrasing with citations
attached, one of them misattributed (`health-16`).

### Measured on dev (n=40), generator pinned at temperature 0

| | Before | After |
|---|---|---|
| Correctness (mean, 0-2) | 1.30 | 1.50 |
| Fully correct | 0.60 | 0.70 |
| **Over-refused answerable questions** | **11 of 34 (32%)** | **5 of 34 (15%)** |
| -- of those, with the gold passage retrieved | 7 | 3 |
| Faithfulness | 0.95 | 0.90 |
| Citation validity | 0.95 | 0.925 |
| Traps | 6/6, nothing invented | 6/6, nothing invented |

Two variants of the revision were measured. `eval/compare.py` diffs two runs question by question,
which is what tells a real effect from noise -- a real effect moves questions in one direction:

| | better | worse | mixed |
|---|---|---|---|
| baseline -> revision | 8 | 3 | 1 |
| revision variant A -> variant B | 2 | 4 | 1 |

They differ by 0.025 correctness and one faithfulness point -- and seven of forty questions flip
between them, which is the clearest evidence in this document that differences that size are
noise, not signal. **What replicates is the effect that
matters: both variants halve over-refusal, 11 to 5, without costing a single trap.** The variant
kept is the one whose rules do not contradict each other, not the one that scored 0.025 higher.

The cost is real and is not papered over: faithfulness fell 0.95 to 0.90, two answers in forty.
Letting the model take an inferential step is what buys the refusals back, and the same licence is
what lets it overreach -- `wh-11` states a correct figure and then conflates two findings around
it. Erring toward silence was safer; it was also, on this corpus, wrong about a third of the time.

### Confirmed on the held-out split

The owner authorised one fresh scoring of the test split, since the change made the published
held-out numbers describe a system that no longer existed. Correctness 1.30 -> 1.40, faithfulness
0.95 -> **1.00**, traps 3/3 with nothing invented, over-refusals 7 of 17 -> 6 of 17. Full account
in [m4](m4-evaluation-and-scale.md#the-held-out-split-scored-again-2026-09-16).

The held-out split also corrects the story this section tells. On dev, 7 of 11 refusals had the
gold passage in front of them; on the held-out split only **2 of 6** did, and four were refused
because retrieval never produced a gold passage at all -- where refusing is right. The prompt
change took what was reachable; the rest of the over-refusal is a retrieval problem
(recall@5 0.6471 held-out against 0.8235 on dev), and that is where the next lever is.

## A real bug found and fixed

`version-07`'s response opened with the refusal phrase, exactly as instructed, then continued in
a second paragraph with a partial answer citing `[5]` anyway -- violating the system prompt's own
"never cite a passage in a refusal" rule. `generate_answer` now truncates a refused response to
its first paragraph (`clean_refusal` in `answer/generate.py`, unit-tested against this exact
shape) rather than trusting the model to stop cleanly on its own.

## Cost

Two hand-checked calls (one answerable, one trap) cost about half a cent combined
(2261+160 and 1194+27 tokens at Haiku's $1/$5 per MTok). The full `check_answers.py` sweep over
the 40-question dev split (retrieval + generation per question) cost well under 15 cents.

## Open items into M4

- The scored generation eval: correctness vs. reference (LLM-judge), faithfulness (every claim
  traceable to a cited chunk), citation validity (cited chunks actually contain the supporting
  text), refusal accuracy on the full trap set -- each against a numeric target, on the held-out
  test split, reported once.
- Whether the model's conservatism on decision-why/cross-doc questions is well-calibrated or
  over-cautious is an open question this milestone deliberately leaves to M4's judge, not this
  one's crude citation-presence check.
- A chunk-size and reranker ablation (M4's planned ablation table).

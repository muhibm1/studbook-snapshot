# Where retrieval loses, and why showing the model more did not help

Date: 2026-09-16. The held-out run left over-refusal looking like a retrieval problem: four of the
six answerable questions the system refused were refused because no gold passage ever reached it
([m4](m4-evaluation-and-scale.md#the-held-out-split-scored-again-2026-09-16)). This is the
measurement that decides what to fix, and the change it suggested, which turned out not to work.

Everything here runs on dev, on local models and read-only queries. No API calls except the final
generation run.

## The ceiling (`eval/ceiling.py`)

Production fuses 20 candidates from each arm and reranks them down to 5, so a gold chunk outside
that pool can never be recovered, whatever the reranker does. Measured over the 34 answerable dev
questions:

| stage | @1 | @3 | @5 | @8 | @10 | @20 |
|---|---|---|---|---|---|---|
| vector | | | 0.5588 | | 0.6765 | 0.8235 |
| full-text | | | 0.6176 | | 0.7941 | 0.8235 |
| **fusion** | | | 0.7353 | | 0.8824 | **0.9118** |
| fusion, then reranked | 0.5000 | 0.7647 | **0.8235** | 0.8824 | **0.9118** | 0.9118 |

Three readings:

- **Almost nothing is unreachable.** 0.9118 of questions have a gold chunk inside the pool of 20,
  and only two (`version-06`, `version-14`) have none. Those two are genuine embedding-and-lexical
  failures; no reranker and no larger pool reaches them.
- **The loss splits evenly.** Upstream loses 0.0882 (the chunk never enters the pool); the cut to
  five loses another 0.0882 (it enters the pool and does not survive).
- **The reranked gold chunk is never below rank 10.** Recall at 10, 15 and 20 are identical, so the
  reranker is not scattering the answer deep into the list -- it is placing it just outside the
  cut.

A larger *candidate pool* does not help either: pools of 20, 50 and 100 all produce the same top
five. Feeding `bge-reranker-base` more to choose from only gives it more plausible distractors.

### One trap in this measurement, worth recording

The first run of this tool had each arm return 100 candidates rather than the production 20, and
reported a different fusion recall@10 (0.8529 against 0.8824). Rank fusion sees *every* candidate,
so deepening the arms reorders the fused list all the way to the top -- an arm depth is a different
configuration, not a deeper look at the same one. The production-faithful run reproduces M4's
published 0.8235 for hybrid+rerank exactly, which is what says it is measuring the real pipeline.
`--arm-k` now defaults to the production value and refuses to report depths beyond it.

## The change it suggested: give the generator 10 passages, not 5

Recall says this recovers the entire reranker loss, 0.8235 -> 0.9118, for nothing but input
tokens. It does not.

| dev, n=40 | top 5 | top 10 |
|---|---|---|
| correctness (mean) | 1.500 | 1.500 |
| faithfulness | 0.900 | 0.900 |
| over-refused answerable | 5 | 4 |
| traps | 6/6, nothing invented | 6/6, nothing invented |

Thirteen of forty questions changed verdict -- three better, six worse, four mixed -- and the means
did not move at all. The extra passages are distraction in about the same measure as they are
signal.

**So recall@k is a weak proxy for answer quality on this corpus.** A gold chunk arriving at rank 7
is not the same as a gold chunk being used, and the five passages that come with it cost roughly
what it gains. Production stays at 5.

## The two questions nothing reaches, and the second change that did not work

`version-06` and `version-14` both point at the same gold chunk: the `/version` verification's
**Checks** table. Its body is eight rows of "(none defined) | n/a | no check defined" boilerplate
with the TAP summary (`# pass 47`) at the very bottom. Embedded as a body, it is boilerplate:
the gold chunk sits at vector rank 77 and 102 for the two questions, full-text rank 24 and 52,
fused rank 39 and 122. The passage that identifies it -- "Verification: Add GET /version … >
Checks" -- is its *heading*, which the full-text arm indexes and the vector arm never saw:
`ingest/sync.py` embedded `chunk.body` alone.

So the obvious fix was to embed `heading_path + body`, with the content hash widened to cover it
so every stored vector re-embeds. All 491 did. Measured on dev:

| dev recall@5 | body only | heading + body |
|---|---|---|
| vector only | 0.5588 | 0.5882 (+1) |
| hybrid, RRF k=10 | 0.7353 | **0.6765 (−2)** |
| hybrid + rerank (production) | **0.8235** | **0.7647 (−2)** |

The vector arm improved by one question and production got worse by two. The likely mechanism:
the full-text arm already sees the heading, so giving it to the vector arm too makes the two arms
*agree* more -- and rank fusion earns its gain from the arms disagreeing. MRR@10 for hybrid rose
(0.4986 to 0.5620), so what fusion still found it ranked higher; it simply found less in the top
five. Reverted, and the store re-embedded body-only -- confirmed restored, not assumed: the
production configuration on dev came back at recall@5 0.8235, MRR@10 0.6485, precision@5 0.1824,
M4's published triple to four places.

The lesson is the same one the top-10 experiment taught from the other side: **a change that
helps the stage you were looking at can hurt the pipeline.** Vector-only recall was the wrong
thing to optimise, because nothing in production consumes vector-only recall.

## The pool went to 10, for latency (2026-09-18)

The tab made the cost visible in a way the eval never had: streaming showed first text arriving
12.2s into a 14.3s answer ([m5](m5-api-and-ui.md)). Generation is two seconds of it. The other
twelve are the local models, and the rerank is most of that -- it is the only step whose work
scales with the pool.

This document already had the evidence that the pool could shrink, so it was measured properly,
retrieval and answers both:

| candidate pool | recall@5 | MRR@10 | precision@5 | wall, 34 questions |
|---|---|---|---|---|
| 20 (previous) | 0.8235 | **0.6485** | 0.1824 | 319.8s |
| **10 (now)** | **0.8529** | 0.6385 | **0.1882** | **147.1s** |
| 5 | 0.7353 | 0.5564 | 0.1588 | 74.3s |

Pool 5 is just fusion's own top 5 reranked within itself, and it costs a tenth of recall -- the
reranker needs something to choose from. Pool 10 wins two of three retrieval metrics by about one
question each and loses MRR by about one rank, which at n=34 is not a difference worth naming.

Answers were then scored end to end, because this document's own top-10 experiment is the reason
not to ship a retrieval change on recall alone: **correctness 1.500 -> 1.513, faithfulness
0.900 -> 0.974, over-refusals 5 -> 5, traps 6/6 with nothing invented** -- and `eval/compare.py`
puts three questions better against five worse and says to read it as noise. (One judge verdict
truncated and was excluded, so that run is n=39 rather than 40; the faithfulness figure in
particular should not be read as a gain.)

So: **quality is a wash and the wall clock halves.** The change is made for the latency and is
not a quality claim. `PRODUCTION_POOL` here, `RERANK_CANDIDATES` in `api/main.py`,
`eval/run_generation.py` and Paddock's `studbook/index.ts`, and the parity gate's baseline all
moved together; a test on each side pins them to each other.

**The held-out numbers now describe the previous pool.** The test split was last scored at pool
20 with the previous prompt, and re-scoring it is the owner's call to spend, not a thing to do on
the way past.

## What this rules in and out

- **Ruled out:** a bigger candidate pool, a deeper arm depth, showing the generator more passages,
  prefixing the heading to the embedded text, and any query-side or chunking work aimed at
  reachability -- fusion already reaches 0.9118, and the two misses are one boilerplate table
  rather than a systemic gap.
- **Still open:** getting the gold chunk into the *top five* rather than merely into the top ten.
  That is a reranker-quality question, not a pipeline-shape one: a stronger cross-encoder
  (`bge-reranker-large`) is a local model swap that can be scored on recall alone, for free, before
  any generation run is paid for. Given that the top-10 experiment showed recall gains do not
  convert to answer quality automatically, it would need the generation run too before it could be
  believed.
- **The two misses are a chunking shape, not a retrieval one.** A checks table whose only
  informative line is a TAP summary under eight rows of "no check defined" would be reachable if
  the summary were its own chunk. That is a change to `ingest/chunker.py`'s table handling with a
  blast radius across every evidence table in the corpus, and two dev questions is not enough
  evidence to make it.

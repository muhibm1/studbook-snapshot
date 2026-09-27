# The cited refusal: two prompt experiments, both rejected

Date: 2026-09-18. The generator's remaining weakness on dev is an answer that opens "Not in the
record:" and then describes and cites what the passages do say. Five answerable dev questions were
refused in a saved run; in two of them no gold passage had been retrieved, so refusing was right and
no prompt should change it. The target was the other three, and the shape they share.

## Protocol, fixed before any run

Two dev runs of each prompt (40 questions, 6 of them traps), because one run cannot tell a small
change from noise. Adopt a candidate only if **both** of its runs refuse fewer answerable questions
than **both** baseline runs, no trap fails, nothing is invented, and neither faithfulness nor mean
correctness drops. The held-out split was not touched. `eval/run_generation.py --system-prompt-file`
runs a candidate without changing production and never logs it as a published run.

## Results

| run | answerable refused | correctness | fully correct | faithful | traps | invented |
|---|---|---|---|---|---|---|
| baseline 1 | 4 | 1.500 | 23 | 38/40 | 6/6 | 0 |
| baseline 2 | 4 | 1.471 | 23 | 39/40 | 6/6 | 1 |
| `partial-v4` 1 | 5 | 1.441 | 22 | 38/40 | 6/6 | 1 |
| `partial-v4` 2 | 4 | 1.485 | 23 | 38/39 | 6/6 | 0 |
| `no-cited-refusal-v5` 1 | 4 | 1.529 | 24 | 39/40 | 6/6 | 1 |
| `no-cited-refusal-v5` 2 | 3 | 1.529 | 24 | 37/40 | 6/6 | 1 |

**`partial-v4`** (`eval/prompts/partial-v4.txt`) rewrote the partial-answer and refusal rules to give
a partial gap its own wording ("The record does not say…") and reserve "Not in the record" for full
refusals. Rejected: more refusals, not fewer. The new phrase bled onto answers that were complete —
`version-02` became a judged refusal in both runs — while the stubborn case, `health-13`, kept
opening with the refusal phrase.

**`no-cited-refusal-v5`** (`eval/prompts/no-cited-refusal-v5.txt`) added one line and changed nothing
else: an answer that cites never begins with "Not in the record". Rejected, narrowly: refusals tied
in one run, and faithfulness dipped in the other. Reading every unfaithful answer, the dip is not
the new line's doing (`wh-11` flips between runs, and was unfaithful in a baseline too) — but the line
did not do its job either: `wh-13` still opened "Not in the record" and cited in the same answer.

## What replicated

- **`version-12` is fixed by either candidate, in all four candidate runs.** The answer-first shape
  works when the model adopts it.
- **`health-13` refused in all six runs**, whatever the prompt said. With its gold passage in the top
  five, it is the clearest case the prompt lever does not reach.
- **The baseline is stable**: both baseline runs refused exactly the same four questions, which is
  what pinning temperature 0 (docs/gold-set-audit.md) bought.
- **A standing faithfulness error, independent of the prompt**: `wh-15` was judged unfaithful in all
  six runs. It names the wrong release as the one "immediately after" another — an ordering claim
  across passages that no passage states. It is the project's one persistent unfaithful answer on dev,
  and it is temporal, not a refusal problem.

## Where that leaves it

Production keeps its prompt. Six paid runs say the prompt lever moves one question reliably and
leaves the stubborn ones where they are, which is the size of effect that fitting to dev would
exaggerate. The levers left are structural rather than wording: letting the generator see more of
a cross-doc question's evidence (docs/cross-doc.md found coverage, not wording, is what those
refusals track), and, for `wh-15`, making the record state release order explicitly.

## Follow-up: giving the record the release order (2026-09-19)

`wh-15` asks for "the plugin release immediately after the one that fixed the conductor's dispatch
contract". No passage stated release order, so WorkHorse gained a root `CHANGELOG.md` (and
`desk/CHANGELOG.md` for Paddock), built from the commits each release shipped, every hash checked to
exist. Both ingesters read it as a new `changelog` document type (migration
`0002_changelog_doc_type.sql`; the chunk parity gate holds with it included).

A first version said only "Follows 0.2.3." and was **not retrieved**: the question never names
0.2.3, it describes it. So every entry now names its predecessor by what it did ("Follows 0.2.3
(the conductor's dispatch contract: ...)"). That is a sound rule for any record retrieved in chunks,
and it was also written knowing this question exists, so what follows is a demonstration, not
evidence that it generalises.

- **Retrieval: fixed.** The 0.2.4 entry now comes back at rank 1 for `wh-15`.
- **The answer: not fixed.** In five runs the model still names 0.2.5 and the retro fix -- citing
  other passages and ignoring the one at rank 1 that states the answer. `wh-15` is a generation
  failure with the evidence in front of it, not a gap in the record.
- **Cost:** dev recall@5, coverage@5 and precision@5 unchanged; MRR@10 0.6385 -> 0.6282, because a
  changelog entry now outranks a gold passage for some questions (measured on the test project
  first, then reproduced exactly on production).

The changelogs stay: they are part of the record the tool exists to answer from, and Paddock's
Refresh would ingest them regardless. The published dev numbers are re-baselined for the larger
record (README), and Paddock's retrieval gate reproduces the new ones exactly.

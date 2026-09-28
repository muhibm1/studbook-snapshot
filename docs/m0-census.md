# M0: corpus census and design

Date: 2026-09-15. Numbers are measured with `wc -w` and small scripts over the files named, unless
a line says otherwise.

## 1. What is in the record

| Source | Location | Documents | Words |
|---|---|---|---|
| Change records (2 changes: `/health`, `/version`) | `paddock-demo/docs/sdlc/**` | 30 markdown files | 62,236 |
| WorkHorse commit messages | `git log` in `WorkHorse` | 53 commits | 6,115 |
| paddock-demo commit messages | `git log` in `paddock-demo` | 48 commits | 899 |
| Design docs and README | `WorkHorse/docs/superpowers/**`, `README.md` | 4 files | 12,896 |
| **Total** | | **135 documents** | **~82,000** |

That is about 107,000 tokens (at roughly 1.3 tokens per word, believed, not measured with a
tokenizer). At 400 to 600 words per chunk the live corpus is **roughly 200 to 260 chunks**. The
plan said "thousands"; that was wrong by an order of magnitude, and the README will say so. The
scale story in M4 therefore benchmarks at about 400 times the live corpus (around 100,000 synthetic
chunks) rather than 100 times.

Left out on purpose:

- `verify-logs/` (11 raw command logs). `verification.md` already records each command, exit code
  and log path; the raw logs are noise for retrieval.
- `WorkHorse/agents`, `skills`, `templates`, `vendor`: these are the system's instructions, not its
  history. They could become a second corpus later ("how does WorkHorse work?"), with their own
  questions.
- `examples/hello-service`: registered in Paddock but never onboarded, so it has no `docs/sdlc`.

## 2. Structure the corpus gives for free

Measured over the 30 change-record files:

| Doc type | Files | Words | Words per H2/H3 section | Table rows |
|---|---|---|---|---|
| spec | 2 | 12,027 | 286 | 207 |
| plan | 2 | 9,951 | 237 | 153 |
| release | 2 | 7,359 | 254 | 127 |
| intent | 2 | 6,030 | 195 | 98 |
| evals | 2 | 5,898 | 347 | 114 |
| retro | 2 | 4,977 | 262 | 41 |
| review-packet | 2 | 4,437 | 277 | 90 |
| adr | 8 | 3,589 | 112 | 46 |
| constraints | 1 | 2,677 | 191 | 45 |
| verification | 2 | 2,628 | 263 | 55 |
| conductor-log | 2 | 1,288 | 1,288 (no headings) | 0 |
| codebase-map | 1 | 1,104 | 123 | 30 |
| approvals | 2 | 271 | 54 | 0 |

What follows from it:

- **A heading section is usually one chunk.** Most sections are 110 to 350 words, inside the chunk
  budget. The chunker splits on H2/H3 first and only windows a section that is too long.
- **Tables are a quarter of the text.** The chunker must never split a table row, and when a table
  is longer than a chunk it repeats the header row in each piece, or the continuation rows lose
  their column meanings.
- **The conductor log has no headings** (1,288 words as one block), so it is split by paragraph
  with a small overlap.
- **No front matter anywhere**, but every change document opens with the same header lines
  (`Change id:`, `Date:`, `Risk tier: N (reason)`), and `approvals.md` is a list of
  `## G1: approved` blocks with `Who` and `When`. Metadata comes from parsing those, not from
  WorkHorse's `wh.js digest` (which only describes the active change).
- **Personal data:** 6 files contain the approver's email address (the owner's own). The store is
  private and never exposed; public screenshots and the README must not show it.

## 3. Metadata per chunk

| Field | Source | Used for |
|---|---|---|
| `id` | `<repo>/<path>#<heading-slug>`, `~n` for a continued section; `<repo>/git:<sha>` for commits | idempotent upsert, citations |
| `repo`, `path` | walk | filters, receipts |
| `doc_type` | file name (`spec.md`), folder (`adr/`), or `commit` | filters, per-type metrics |
| `change_id` | `docs/sdlc/<change>/` folder, confirmed against the `Change id:` line | cross-document questions |
| `tier` | `Risk tier:` line of the change's `intent.md` | filters |
| `gate` | heading or `approvals.md` block naming one gate | "what happened at G4" |
| `doc_date` | `Date:` line, or the commit date | temporal questions |
| `commit_sha` | repo HEAD at ingest (documents) or the commit itself | provenance: "pinned at" |
| `heading_path` | `H1 > H2 > H3`, also prepended to the embedded text | retrieval and receipts |
| `content_hash` | sha256 of the chunk text | skip unchanged chunks on re-ingest |

The schema draft is [store/migrations/0001_init.sql](../store/migrations/0001_init.sql): a private
`studbook` schema outside the Data API, RLS on with no policies, default privileges revoked, an
HNSW cosine index and a GIN full-text index, and an `eval_runs` table so every reported number has
a config hash and a commit behind it.

## 4. Gold set

Drafted from the corpus by three readers in parallel (one per change, one for WorkHorse's history),
then merged, balanced and checked. Each question carries a type, a reference answer, and gold
passages **anchored by verbatim quote**, not by chunk id: a chunk counts as gold when it contains
the quote. That keeps the labels valid across chunk sizes, which the chunk-size ablation in M4
needs, and a script re-checks every quote against its source on each run.

Types: decision-why, factual, evidence, cross-doc, temporal, and unanswerable traps (the system
must say it is not in the record). The set is split once, stratified by type, into dev (tuning)
and test (reported once, never tuned against). See `eval/`.

Result (checked with `python -m eval.gold split eval/drafts/*.jsonl`, all 60 quotes verbatim):

| Type | Total | Dev | Test |
|---|---|---|---|
| decision-why | 14 | 9 | 5 |
| factual | 12 | 8 | 4 |
| evidence | 8 | 5 | 3 |
| cross-doc | 10 | 7 | 3 |
| temporal | 7 | 5 | 2 |
| unanswerable | 9 | 6 | 3 |
| **All** | **60** | **40** | **20** |

By source: 16 from the `/health` change, 18 from `/version` (two of them compare the two changes),
20 from WorkHorse's commit history and design docs, and 6 hand-written (four traps, two that need
both repositories).

What review changed in the drafts:

- One drafted trap was answerable (it asked for the Node version; the onboarding docs record
  v21.7.3). It became a factual question, which also tests that the system does not refuse what it
  can answer.
- Two hand-written traps duplicated reader traps (Kubernetes, `/version` latency) and were replaced
  with absent terms checked by grep (Datadog, load balancer).
- Four `/version` quotes had been matched with whitespace normalised across line breaks; each was
  trimmed to its longest exact line so the check stays strict.

`eval/test.sha256` pins the test file. Any edit to `eval/test.jsonl` after this commit changes the
hash and is visible in review; the held-out numbers are only reported against the pinned file.

**Re-frozen 2026-09-16** (`816d0ef9…` -> `e6b49bd0…`) when the trap audit added `trap_kind` and
`denial_evidence`. All 20 test rows are identical on `id`, `type`, `question`, `reference_answer`
and `gold` -- diffed field by field against the previous frozen file, not asserted. See
[gold-set-audit.md](gold-set-audit.md).

**Read-through: done 2026-09-16.** The 60 questions were published as a reviewable page, and the
nine traps were additionally checked mechanically against the 491 ingested chunks. One dev question
(`health-16`) was rewritten because part of it was answerable; the rest stand.
[gold-set-audit.md](gold-set-audit.md) has the method and the per-trap result.

## 5. Design questions left open at M0

Two questions were deliberately deferred past the census. How each was settled:

| Question | Settled |
|---|---|
| Full-text configuration: `english` (stems words) or `simple` (keeps identifiers such as `wh.js` intact) | `english`, used by the GIN index (`store/migrations/0001_init.sql`) and by the query side (`retrieve/hybrid.py`). |
| LlamaIndex as a thin layer, or plain modules | Plain modules: retrieval, reranking and generation are small, direct code (`retrieve/`, `answer/`), with no framework dependency. |

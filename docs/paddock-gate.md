# Parity gate: can Studbook's query path run inside Paddock?

Date: 2026-09-16. The question was whether Studbook could be baked into Paddock (the Electron
desktop app) as a self-contained tab -- no Python, no service, no port, no API key -- by running
the same two models through their ONNX ports in Node. The risk is drift: ONNX exports do not
always reproduce sentence-transformers, and the store was embedded by Python. Drift would not
error; it would silently degrade recall. So before writing the port, a gate: a Node
reimplementation of the query path held to the Python pipeline, question by question.

`gate/` holds it. `dump_python_side.py` records what Python does for every answerable dev
question -- the query embedding, both arms' top 20, the fused top 20, the reranked top 10 with
scores. `gate.mjs` does the same in Node (`@huggingface/transformers` 3.7.1 for both models,
`pg` for the SQL, the same fusion) and compares.

## Result

| Stage | Node vs Python, 34 dev questions |
|---|---|
| Query embedding (`Xenova/bge-base-en-v1.5` fp32, CLS pooling, normalised) | cosine min **1.000000** |
| Vector arm, top 20 | **34/34 identical** |
| Full-text arm, top 20 | **34/34 identical** |
| Fused top 20, RRF k=10 | **34/34 identical** |
| Reranked top 5 (`Xenova/bge-reranker-base` fp32) | **34/34 identical**, max probability diff **4e-6** |
| recall@5 / MRR@10 / precision@5 | **0.8235 / 0.6485 / 0.1824** -- M2's published triple, to four places |

The port is safe to write. It is not, however, safe to write *naively*: the gate found three
things that a straight transliteration would have got wrong, two of them silently.

## What the gate caught

### 1. The reranker's truncation, which would have shipped as a silent regression

The first run agreed on 32 of 34 top-5 lists with raw scores differing by up to 7.7, and scored
*higher* than Python. Half of that was apples to oranges: sentence-transformers' `CrossEncoder`
applies a sigmoid, the ONNX head returns the logit. Compared as probabilities, short pairs agree
to 4e-6 -- and every remaining disagreement was a pair over 512 tokens, which is **46% of what the
reranker sees on this corpus** (157 of 340 pairs).

transformers.js's `truncation: true` on a text pair does not do what HuggingFace's default
`longest_first` does. Three strategies measured against Python's probabilities on all 157 long
pairs (`gate/truncation.mjs`):

| Strategy | max diff | pairs within 1e-4 |
|---|---|---|
| A: the library's own pair truncation | 0.0936 | 4/157 |
| **B: keep the first `512 - 4 - |query|` passage tokens, assemble the pair by id** | **0.000004** | **157/157** |
| C: keep the passage's last tokens | 0.971 | 1/157 |

B is what the port has to do, by token id, with the three special-token ids read off real
encodings (transformers.js does not expose `cls_token_id`; `tolist()` on int64 tensors yields
BigInts). Left to the library default, the port would have scored two questions differently, one
of them by luck in its favour, and nothing would have flagged it.

### 2. The connection string is libpq conninfo, not a URL

`STUDBOOK_DATABASE_URL` is `user=... password=... host=...`, which psycopg parses natively and
`pg` does not -- fed the conninfo it resolved the host as "base" (from "database"). The gate
parses it as libpq does (whitespace-separated key=value, single-quoted values with backslash
escapes). The port needs the same, or a URL-form secret of its own.

### 3. Every Python connection so far was encrypted but never authenticated

Node refused the pooler's certificate: `self-signed certificate in certificate chain`. The chain
is Supabase's own PKI -- `Supabase Root 2021 CA -> Supabase Intermediate 2021 CA ->
*.pooler.supabase.com` -- which the public CA store rightly rejects. **psycopg never surfaced this
because its default `sslmode=prefer` encrypts without verifying.** Ingestion, every eval run and
the service have all been talking to a host they never checked. The gate verifies against that
root, pinned in `gate/supabase-ca.pem` (extracted from the handshake by `gate/extract_ca.mjs`;
trust-on-first-use, so its subject and SHA-256 `80:70:25:AD:...:E6:CA:FA` should be cross-checked
against the CA certificate the Supabase dashboard offers).

**Fixed on the Python side the same day.** `store/connect.py` is now the only place a connection
is opened -- sixteen `psycopg.connect` call sites across twelve files went through it -- and it
sets `sslmode=verify-full` and `sslrootcert=store/supabase-ca.pem` as keyword arguments, which
psycopg applies over anything in the conninfo, so neither a connection string nor an environment
variable can downgrade it. A missing CA file refuses to connect rather than connecting
unverified. Proven deny-side first (`tests/test_connect.py`): the same host and credentials
against a genuine but unrelated self-signed root fail the handshake, and only then is the pinned
root shown to be accepted with `pg_stat_ssl` confirming the session is TLS. The port inherits
the same requirement and the same pinned root.

## The port (2026-09-17)

Landed in Paddock as `desk/src/main/studbook/` — conninfo parser, retrieval SQL and fusion, the
two ONNX models with the by-token-id truncation above, generation with the prompt from
`answer/prompts.py` verbatim — behind a `Studbook` tab on the rail, with the connection string
stored encrypted alongside the API key and every connection pinned to the same root as
`store/connect.py`. The gate became a Paddock test, `tests/studbook.gate.test.ts`, which runs this
dev split through the TypeScript and asserts the published triple to four places. First run
through the finished port: **0.8235 / 0.6485 / 0.1824, pass**, 324 seconds including a cold model
load. It is what keeps the two implementations honest, and it is free to re-run.

## Ingestion too, and the coupling it creates (2026-09-17)

The record grows every time WorkHorse runs, but the store only updated when someone ran this
repo's `ingest.sync`. So the ingester was ported as well — `parser.ts`, `chunker.ts`, `sync.ts` —
behind a **Refresh the record** button that re-ingests every repository in Paddock's registry
through the ingest role's connection string (a second stored secret). Its gate is
`gate/dump_chunks.py` on this side and `tests/studbook.chunks.gate.test.ts` on Paddock's: the two
parsers and chunkers must produce identical document and chunk ids, content hashes, heading paths,
gates and token counts over the same working trees. They did, first run: 139 documents, 494
chunks. Two things a transliteration would have missed and the tests pin: Python's `round()` is
half-to-even (`token_count` for every word count ending in 5), and `len("".split())` is zero.

One thing this port does that the Python one never needed: a registered folder inside another
repository (Paddock's `examples/hello-service`) contributes its files but not the enclosing
repository's commit history, which `git log` there would otherwise attribute to it.

**The product store and the eval store are the same store.** The first refresh from Paddock
added `hello-service` — one README, one chunk, 136 -> 137 documents — and the retrieval gate was
re-run against the refreshed store: **0.8235 / 0.6485 / 0.1824, unchanged**, because that
document reaches no dev question's top 5. That was a measurement, not a guarantee: a refresh that
adds a repository whose passages *do* compete would have moved the published numbers without
touching the eval.

**Closed the same day.** `Scope` gained a `repos` set on both sides (`retrieve/hybrid.py`,
Paddock's `retrieval.ts`), and every eval entry point — `eval/run.py`, `run_generation.py`,
`ceiling.py`, `check_answers.py`, the gate dumps, Paddock's gate test — now retrieves from exactly
`corpus.json`'s repositories (`eval/scope.py`); the service and the CLI deliberately do not, and
answer over the whole record. The set is part of the logged config, so a run over a different
corpus can never share a `config_hash` with one over this one. Re-measured under the scope, both
sides: **0.8235 / 0.6485 / 0.1824**. A Paddock refresh can now add whatever it likes to the store
without an eval number moving unless an eval change says so.

## What the port inherits

- Both models load in Node in 16 s and 57 s respectively, including a 1.5 GB first download;
  ~500 MB resident after. Fine for a long-lived desktop process, not for per-request startup.
- Everything stays fp32. The quantised defaults were never measured and must not be assumed.
- Ingestion and the eval harness stay Python: offline tools writing to a store Paddock only reads.
  The gate is what keeps the two implementations honest; it should be re-run whenever either side
  touches retrieval, and it is free.
- The store stays on Supabase, so "self-contained" still means "needs network". Secrets live in
  Paddock's main process where it already keeps them; the renderer gets an IPC call and never a
  connection string.
- The FastAPI service and its bearer key remain for the CLI and for anything outside Paddock.

## Where the two repositories stand now (2026-09-17)

The tab has two things the page does not — a thread, so a follow-up can say "it", and a streamed
answer — and neither was ported back. The page exists for use without Paddock, and a second
implementation of a thread is a second thing to keep honest.

So the split is: **Paddock is where the record gets asked and refreshed; this repo is the
reference implementation and the measurement harness.** The eval, the gold set, the ablations and
both gates live here, and every number either side reports is produced here. That only holds while
the gates are run — they are the whole reason two implementations of one retriever is a
defensible arrangement rather than a slow-motion divergence — so run them whenever either side
touches retrieval, chunking, or the prompt. Both are free apart from a few minutes of CPU.

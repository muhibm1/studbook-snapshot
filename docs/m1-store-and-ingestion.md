# M1: store and ingestion

Date: 2026-09-15. `studbook` Supabase project (`productionprojectref`, `us-east-1`), schema applied,
both repos ingested.

## What shipped

- **Schema applied** (`store/migrations/0001_init.sql`, via `store/bootstrap_db.py`): the
  `studbook` schema, outside the Data API's exposed schemas; RLS on every table; explicit
  permissive policies for `studbook_ingest`/`studbook_reader` rather than `bypassrls`; default
  privileges revoked; an HNSW cosine index and a GIN full-text index. A migration ledger
  (`studbook_meta.migrations`) makes re-running the bootstrap script safe.
- **Two login roles**, least-privilege: `studbook_ingest_login` (read/write on `documents`/
  `chunks`), `studbook_reader_login` (read-only there, read/write on `eval_runs`). Passwords are
  generated and rotated by `bootstrap_db.py` and written straight to Windows User environment
  variables via the registry -- never printed, never in a file.
- **Deny-side RLS tests** (`tests/test_rls.py`, 8 tests, run live): each role's grant proven, and
  proven denied where it should be -- a blocked write raises, and a blocked UPDATE is confirmed
  unchanged via the *other* role's connection, since a blocked UPDATE itself reports success on
  zero matched rows.
- **Ingestion pipeline** (`ingest/`): `parser.py` walks a repo's `docs/sdlc/**`, README, and
  `docs/superpowers/**`, plus `git log`, into `ParsedDocument`s; `chunker.py` splits each into
  `Chunk`s by heading (H2, then H3 if a section is oversized), windowing a heading-less document
  (a commit, `conductor-log.md`) by paragraph, never splitting a markdown table's rows unless the
  table alone is over budget, in which case each window repeats its header; `embed.py` wraps
  local `bge-base-en-v1.5`; `sync.py` upserts by content hash (document *and* chunk level) and
  deletes what a repo no longer produces. 34 unit tests, no DB needed for parsing/chunking logic.
- **Both configured repos ingested**: `paddock-demo` (79 documents, 362 chunks) and `workhorse`
  (57 documents, 129 chunks) -- 136 documents, 491 chunks. Re-running immediately re-embedded
  zero chunks (491 unchanged), confirming the idempotent-upsert design works.
- **Spot-checked against the gold set**: 6 random gold quotes (spread across both repos, markdown
  docs and commits) are retrievable verbatim from `chunks.body`. A 5-question vector-only
  retrieval probe hit 3/5 in the top 5 -- the two misses were short factual/numeric lookups
  (a risk tier, in one case), exactly the shape M2's planned hybrid (vector + full-text) retrieval
  is for; not a defect in this milestone.

## Incidents this milestone (all fixed, all regression-tested)

1. **A parsing bug leaked a password fragment into this conversation.** `connection_url()` used
   `urllib.parse`, a generic-URI parser that requires percent-encoding; a real Supabase password
   didn't have it, the resulting `ValueError` echoed a fragment of the password, and it reached
   the terminal. Fixed by switching to `psycopg`'s own libpq-aware `conninfo_to_dict`/
   `make_conninfo` (no manual encoding, can't produce that failure shape) and regression-tested
   with nine deliberately awkward passwords (`tests/test_bootstrap_db.py`). **The exposed password
   was rotated** before any further connection attempt.
2. **The direct-connection host is IPv6-only**; this machine has no IPv6 route. Added automatic
   fallback to Supabase's Supavisor pooler (IPv4), used for every connection string this project
   writes -- confirmed live which of the two regional pooler shards actually owns this project.
3. **The Supabase dashboard's connection-string box ships a literal bracketed placeholder**
   (`[YOUR-PASSWORD]`) instead of the real password; pasted as-is, authentication fails with no
   hint of why. No code fix (nothing to fix), but worth recording: always confirmed by connecting
   the actual database password to substitute, not by trusting a connection string was final.
4. **`ALTER ROLE ... PASSWORD %s` is DDL** and rejects a bind parameter ("syntax error at or near
   `$1`") -- fixed with `sql.Literal`, which quotes/escapes into valid literal SQL text instead.
5. **`run()` rehomed a connection onto the working route by host/port only, forgetting the
   Supavisor username suffix** -- connects, then fails mid-session ("no tenant identifier
   provided"). Extracted into `admin_url_for_route()` and regression-tested alongside (3).
6. **`ALTER ROLE ... SET search_path` is a role-level default that Supabase's pooler does not
   apply** (it reuses backend connections across logical sessions; a role-level GUC default is
   only applied when Postgres starts a genuinely new backend). The pgvector type
   (`extensions.vector`) therefore never resolved for `pgvector-python`'s `register_vector()`
   through the pooler. Fixed by having every client `SET search_path` explicitly, once, right
   after connecting -- reliable regardless of pooling. The role-level default stays too, as a
   no-cost fallback for a future direct connection.

## Open items into M2

- Hybrid retrieval (vector + Postgres full-text, reciprocal rank fusion) -- the vector-only probe
  above is the concrete motivating data point.
- The eval harness proper: Recall@5/MRR@10/Precision@5 against `eval/dev.jsonl`, broken down by
  question type and doc type.
- CI for this project doesn't exist yet; when it does, per the security baseline,
  `tests/test_rls.py` must be asserted to actually run (non-zero test count), never allowed to
  skip silently green for missing credentials.

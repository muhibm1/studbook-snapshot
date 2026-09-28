-- Studbook store, migration 0001. Applied to the `studbook` Supabase project in M1; deny-side
-- tests live in tests/test_rls.py, which connects as each role in turn and checks what it cannot
-- do, not only what it can.
--
-- Design:
--   * Everything lives in the `studbook` schema, which is NOT added to the Data API's exposed
--     schemas, so anon and authenticated clients cannot reach it over HTTP at all. Only the
--     server-side ingest and query code connects, directly over Postgres.
--   * RLS is enabled on every table. public/anon/authenticated get no policy at all (deny-all,
--     confirmed by test); studbook_ingest and studbook_reader get explicit permissive policies
--     (below) rather than `bypassrls`, so the scope is a visible line here, not a role attribute.
--   * Chunk ids are natural keys ('<repo>/<path>#<heading-slug>', '~n' for a continued section),
--     so re-ingestion is an idempotent upsert on id and a stale chunk is anything not re-emitted.

create schema if not exists extensions;
create extension if not exists vector with schema extensions;

create schema if not exists studbook;

revoke all on schema studbook from public, anon, authenticated;
-- Functions created later in this schema must be granted explicitly, never by platform default.
alter default privileges in schema studbook revoke execute on functions from public, anon, authenticated;
alter default privileges in schema studbook revoke all on tables from public, anon, authenticated;

-- One row per source document: a markdown file, or a single git commit message.
create table studbook.documents (
  id text primary key,                          -- '<repo>/<path>' or '<repo>/git:<sha>'
  repo text not null check (repo ~ '^[a-z0-9][a-z0-9._-]{0,63}$'),
  path text not null check (char_length(path) between 1 and 400),
  doc_type text not null check (doc_type in (
    'intent', 'spec', 'adr', 'evals', 'plan', 'verification', 'review-packet', 'release',
    'approvals', 'retro', 'conductor-log', 'instinct', 'codebase-map', 'constraints',
    'design-doc', 'readme', 'commit'
  )),
  change_id text check (char_length(change_id) <= 120),   -- null outside docs/sdlc/<change>/
  tier smallint check (tier between 0 and 3),              -- from the change's intent.md
  doc_date date,
  commit_sha text not null check (commit_sha ~ '^[0-9a-f]{40}$'), -- repo HEAD at ingest, or the commit itself
  content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
  ingested_at timestamptz not null default now(),
  unique (repo, path)
);

-- One row per retrievable passage. Metadata used for filtering lives on documents (joined);
-- at a few thousand chunks the join is cheaper than keeping denormalised copies consistent.
create table studbook.chunks (
  id text primary key,
  document_id text not null references studbook.documents (id) on delete cascade,
  ordinal integer not null check (ordinal >= 0),
  heading_path text not null check (char_length(heading_path) <= 1000),
  gate text check (gate ~ '^G[0-5]$'),          -- set when the section is about one gate
  body text not null check (char_length(body) between 1 and 20000),
  token_count integer not null check (token_count > 0),
  content_hash text not null check (content_hash ~ '^[0-9a-f]{64}$'),
  embedding extensions.vector(768) not null,    -- bge-base-en-v1.5, normalised
  fts tsvector generated always as (
    to_tsvector('english', heading_path || ' ' || body)
  ) stored,
  unique (document_id, ordinal)
);

create index chunks_document_id_idx on studbook.chunks (document_id);
create index chunks_fts_idx on studbook.chunks using gin (fts);
-- HNSW over cosine distance. m/ef_construction are pgvector's defaults, restated so the scale
-- benchmark in M4 has a named baseline to vary.
create index chunks_embedding_idx on studbook.chunks
  using hnsw (embedding extensions.vector_cosine_ops) with (m = 16, ef_construction = 64);

-- Every evaluation run, so each reported number is a reproducible claim.
create table studbook.eval_runs (
  id bigint generated always as identity primary key,
  split text not null check (split in ('dev', 'test')),
  config_hash text not null check (config_hash ~ '^[0-9a-f]{64}$'),
  config jsonb not null,
  studbook_commit text not null check (studbook_commit ~ '^[0-9a-f]{40}$'),
  metrics jsonb not null,
  started_at timestamptz not null default now()
);

alter table studbook.documents enable row level security;
alter table studbook.chunks enable row level security;
alter table studbook.eval_runs enable row level security;
-- No policies for public/anon/authenticated on purpose: those roles see and change nothing (and
-- cannot reach this schema over the Data API regardless, since it is not exposed there). The two
-- server roles below get their own explicit policies rather than `bypassrls`, so "who may touch
-- which rows" stays visible in this file instead of hidden in a role attribute (M1 decision).

-- Group roles; the login roles that inherit them are created by store/bootstrap_db.py, which also
-- generates and stores their passwords -- never in this repository.
create role studbook_ingest nologin;
create role studbook_reader nologin;

grant usage on schema extensions to studbook_ingest, studbook_reader;
grant usage on schema studbook to studbook_ingest, studbook_reader;
grant select, insert, update, delete on studbook.documents, studbook.chunks to studbook_ingest;
grant select on studbook.documents, studbook.chunks to studbook_reader;
grant insert, select on studbook.eval_runs to studbook_reader;
grant usage on sequence studbook.eval_runs_id_seq to studbook_reader;

-- Single-tenant tool, no per-row identity to filter by: each policy's USING/CHECK is `true`, so
-- it grants exactly what the table GRANT above already allows, no more -- the policy's job here
-- is to make that scope an explicit, auditable line rather than an implicit bypassrls escape.
create policy studbook_ingest_rw on studbook.documents for all
  to studbook_ingest using (true) with check (true);
create policy studbook_ingest_rw on studbook.chunks for all
  to studbook_ingest using (true) with check (true);
create policy studbook_reader_ro on studbook.documents for select
  to studbook_reader using (true);
create policy studbook_reader_ro on studbook.chunks for select
  to studbook_reader using (true);
create policy studbook_reader_eval on studbook.eval_runs for all
  to studbook_reader using (true) with check (true);

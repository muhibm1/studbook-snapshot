# Hosted configuration

What lives in Supabase rather than in this repository, recorded so it can be reviewed and so drift
is noticeable. Last checked **2026-09-18**, read-only, through the Supabase management API and the
database catalog. Each line says whether it was **confirmed** (queried) or is **not verified from
here** (a dashboard setting the API used did not expose).

## Project

| setting | value | status |
|---|---|---|
| project | `Studbook`, one project; production and the only one | confirmed |
| region | us-east-1 | confirmed |
| Postgres | 17.6 | confirmed |
| pgvector | 0.8.2 (needed for `hnsw.iterative_scan`, docs/m4-evaluation-and-scale.md) | confirmed |
| server TLS | `ssl = on`; clients verify with `sslmode=verify-full` against the pinned root in `store/` | confirmed |
| security advisor | no findings | confirmed |
| performance advisor | one: the HNSW index is unused, expected at this size (see the m4 addendum) | confirmed |

There is no separate test project. CI's live job requires one and will not use this one
(docs/ci.md).

## Who can reach the data

The whole data model is in migrations (`store/migrations/`); these are the properties those
migrations are meant to produce, checked against the live catalog rather than assumed from them.

| property | value | status |
|---|---|---|
| `anon`, `authenticated`, `PUBLIC` grants on any `studbook` table | none | confirmed |
| `anon` / `authenticated` usage on the `studbook` schema | none, so Supabase's REST API cannot reach it with the public key | confirmed |
| functions in `studbook` or `public` | none, so no `SECURITY DEFINER` surface | confirmed |
| row-level security | enabled on `documents`, `chunks`, `eval_runs` | confirmed |
| roles | `studbook_reader`, `studbook_ingest` (no login), with `_login` members for connecting; none can bypass RLS | confirmed |
| what each role may do | reader: select; ingest: write documents and chunks; `eval_runs` append-only by grant | enforced by `tests/test_rls.py`, run live 2026-09-18 |

RLS is enabled but not forced, so the table owner bypasses it. The owner is the admin role that
applies migrations, never a role the app connects as.

## Not verified from here

Dashboard-only settings the API used for this page did not expose. Check them in the dashboard and
record the values here when you do.

- Network restrictions (allowed IP ranges for direct database connections).
- Backups and point-in-time recovery.
- Whether the Data API (PostgREST) has `studbook` in its exposed schemas. It could only ever expose
  what the grants above allow, which is nothing, but a schema that is exposed and ungranted is one
  grant away from public.
- Supabase Auth settings: not used. Studbook has no end users; the service's API key is its own
  (`STUDBOOK_API_KEY`), and Paddock connects directly as the reader role.

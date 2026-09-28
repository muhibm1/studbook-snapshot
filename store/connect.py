"""The one place Studbook opens a database connection, so that TLS verification cannot be
forgotten at a call site.

Until 2026-09-16 every connection went through psycopg's default `sslmode=prefer`, which
encrypts but never checks who is on the other end -- the Paddock parity gate found this when
Node refused the pooler's certificate chain (docs/paddock-gate.md). Supabase signs its database
certificates with its own root ("Supabase Root 2021 CA"), which no public CA store contains, so
verification needs that root pinned. It lives next to this file; `verify-full` checks the chain
against it AND that the hostname matches the certificate, which is the mode that actually rules
out a man-in-the-middle.

Both are set as keyword arguments, which psycopg applies over anything in the conninfo string:
a connection string cannot downgrade them, and neither can an environment variable.
"""

from __future__ import annotations

from pathlib import Path

import psycopg

CA_CERT = Path(__file__).resolve().with_name("supabase-ca.pem")
SSL_MODE = "verify-full"


def connect(conninfo: str, **kwargs) -> psycopg.Connection:
    """psycopg.connect with the certificate chain verified against the pinned Supabase root.

    Refuses to fall back: if the CA file is missing the connection is not attempted, rather than
    attempted unverified. That is the failure shape this project's security baseline forbids --
    silent and consequential -- and here it would be silent in exactly the way it was from the
    first commit until 2026-09-16.
    """
    if not CA_CERT.is_file():
        raise RuntimeError(
            f"{CA_CERT} is missing: refusing to open an unverified database connection. "
            "It is Supabase's public root certificate and is committed; restore it from git."
        )
    return psycopg.connect(conninfo, sslmode=SSL_MODE, sslrootcert=str(CA_CERT), **kwargs)


# Every session that retrieves runs this first. Two settings, kept in one place because the second
# was missing from twenty hand-written copies of the first:
#
#   search_path      the schema, plus `extensions`, where pgvector's operators live.
#   iterative_scan   pgvector's HNSW index returns ef_search candidates (40 by default) and a
#                    WHERE clause filters them afterwards. A scoped search -- one repository, one
#                    change -- could keep none: forced onto the index path, "workhorse passages
#                    nearest a paddock-demo passage" returned 0 of 20 with it off and 20 of 20 with
#                    strict_order (2026-09-18). Masked today because at ~500 passages the planner
#                    sorts instead of using the index; it would surface, silently, as the corpus
#                    grows. strict_order keeps exact distance order, which fusion and the reranker
#                    rely on. Needs pgvector >= 0.8 (the project runs 0.8.2).
READER_SESSION = "set search_path to studbook, extensions; set hnsw.iterative_scan = strict_order"


def prepare_session(cur: psycopg.Cursor) -> None:
    """Run READER_SESSION on a cursor. Its two statements need psycopg's no-parameter path."""
    cur.execute(READER_SESSION)

"""Ingests one or more configured repos (corpus.json) into the studbook store: parse -> chunk ->
embed only what changed -> upsert -> delete what's gone. Idempotent: content-hashed at both the
document and the chunk level, so a re-run with nothing changed touches no rows and embeds nothing.

    .venv/Scripts/python.exe -m ingest.sync --all
    .venv/Scripts/python.exe -m ingest.sync --repo paddock-demo --repo workhorse

Needs STUDBOOK_INGEST_DATABASE_URL in the environment (store/bootstrap_db.py sets it).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from eval.gold import load_corpus
from ingest.chunker import Chunk, chunk_document
from ingest.embed import Embedder
from ingest.parser import ParsedDocument, walk_repo
from store.connect import connect, prepare_session

UPSERT_DOCUMENT = """
insert into studbook.documents (id, repo, path, doc_type, change_id, tier, doc_date, commit_sha, content_hash)
values (%(id)s, %(repo)s, %(path)s, %(doc_type)s, %(change_id)s, %(tier)s, %(doc_date)s, %(commit_sha)s, %(content_hash)s)
on conflict (id) do update set
  repo = excluded.repo, path = excluded.path, doc_type = excluded.doc_type, change_id = excluded.change_id,
  tier = excluded.tier, doc_date = excluded.doc_date, commit_sha = excluded.commit_sha,
  content_hash = excluded.content_hash,
  ingested_at = case when studbook.documents.content_hash is distinct from excluded.content_hash
                      then now() else studbook.documents.ingested_at end
"""

UPSERT_CHUNK = """
insert into studbook.chunks (id, document_id, ordinal, heading_path, gate, body, token_count, content_hash, embedding)
values (%(id)s, %(document_id)s, %(ordinal)s, %(heading_path)s, %(gate)s, %(body)s, %(token_count)s, %(content_hash)s, %(embedding)s)
on conflict (id) do update set
  ordinal = excluded.ordinal, heading_path = excluded.heading_path, gate = excluded.gate,
  body = excluded.body, token_count = excluded.token_count, content_hash = excluded.content_hash,
  embedding = excluded.embedding
"""

EMBED_BATCH_SIZE = 64


@dataclass
class SyncStats:
    repo: str
    documents_upserted: int
    documents_deleted: int
    chunks_embedded: int
    chunks_unchanged: int
    chunks_deleted: int

    def __str__(self) -> str:
        return (
            f"{self.repo}: {self.documents_upserted} document(s) upserted, {self.documents_deleted} deleted; "
            f"{self.chunks_embedded} chunk(s) (re)embedded, {self.chunks_unchanged} unchanged, "
            f"{self.chunks_deleted} deleted"
        )


def _document_row(doc: ParsedDocument) -> dict:
    return {
        "id": doc.id, "repo": doc.repo, "path": doc.path, "doc_type": doc.doc_type,
        "change_id": doc.change_id, "tier": doc.tier, "doc_date": doc.doc_date,
        "commit_sha": doc.commit_sha, "content_hash": doc.content_hash,
    }


def _chunk_row(chunk: Chunk, embedding: list[float]) -> dict:
    return {
        "id": chunk.id, "document_id": chunk.document_id, "ordinal": chunk.ordinal,
        "heading_path": chunk.heading_path, "gate": chunk.gate, "body": chunk.body,
        "token_count": chunk.token_count, "content_hash": chunk.content_hash, "embedding": embedding,
    }


def sync_repo(repo: str, repo_path: Path, embedder: Embedder, conn: psycopg.Connection) -> SyncStats:
    docs = walk_repo(repo, repo_path)
    chunks_by_doc = {d.id: chunk_document(d) for d in docs}
    all_chunks = [c for cs in chunks_by_doc.values() for c in cs]

    with conn.cursor() as cur:
        doc_ids = [d.id for d in docs]
        cur.execute("select id, content_hash from studbook.chunks where document_id = any(%s)", (doc_ids,))
        existing_chunk_hash = dict(cur.fetchall())

        to_embed = [c for c in all_chunks if existing_chunk_hash.get(c.id) != c.content_hash]
        unchanged_count = len(all_chunks) - len(to_embed)

        embeddings: dict[str, list[float]] = {}
        for start in range(0, len(to_embed), EMBED_BATCH_SIZE):
            batch = to_embed[start : start + EMBED_BATCH_SIZE]
            vectors = embedder.embed([c.body for c in batch])
            embeddings.update(zip((c.id for c in batch), vectors))
            print(f"  embedded {min(start + EMBED_BATCH_SIZE, len(to_embed))}/{len(to_embed)}")

        for doc in docs:
            cur.execute(UPSERT_DOCUMENT, _document_row(doc))
            for chunk in chunks_by_doc[doc.id]:
                if chunk.id in embeddings:
                    cur.execute(UPSERT_CHUNK, _chunk_row(chunk, embeddings[chunk.id]))

        produced_chunk_ids = [c.id for c in all_chunks]
        cur.execute(
            "delete from studbook.chunks where document_id = any(%s) and not (id = any(%s))",
            (doc_ids, produced_chunk_ids),
        )
        chunks_deleted = cur.rowcount

        cur.execute(
            "delete from studbook.documents where repo = %s and not (id = any(%s))",
            (repo, doc_ids),
        )
        documents_deleted = cur.rowcount

    conn.commit()
    return SyncStats(
        repo=repo,
        documents_upserted=len(docs),
        documents_deleted=documents_deleted,
        chunks_embedded=len(to_embed),
        chunks_unchanged=unchanged_count,
        chunks_deleted=chunks_deleted,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", action="append", default=[], help="a repo name from corpus.json; repeatable")
    parser.add_argument("--all", action="store_true", help="ingest every repo in corpus.json")
    args = parser.parse_args()

    corpus = load_corpus()
    if args.all:
        names = list(corpus)
    elif args.repo:
        unknown = [r for r in args.repo if r not in corpus]
        if unknown:
            parser.error(f"not in corpus.json: {', '.join(unknown)}")
        names = args.repo
    else:
        parser.error("pass --repo NAME (repeatable) or --all")

    ingest_url = os.environ.get("STUDBOOK_INGEST_DATABASE_URL")
    if not ingest_url:
        print("STUDBOOK_INGEST_DATABASE_URL is not set in this process's environment.", file=sys.stderr)
        return 2

    print("loading the embedding model ...")
    embedder = Embedder()

    with connect(ingest_url, autocommit=False) as conn:
        with conn.cursor() as cur:
            # Not reliable via the role's own default (see bootstrap_db.py's ensure_login_role):
            # Supabase's pooler reuses backend connections across logical sessions, so a role-level
            # ALTER ROLE ... SET search_path is silently skipped through it. Set it explicitly on
            # every connection instead -- works whether this is pooled or direct. Needed so the
            # unqualified `vector` type (extensions.vector) resolves for register_vector() below;
            # every query in this module already fully qualifies studbook.documents/chunks.
            prepare_session(cur)
        register_vector(conn)
        for name in names:
            print(f"ingesting {name} ({corpus[name]}) ...")
            stats = sync_repo(name, corpus[name], embedder, conn)
            print(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Synthetic scale benchmark: pgvector HNSW query latency at ~100x the live corpus's chunk count.
The live corpus is small -- 491 chunks (docs/m1-store-and-ingestion.md) -- so this measures a
temporary table of synthetic random unit vectors at the same dimensionality and the same HNSW
parameters as the real one, not the real data, and drops the table when done (a `finally` block
covers a failed run too, not just a clean one).

    .venv/Scripts/python.exe -m eval.scale_benchmark

Needs STUDBOOK_ADMIN_DATABASE_URL: creating and dropping a table needs CREATE privilege on the
studbook schema, which neither application role has, deliberately (store/migrations/0001_init.sql).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path

from pgvector.psycopg import register_vector

from ingest.embed import Embedder
from store.bootstrap_db import admin_url_for_route, probe_route
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
N_SYNTHETIC = 100_000
DIMENSIONS = 768
HNSW_M, HNSW_EF_CONSTRUCTION = 16, 64  # matches store/migrations/0001_init.sql exactly


def sample_query_vectors(embedder: Embedder, n: int) -> list[list[float]]:
    rows = [json.loads(l) for l in (ROOT / "eval" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [r["question"] for r in rows[:n]]
    return embedder.embed(questions)


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def main() -> int:
    admin_url = os.environ.get("STUDBOOK_ADMIN_DATABASE_URL")
    if not admin_url:
        print("STUDBOOK_ADMIN_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    route = probe_route(admin_url)
    url = admin_url_for_route(admin_url, route)

    print("embedding query sample ...")
    query_vectors = sample_query_vectors(Embedder(), n=20)

    with connect(url, autocommit=True) as conn:
        with conn.cursor() as cur:
            prepare_session(cur)
            # Supabase's default statement_timeout is short enough that the 100k-row COPY got
            # cancelled partway through the first time this ran (30,480 of 100,000 rows) --
            # generous but still bounded for this one-off bulk load, scoped to this session only.
            cur.execute("set statement_timeout = '10min'")
        register_vector(conn)

        try:
            print(f"creating a benchmark table with {N_SYNTHETIC:,} synthetic vectors ...")
            with conn.cursor() as cur:
                cur.execute("drop table if exists studbook_meta.bench_chunks")
                cur.execute(
                    "create table studbook_meta.bench_chunks "
                    "(id bigint primary key, embedding extensions.vector(768) not null)"
                )

            # Generated server-side, deliberately: 100,000 x 768 floats in libpq's text format is
            # roughly 1.5 GB on the wire, which a COPY from this machine to a hosted Postgres
            # spent many minutes on before being killed. `random()` per element inside a LATERAL
            # (so each row gets its own vector, not one vector repeated) with pgvector's own
            # l2_normalize to match the unit-normalised embeddings ingest/embed.py produces.
            t0 = time.monotonic()
            with conn.cursor() as cur:
                cur.execute(
                    "insert into studbook_meta.bench_chunks (id, embedding) "
                    "select g.i, extensions.l2_normalize(v.vec) "
                    "from generate_series(1, %s) as g(i) "
                    "cross join lateral ("
                    "  select array_agg(random())::real[]::extensions.vector as vec"
                    "  from generate_series(1, %s)"
                    ") as v",
                    (N_SYNTHETIC, DIMENSIONS),
                )
            print(f"  inserted in {time.monotonic() - t0:.1f}s")

            print(f"building HNSW index (m={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}) ...")
            t0 = time.monotonic()
            with conn.cursor() as cur:
                cur.execute(
                    "create index bench_chunks_embedding_idx on studbook_meta.bench_chunks "
                    "using hnsw (embedding extensions.vector_cosine_ops) "
                    f"with (m = {HNSW_M}, ef_construction = {HNSW_EF_CONSTRUCTION})"
                )
            print(f"  built in {time.monotonic() - t0:.1f}s")

            print(f"timing {len(query_vectors)} nearest-neighbour queries (top 5) ...")
            latencies_ms = []
            with conn.cursor() as cur:
                for v in query_vectors:
                    t0 = time.perf_counter()
                    # `<=>` (cosine), matching the vector_cosine_ops index -- see
                    # retrieve/hybrid.py's retrieve_vector for why this operator choice matters.
                    cur.execute(
                        "select id from studbook_meta.bench_chunks order by embedding <=> %s::vector limit 5", (v,)
                    )
                    cur.fetchall()
                    latencies_ms.append((time.perf_counter() - t0) * 1000)

            print(f"\np50: {percentile(latencies_ms, 0.50):.2f} ms")
            print(f"p95: {percentile(latencies_ms, 0.95):.2f} ms")
            print(f"max: {max(latencies_ms):.2f} ms")
            print(f"mean: {statistics.mean(latencies_ms):.2f} ms")
        finally:
            print("\ndropping the benchmark table ...")
            with conn.cursor() as cur:
                cur.execute("drop table if exists studbook_meta.bench_chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

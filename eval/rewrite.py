"""Does rewriting a follow-up into a standalone question beat carrying the previous question?

The two sets from eval/dual_query.py -- genuine follow-ups (threads.jsonl, n=12) and changes of
subject (unrelated pairs over dev, n=34) -- with the rewrite (retrieve/rewrite.py) as a new arm:

    bare          the question alone
    carried       previous + question: what Paddock sends today
    rewrite       the rewritten question, retrieved AND reranked with -- the cross-encoder gets a
                  self-contained question too, which it never had for a follow-up
    rewrite/q     the rewritten question retrieved with, the original reranked against: isolates
                  whether the gain is in retrieval or in the reranker

Every rewrite is printed with its method, because a number for this is worth less than reading
what it actually wrote. Costs one Haiku call per pair (46 in all, well under a cent).

    .venv/Scripts/python.exe -m eval.rewrite
"""

from __future__ import annotations

import os
import sys
from collections import Counter

from pgvector.psycopg import register_vector

from eval.dual_query import candidates, load_sets
from eval.metrics import is_hit
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.rerank import Reranker
from retrieve.rewrite import carried, rewrite_query
from store.connect import connect, prepare_session

TOP_K = 5
ARMS = ("bare", "carried", "rewrite", "rewrite/q")


def main() -> int:
    for var in ("STUDBOOK_DATABASE_URL", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            print(f"{var} is not set.", file=sys.stderr)
            return 2

    import anthropic

    client = anthropic.Anthropic()
    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    table = {}

    with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        for name, rows in load_sets().items():
            print(f"\n{name}, n={len(rows)}")
            hits, methods = Counter(), Counter()
            for rid, previous, question, gold in rows:
                rw = rewrite_query(question, [previous], client=client)
                methods[rw.method] += 1
                plans = {
                    "bare": (question, question),
                    "carried": (carried(question, [previous]), question),
                    "rewrite": (rw.query, rw.query),
                    "rewrite/q": (rw.query, question),
                }
                found = {}
                for arm, (query, rerank_against) in plans.items():
                    top = reranker.rerank(rerank_against, candidates(cur, embedder, [query], scope), top_k=TOP_K)
                    found[arm] = any(is_hit(r, gold) for r in top)
                    hits[arm] += found[arm]
                marks = "  ".join(f"{a}={'hit ' if found[a] else 'MISS'}" for a in ARMS)
                print(f"  {rid:<12} {marks}  [{rw.method}]")
                print(f"      asked:    {question}")
                if rw.method == "fallback-guard":
                    print(f"      rejected: {rw.rejected}")
                elif rw.method == "rewritten":
                    print(f"      searched: {rw.query}")
            table[name] = (hits, len(rows), methods)

    print(f"\n{'':34}" + "".join(f"{a:>13}" for a in ARMS))
    for name, (hits, n, methods) in table.items():
        print(f"{name:<34}" + "".join(f"{hits[a]:>5}/{n:<3} {hits[a] / n:.2f}" for a in ARMS))
        print(f"{'':34}rewrite methods: " + ", ".join(f"{m} {c}" for m, c in sorted(methods.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

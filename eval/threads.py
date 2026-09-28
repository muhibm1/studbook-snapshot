"""Does carrying the previous question into a follow-up's retrieval query actually help?

Paddock's Studbook tab keeps a thread, and a follow-up retrieves on the last two questions plus
the current one (`retrievalQuery` in desk/src/main/studbook/index.ts). That heuristic shipped
unmeasured, with its own docstring saying so, because the gold set is single-turn. This is the
measurement it owed.

`eval/threads.jsonl` pairs an existing gold question with a follow-up written to be answered by
**the same passage** -- so the gold quote is the first question's, already verified verbatim by
`eval/gold.py`, and the only thing being tested is whether retrieval can still find that passage
when the question stops naming its subject. Three queries per pair:

    bare       the follow-up alone, which is what a stateless tab would send
    carried    the first question + the follow-up, which is what the tab sends
    first      the original question, as an upper bound -- the passage was written for it

Machine-drafted from headings, then reviewed on 2026-09-18 by reading every gold passage in full:
t-07 was reworded (its draft asked something its passage did not answer) and t-08 is marked weak
(docs/follow-ups.md). It is deliberately separate from gold.jsonl and feeds no published metric.

    .venv/Scripts/python.exe -m eval.threads
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pgvector.psycopg import register_vector

from eval.gold import read_jsonl
from eval.metrics import is_hit
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import retrieve_hybrid
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
RERANK_CANDIDATES = 10  # production (api/main.py)
HISTORY_TURNS = 2  # what the tab carries


def main() -> int:
    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    gold = {r["id"]: r for r in read_jsonl(ROOT / "eval" / "gold.jsonl")}
    pairs = read_jsonl(ROOT / "eval" / "threads.jsonl")
    missing = [p["id"] for p in pairs if p["first"] not in gold]
    if missing:
        print(f"threads.jsonl references unknown gold ids: {missing}", file=sys.stderr)
        return 2

    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    hits = {"bare": 0, "carried": 0, "first": 0}
    rows = []

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        for pair in pairs:
            source = gold[pair["first"]]
            first, follow = source["question"], pair["follow_up"]
            queries = {
                "bare": follow,
                "carried": " ".join([*[first][-HISTORY_TURNS:], follow]),
                "first": first,
            }
            result = {}
            for name, query in queries.items():
                embedding = embedder.embed([query])[0]
                candidates = retrieve_hybrid(cur, query, embedding, top_k=RERANK_CANDIDATES, scope=scope)
                # The cross-encoder re-scores against the question actually being answered, which
                # for a follow-up is the follow-up -- matching what the tab does.
                top = reranker.rerank(follow if name != "first" else first, candidates, top_k=TOP_K)
                found = any(is_hit(r, source["gold"]) for r in top)
                hits[name] += found
                result[name] = found
            rows.append((pair["id"], pair["first"], result, follow))
            print(f"  {pair['id']}  bare={'hit ' if result['bare'] else 'MISS'}  "
                  f"carried={'hit ' if result['carried'] else 'MISS'}  "
                  f"first={'hit ' if result['first'] else 'MISS'}   {follow[:52]}")

    n = len(pairs)
    print(f"\n{n} follow-up pair(s), recall@{TOP_K} of the passage the first question was written for:")
    for name, label in (("bare", "follow-up alone (a stateless tab)"),
                        ("carried", "first + follow-up (what the tab sends)"),
                        ("first", "the original question (upper bound)")):
        print(f"  {label:42} {hits[name]:2}/{n}  {hits[name] / n:.4f}")
    rescued = [r[0] for r in rows if r[2]["carried"] and not r[2]["bare"]]
    broken = [r[0] for r in rows if r[2]["bare"] and not r[2]["carried"]]
    print(f"\ncarrying the question rescued: {', '.join(rescued) if rescued else 'nothing'}")
    print(f"carrying the question broke:   {', '.join(broken) if broken else 'nothing'}")
    print("\nReviewed pairs; t-08 is marked weak -- see this module's docstring.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

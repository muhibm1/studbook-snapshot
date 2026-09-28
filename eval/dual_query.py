"""Can a follow-up keep the upside of carrying the previous question without the downside?

Two measurements bracket the thread heuristic, and they disagree about it:

    eval/threads.py     genuine follow-ups   bare 2/12 -> carried 8/12   (+0.50, docs/follow-ups.md)
    eval/unrelated.py   a change of subject  bare 29/34 -> carried 20/34 (-0.26)

So concatenating is a bet that the next question follows on, and it pays about two to one. Losing
it means a question that would have worked stops working, which is the worse failure: the user
typed a perfectly good question and got nothing, and nothing in the tab tells them the previous
turn is why.

This measures the arm that does not bet. Both queries are retrieved with, and all four rankings --
vector and full text, bare and carried -- go into the same reciprocal rank fusion the retriever
already uses. A genuine follow-up gets its passage from the carried rankings; a change of subject
gets it from the bare ones; RRF picks whichever put a chunk near the top. Nothing classifies the
question, so there is no classifier to be wrong.

It is nearly free. The reranker sees the same ten candidates it always does, and reranking is the
dominant cost of a query (docs/retrieval-ceiling.md), so the addition is one embedding and two
more index lookups against a ~600-chunk store.

    .venv/Scripts/python.exe -m eval.dual_query
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from pgvector.psycopg import register_vector

from eval.gold import read_jsonl
from eval.metrics import is_hit
from eval.scope import corpus_scope
from eval.unrelated import pair_up
from ingest.embed import Embedder
from retrieve.hybrid import DEFAULT_K, reciprocal_rank_fusion, retrieve_fulltext, retrieve_vector
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
RERANK_CANDIDATES = 10  # production (api/main.py) -- unchanged, so the rerank cost is unchanged
RRF_K = 10


def arms(cur, embedder: Embedder, query: str, scope) -> list[list]:
    """One query's two rankings, deep (DEFAULT_K) rather than truncated, so fusion has something
    to work with below the cut."""
    embedding = embedder.embed([query])[0]
    return [retrieve_vector(cur, embedding, DEFAULT_K, scope),
            retrieve_fulltext(cur, query, DEFAULT_K, scope)]


def candidates(cur, embedder: Embedder, queries: list[str], scope) -> list:
    rankings = [r for q in queries for r in arms(cur, embedder, q, scope)]
    return reciprocal_rank_fusion(rankings, k=RRF_K)[:RERANK_CANDIDATES]


def load_sets() -> dict[str, list[tuple[str, str, str, list[dict]]]]:
    """(id, previous question, question, gold) for each set. `threads` is genuine follow-ups where
    carrying should help; `unrelated` is changes of subject where it should hurt."""
    gold = {r["id"]: r for r in read_jsonl(ROOT / "eval" / "gold.jsonl")}
    threads = [
        (p["id"], gold[p["first"]]["question"], p["follow_up"], gold[p["first"]]["gold"])
        for p in read_jsonl(ROOT / "eval" / "threads.jsonl")
    ]
    dev = sorted([q for q in read_jsonl(ROOT / "eval" / "dev.jsonl") if q["type"] != "unanswerable"],
                 key=lambda q: q["id"])
    unrelated = [(q["id"], previous["question"], q["question"], q["gold"]) for previous, q in pair_up(dev)]
    return {"threads (genuine follow-ups)": threads, "unrelated (a change of subject)": unrelated}


def main() -> int:
    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    sets = load_sets()
    table = {}

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        for name, rows in sets.items():
            print(f"\n{name}, n={len(rows)}")
            hits = {"bare": 0, "carried": 0, "both": 0}
            for rid, previous, question, gold in rows:
                queries = {"bare": [question], "carried": [f"{previous} {question}"],
                           "both": [question, f"{previous} {question}"]}
                found = {}
                for arm, qs in queries.items():
                    top = reranker.rerank(question, candidates(cur, embedder, qs, scope), top_k=TOP_K)
                    found[arm] = any(is_hit(r, gold) for r in top)
                    hits[arm] += found[arm]
                print(f"  {rid:<12} " + "  ".join(
                    f"{a}={'hit ' if found[a] else 'MISS'}" for a in ("bare", "carried", "both")))
            table[name] = (hits, len(rows))

    print(f"\n{'':34} {'bare':>12} {'carried':>12} {'both':>12}")
    for name, (hits, n) in table.items():
        cells = "".join(f"{hits[a]:>4}/{n:<3} {hits[a] / n:.3f}" for a in ("bare", "carried", "both"))
        print(f"{name:<34}{cells}")
    print("\nbare = what a stateless tab sends; carried = what the tab sends today;")
    print("both = every ranking from both queries, fused. Same reranker pool, so the same cost.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""What does carrying the previous question cost when the follow-up is about something else?

`eval/threads.py` measured the heuristic where it was meant to help: a genuine follow-up, whose
subject lives in the previous turn. It said the heuristic works (2/12 -> 8/12). It could not
measure the other side, and `docs/follow-ups.md` says so: a question that changes subject gets the
previous question's words as noise, and every pair in that set was written to be a real follow-up.

This is the other side, and it needs no new labels at all. Take two gold questions about different
parts of the record, treat the first as the previous turn and the second as what the user typed
next, and ask whether the second still retrieves its own gold passage. Both questions are already
in `eval/gold.jsonl` with quotes verified verbatim by `eval/gold.py`, so unlike threads.jsonl
there is nothing here awaiting a human read-through -- the only thing invented is the pairing, and
that is mechanical:

    Sort the dev split's answerable questions by id. Each question's "previous turn" is the next
    question round-robin whose gold documents are disjoint from its own, so the pair is unrelated
    by construction rather than by my judgement, and every question is measured.

Two queries per pair, differing only in the retrieval query -- the cross-encoder re-scores against
the real question in both arms, exactly as the tab does:

    bare       the question alone: what "New thread" sends, and the control
    carried    previous + question: what the tab sends when the user does not start a new thread

Free -- retrieval only, no API calls.

    .venv/Scripts/python.exe -m eval.unrelated
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


def docs_of(question: dict) -> set[tuple[str, str]]:
    return {(g["source"], g["path"]) for g in question["gold"]}


def pair_up(questions: list[dict]) -> list[tuple[dict, dict]]:
    """(previous turn, question), the previous turn being the nearest later question round-robin
    whose gold documents are disjoint. Deterministic, and nothing is hand-picked."""
    n = len(questions)
    pairs = []
    for i, q in enumerate(questions):
        mine = docs_of(q)
        previous = next(
            (questions[(i + 1 + j) % n] for j in range(n - 1) if not docs_of(questions[(i + 1 + j) % n]) & mine),
            None,
        )
        if previous is not None:
            pairs.append((previous, q))
    return pairs


def main() -> int:
    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    dev = [q for q in read_jsonl(ROOT / "eval" / "dev.jsonl") if q["type"] != "unanswerable"]
    pairs = pair_up(sorted(dev, key=lambda q: q["id"]))
    if len(pairs) < len(dev):
        print(f"note: {len(dev) - len(pairs)} question(s) had no disjoint partner and were skipped")

    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    hits = {"bare": 0, "carried": 0}
    broke = []

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        for previous, q in pairs:
            queries = {"bare": q["question"], "carried": f"{previous['question']} {q['question']}"}
            found = {}
            for name, query in queries.items():
                embedding = embedder.embed([query])[0]
                candidates = retrieve_hybrid(cur, query, embedding, top_k=RERANK_CANDIDATES, scope=scope)
                top = reranker.rerank(q["question"], candidates, top_k=TOP_K)
                found[name] = any(is_hit(r, q["gold"]) for r in top)
                hits[name] += found[name]
            if found["bare"] and not found["carried"]:
                broke.append(q["id"])
            flag = "  <- broken by the noise" if found["bare"] and not found["carried"] else ""
            print(f"  {q['id']:<12} bare={'hit ' if found['bare'] else 'MISS'}  "
                  f"carried={'hit ' if found['carried'] else 'MISS'}   after {previous['id']}{flag}")

    n = len(pairs)
    print(f"\n{n} unrelated pair(s), recall@{TOP_K} of the question's own gold passage:")
    for name, label in (("bare", "the question alone (New thread)"),
                        ("carried", "previous + question (no New thread)")):
        print(f"  {label:38} {hits[name]:2}/{n}  {hits[name] / n:.4f}")
    cost = (hits["bare"] - hits["carried"]) / n
    print(f"\ncarrying an unrelated question costs {cost:+.4f} recall@{TOP_K}"
          f" ({len(broke)} broken: {', '.join(broke) if broke else 'none'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

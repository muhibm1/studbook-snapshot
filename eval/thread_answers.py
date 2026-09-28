"""Does a thread give better *answers* to follow-ups, not just better retrieval?

eval/threads.py and eval/rewrite.py measure whether a follow-up's passage is retrieved. That is
the half the thread can be tuned on for free; it is not what the reader sees. This scores the
answer, with the same LLM judge as eval/run_generation.py, against a reference answer written
for each follow-up from its gold passage (eval/threads.jsonl, `reference_answer`, written by
reading each passage in full on 2026-09-18).

Two arms, each exactly what a real tab would do:

    stateless   the follow-up asked on its own: retrieved as typed, answered with no history.
                What a tab without a thread gives, and what Paddock gave before 0.4.2.
    thread      the first question answered first; then the follow-up retrieved with its
                rewrite (retrieve/rewrite.py) and answered with the first turn as prior messages
                and the follow-up note (answer/prompts.py). What Paddock 0.4.4 does.

The judge is told the thread's first question alongside the follow-up, since "was it ever
fixed?" means nothing to a grader on its own; the stateless answer is held to the same reference,
because a reader asking that follow-up wanted the same thing either way.

Costs per pair: up to three Haiku generations, one rewrite call, two Sonnet judge calls.

    .venv/Scripts/python.exe -m eval.thread_answers [--dump PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pgvector.psycopg import register_vector

from answer.generate import generate_answer
from eval.gold import read_jsonl
from eval.judge import JudgeParseError, judge_answer
from eval.metrics import is_hit
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import retrieve_hybrid
from retrieve.rerank import Reranker
from retrieve.rewrite import rewrite_query
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
RERANK_CANDIDATES = 10  # production (api/main.py)
ARMS = ("stateless", "thread")


def judge_question(first: str, follow_up: str) -> str:
    return f"(Earlier in the same conversation the user asked: \"{first}\")\nFollow-up: {follow_up}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", metavar="PATH", help="one JSON record per pair and arm")
    args = parser.parse_args()
    for var in ("STUDBOOK_DATABASE_URL", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            print(f"{var} is not set.", file=sys.stderr)
            return 2

    gold = {r["id"]: r for r in read_jsonl(ROOT / "eval" / "gold.jsonl")}
    pairs = read_jsonl(ROOT / "eval" / "threads.jsonl")
    missing = [p["id"] for p in pairs if "reference_answer" not in p]
    if missing:
        print(f"no reference_answer for: {missing}", file=sys.stderr)
        return 2

    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    totals = {a: {"correctness": 0, "full": 0, "faithful": 0, "refused": 0, "invented": 0, "hit": 0, "n": 0}
              for a in ARMS}
    records, judge_failures = [], []

    def retrieve(cur, query: str) -> list:
        embedding = embedder.embed([query])[0]
        candidates = retrieve_hybrid(cur, query, embedding, top_k=RERANK_CANDIDATES, scope=scope)
        return reranker.rerank(query, candidates, top_k=TOP_K)

    with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        for pair in pairs:
            source = gold[pair["first"]]
            first, follow = source["question"], pair["follow_up"]

            first_answer = generate_answer(first, retrieve(cur, first))
            rewrite = rewrite_query(follow, [first])
            runs = {
                "stateless": (retrieve(cur, follow), []),
                "thread": (retrieve(cur, rewrite.query), [(first, first_answer.text)]),
            }
            line = [f"  {pair['id']:<5}"]
            for arm, (chunks, history) in runs.items():
                answer = generate_answer(follow, chunks, history=history)
                hit = any(is_hit(c, source["gold"]) for c in chunks)
                try:
                    v = judge_answer(judge_question(first, follow), pair["reference_answer"], chunks, answer)
                except JudgeParseError as e:
                    print(f"  judge parse failed for {pair['id']}/{arm}: {e}")
                    judge_failures.append(f"{pair['id']}/{arm}")
                    continue
                t = totals[arm]
                t["n"] += 1
                t["correctness"] += v.correctness
                t["full"] += v.correctness == 2
                t["faithful"] += v.faithful
                t["refused"] += v.is_refusal
                t["invented"] += v.invented_entity
                t["hit"] += hit
                line.append(f"{arm}={v.correctness}{'R' if v.is_refusal else ''}{'!' if v.invented_entity else ''}")
                records.append({"id": pair["id"], "arm": arm, "follow_up": follow,
                                "search": rewrite.query if arm == "thread" else follow,
                                "method": rewrite.method if arm == "thread" else "stateless",
                                "gold_retrieved": hit, "answer": answer.text,
                                "verdict": v.model_dump()})
            print("  ".join(line))

    print(f"\n{len(pairs)} follow-ups, judged against a reference written from the gold passage:")
    print(f"  {'':24}" + "".join(f"{a:>12}" for a in ARMS))
    rows = (("correctness (0-2, mean)", lambda t: f"{t['correctness'] / t['n']:.2f}"),
            ("fully correct", lambda t: f"{t['full']}/{t['n']}"),
            ("faithful", lambda t: f"{t['faithful']}/{t['n']}"),
            ("refused", lambda t: f"{t['refused']}/{t['n']}"),
            ("invented an entity", lambda t: f"{t['invented']}/{t['n']}"),
            ("gold passage retrieved", lambda t: f"{t['hit']}/{t['n']}"))
    for label, fmt in rows:
        print(f"  {label:24}" + "".join(f"{fmt(totals[a]) if totals[a]['n'] else '-':>12}" for a in ARMS))
    if judge_failures:
        print(f"\njudge parse failures (not scored): {', '.join(judge_failures)}")
    print("\nn=12, and every pair shares one first question with its arm: read differences of one or two "
          "as noise. The follow-up pairs were reviewed against their passages (docs/follow-ups.md).")
    if args.dump:
        Path(args.dump).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n", encoding="utf-8")
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

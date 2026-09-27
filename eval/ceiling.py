"""How much recall is even available to the reranker, and where is it lost?

The held-out run (docs/m4-evaluation-and-scale.md) left over-refusal looking like a retrieval
problem: four of six refused questions were refused because no gold passage reached the model.
This asks the question that decides what to fix. Production fuses a pool of candidates and reranks
them down to 5, so a gold chunk outside that pool can never be recovered, whatever the reranker
does. (The pool was 20 when this was written and is 10 now -- this tool is what measured the
change, and `PRODUCTION_POOL` below follows it.)

  - If fusion recall at the pool size is near 1.0, the pool is fine and the reranker (or the cut
    to 5) is the bottleneck.
  - If it is not, the bottleneck is upstream -- the embedding, the lexical query, or chunking --
    and a better reranker cannot help.

Free to run: local embedding and reranker models, and read-only queries. No API calls.

    .venv/Scripts/python.exe -m eval.ceiling --split dev
    .venv/Scripts/python.exe -m eval.ceiling --split dev --arm-k 100 --depths 10 20 50 100

`--arm-k` is how many candidates each arm returns before fusion, and it is a different
configuration, not a deeper look at the same one: rank fusion sees every candidate, so deepening
the arms reorders the fused list all the way to the top. On dev, fusion recall@10 is 0.8824 with
arms at 20 (production) and 0.8529 with arms at 100. Two runs of this tool that differ only in
`--arm-k` are not comparable to each other, and only the production value is comparable to the
numbers in the milestone write-ups.
"""

from __future__ import annotations

import argparse
import os
import sys

from pgvector.psycopg import register_vector

from eval.metrics import is_hit
from eval.run import load_split
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import reciprocal_rank_fusion, retrieve_fulltext, retrieve_vector
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

DEFAULT_DEPTHS = (1, 3, 5, 8, 10, 20)
ARM_K = 20  # DEFAULT_K in retrieve/hybrid.py -- what each arm returns in production
PRODUCTION_POOL = 10  # retrieve/rerank candidates in api/main.py and eval/run_generation.py
FINAL_K = 5


def first_hit_rank(results, gold: list[dict]) -> int | None:
    for i, r in enumerate(results, start=1):
        if is_hit(r, gold):
            return i
    return None


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--depths", type=int, nargs="+", default=list(DEFAULT_DEPTHS))
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--arm-k", type=int, default=ARM_K,
                        help=f"how many candidates each arm returns before fusion (default {ARM_K}, "
                             "the production value). This is NOT a reporting depth: rank fusion "
                             "sees every candidate, so deepening the arms changes the fused "
                             "ordering all the way to the top. Measured on dev: fusion recall@10 "
                             "is 0.8824 with arms at 20 and 0.8529 with arms at 100. Leave it at "
                             "the production value unless the arm depth is what you are ablating.")
    args = parser.parse_args(argv)

    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    questions = load_split(args.split)  # answerable only: a trap has no gold chunk to find
    arm_k = args.arm_k
    depths = sorted(d for d in set(args.depths) if d <= arm_k)
    dropped = sorted(d for d in set(args.depths) if d > arm_k)
    if dropped:
        # Reporting a depth past the arm depth would silently measure a different pipeline than
        # the one being described, which is exactly the confusion this flag exists to prevent.
        print(f"ignoring depth(s) {dropped} deeper than --arm-k {arm_k}; "
              f"pass --arm-k {max(dropped)} to measure those, and read it as a different "
              f"configuration, not a deeper look at this one.", file=sys.stderr)
    if not depths:
        print("no reporting depth is within --arm-k", file=sys.stderr)
        return 2

    print("loading models (embedder, reranker) ...")
    embedder = Embedder()
    reranker = Reranker()

    ranks: dict[str, list[int | None]] = {"vector": [], "fulltext": [], "fusion": []}
    reranked_at: dict[int, list[int | None]] = {d: [] for d in depths}
    # The rank the gold chunk lands at in the FULL reranked ordering, not just the top 5 -- that is
    # what says whether passing the generator more passages would recover what the cut throws away.
    reranked_full: dict[int, list[int | None]] = {d: [] for d in depths}
    never_found: list[str] = []

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        scope = corpus_scope()
        for q in questions:
            embedding = embedder.embed([q["question"]])[0]
            vector = retrieve_vector(cur, embedding, k=arm_k, scope=scope)
            fulltext = retrieve_fulltext(cur, q["question"], k=arm_k, scope=scope)
            fused = reciprocal_rank_fusion([vector, fulltext], k=args.rrf_k)

            ranks["vector"].append(first_hit_rank(vector, q["gold"]))
            ranks["fulltext"].append(first_hit_rank(fulltext, q["gold"]))
            ranks["fusion"].append(first_hit_rank(fused, q["gold"]))
            if ranks["fusion"][-1] is None:
                never_found.append(q["id"])

            # Does a bigger candidate pool help the reranker, or does it just add distractors?
            for depth in depths:
                ordered = reranker.rerank(q["question"], fused[:depth], top_k=depth)
                reranked_full[depth].append(first_hit_rank(ordered, q["gold"]))
                reranked_at[depth].append(first_hit_rank(ordered[:FINAL_K], q["gold"]))

    n = len(questions)

    def recall(rank_list: list[int | None], at: int) -> float:
        return sum(1 for r in rank_list if r is not None and r <= at) / (n or 1)

    print(f"\n{args.split}: {n} answerable question(s), rrf_k={args.rrf_k}, "
          f"each arm returning {arm_k} candidates\n")
    header = "  ".join(f"@{d:<5}" for d in depths)
    print(f"{'stage':22} {header}")
    for stage in ("vector", "fulltext", "fusion"):
        row = "  ".join(f"{recall(ranks[stage], d):<6.4f}" for d in depths)
        print(f"  {stage:20} {row}")

    print(f"\nreranked from the production pool of {PRODUCTION_POOL}, by how many passages "
          f"survive to the generator:")
    if PRODUCTION_POOL in reranked_at:
        pool_ceiling = recall(ranks["fusion"], PRODUCTION_POOL)
        for cut in (1, 3, 5, 8, 10, 15, PRODUCTION_POOL):
            if cut > PRODUCTION_POOL:
                continue
            value = recall(reranked_full[PRODUCTION_POOL], cut)
            mark = "  <- production" if cut == FINAL_K else ""
            print(f"  top {cut:<18} {value:<10.4f} ({pool_ceiling - value:+.4f} vs the pool's "
                  f"{pool_ceiling:.4f} ceiling){mark}")

    print(f"\nreranked to top {FINAL_K}, by how many candidates it was given:")
    print(f"{'candidate pool':22} {'recall@5':<10} {'change vs pool of ' + str(PRODUCTION_POOL):<10}")
    base = recall(reranked_at.get(PRODUCTION_POOL, []), FINAL_K) if PRODUCTION_POOL in reranked_at else None
    for depth in depths:
        value = recall(reranked_at[depth], FINAL_K)
        delta = "" if base is None else f"{value - base:+.4f}"
        mark = "  <- production" if depth == PRODUCTION_POOL else ""
        print(f"  {depth:<20} {value:<10.4f} {delta:<10}{mark}")

    ceiling = recall(ranks["fusion"], PRODUCTION_POOL)
    achieved = recall(reranked_at.get(PRODUCTION_POOL, []), FINAL_K) if PRODUCTION_POOL in reranked_at else 0.0
    print(f"\nceiling: {ceiling:.4f} of questions have a gold chunk inside the production pool of "
          f"{PRODUCTION_POOL};\n         {achieved:.4f} survive the cut to {FINAL_K}. "
          f"The reranker is losing {ceiling - achieved:.4f}, upstream is losing {1 - ceiling:.4f}.")

    if never_found:
        print(f"\nno gold chunk in either arm's top {arm_k} ({len(never_found)}): "
              f"{', '.join(never_found)}")
        print("  These are embedding or lexical failures. No reranker and no larger pool reaches them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

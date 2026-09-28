"""Where does a multi-document question lose one of its documents?

coverage@5 (eval/metrics.py) says whether every gold document reached the generator. For a question
whose answer spans documents, this says where each missing one was lost, which decides the fix:

    not in either arm's top ARM_K     retrieval never saw it: the query does not reach that
                                      document at all, and only a different query can (decomposing
                                      the question into one query per part)
    fused, but below the rerank pool  fusion ranked it too low for the reranker to see it
    in the pool, cut by the reranker  the reranker saw it and put it below 5: the pool is fine, the
                                      single rerank ranking against the whole question is not
    in the top 5                      covered

Only questions with gold quotes in more than one document are shown. Free: local models,
read-only queries, no API calls.

    .venv/Scripts/python.exe -m eval.coverage --split dev
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

from pgvector.psycopg import register_vector

from eval.ceiling import ARM_K, FINAL_K, PRODUCTION_POOL, first_hit_rank
from eval.run import load_split
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import reciprocal_rank_fusion, retrieve_fulltext, retrieve_vector
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

RRF_K = 10


def documents(gold: list[dict]) -> dict[str, list[dict]]:
    by_document: dict[str, list[dict]] = defaultdict(list)
    for g in gold:
        by_document[f"{g['source']}:{g['path']}"].append(g)
    return dict(by_document)


def where_lost(fused_rank: int | None, reranked_rank: int | None) -> str:
    if reranked_rank is not None and reranked_rank <= FINAL_K:
        return "covered"
    if fused_rank is None:
        return "never retrieved"
    if fused_rank > PRODUCTION_POOL:
        return "below the rerank pool"
    return "cut by the reranker"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev",), default="dev",
                        help="dev only: the held-out split is scored once, and this is a tuning tool")
    args = parser.parse_args(argv)
    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    questions = [q for q in load_split(args.split) if len(documents(q.get("gold", []))) > 1]
    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    tally: dict[str, int] = defaultdict(int)

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)
        for q in questions:
            embedding = embedder.embed([q["question"]])[0]
            fused = reciprocal_rank_fusion(
                [retrieve_vector(cur, embedding, ARM_K, scope), retrieve_fulltext(cur, q["question"], ARM_K, scope)],
                k=RRF_K,
            )
            reranked = reranker.rerank(q["question"], fused[:PRODUCTION_POOL], top_k=PRODUCTION_POOL)
            print(f"\n{q['id']} ({q['type']}): {q['question']}")
            for doc, quotes in documents(q["gold"]).items():
                f, r = first_hit_rank(fused, quotes), first_hit_rank(reranked, quotes)
                verdict = where_lost(f, r)
                tally[verdict] += 1
                print(f"    fused #{f or '-':<3} reranked #{r or '-':<3} {verdict:22} {doc.split(':', 1)[1][-70:]}")

    total = sum(tally.values())
    print(f"\n{len(questions)} multi-document question(s), {total} gold documents:")
    for verdict in ("covered", "cut by the reranker", "below the rerank pool", "never retrieved"):
        print(f"  {verdict:24} {tally[verdict]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

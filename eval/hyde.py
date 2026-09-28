"""Does searching with a hypothetical passage lift coverage, and what does it cost elsewhere?

eval/hypothetical.py over every answerable dev question. Arms fixed before the first run, all
reranked against the question as typed and cut to 5, exactly as production:

    baseline      vector(question) + full text(question), fused -- production
    +vector       adds vector(passage) as a third ranking
    +both         adds vector(passage) and full text(passage): four rankings
    replace       vector(passage) instead of vector(question), with full text(question): the
                  original HyDE formulation

Reported: coverage@5 and recall@5 by category, every question whose coverage an arm changes, and
version-14's passage in full, since that is the question the idea was built for. 34 Haiku calls.

    .venv/Scripts/python.exe -m eval.hyde [--dump PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

from pgvector.psycopg import register_vector

from eval.hypothetical import hypothetical_passage
from eval.metrics import covers_all_documents, is_hit
from eval.run import load_split
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import DEFAULT_K, reciprocal_rank_fusion, retrieve_fulltext, retrieve_vector
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

TOP_K = 5
RERANK_CANDIDATES = 10
RRF_K = 10
ARMS = ("baseline", "+vector", "+both", "replace")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", metavar="PATH", help="write each question's passage and per-arm coverage")
    args = parser.parse_args()
    for var in ("STUDBOOK_DATABASE_URL", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            print(f"{var} is not set.", file=sys.stderr)
            return 2
    questions = [q for q in load_split("dev") if q["type"] != "unanswerable"]
    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    cov = {a: Counter() for a in ARMS}
    rec = {a: Counter() for a in ARMS}
    n, records = Counter(), []

    with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)
        for q in questions:
            question = q["question"]
            passage = hypothetical_passage(question)
            q_vec, p_vec = embedder.embed([question, passage or question])
            v_q = retrieve_vector(cur, q_vec, DEFAULT_K, scope)
            f_q = retrieve_fulltext(cur, question, DEFAULT_K, scope)
            v_p = retrieve_vector(cur, p_vec, DEFAULT_K, scope) if passage else []
            f_p = retrieve_fulltext(cur, passage, DEFAULT_K, scope) if passage else []
            rankings = {
                "baseline": [v_q, f_q],
                "+vector": [v_q, f_q, v_p],
                "+both": [v_q, f_q, v_p, f_p],
                "replace": [v_p or v_q, f_q],
            }
            n["all"] += 1
            n[q["type"]] += 1
            covered = {}
            for arm, rs in rankings.items():
                candidates = reciprocal_rank_fusion([r for r in rs if r], k=RRF_K)[:RERANK_CANDIDATES]
                top = reranker.rerank(question, candidates, top_k=TOP_K)
                c, h = covers_all_documents(top, q["gold"]), any(is_hit(x, q["gold"]) for x in top)
                covered[arm] = c
                for g in ("all", q["type"]):
                    cov[arm][g] += c
                    rec[arm][g] += h
            changes = [f"{a} {'GAINED' if covered[a] else 'LOST'}" for a in ARMS[1:] if covered[a] != covered["baseline"]]
            if changes:
                print(f"  {q['id']:<11} {q['type']:<12} {', '.join(changes)}")
            if q["id"] == "version-14":
                print("  version-14 passage:\n    " + passage.replace("\n", "\n    "))
            records.append({"id": q["id"], "type": q["type"], "passage": passage, "covered": covered})

    print(f"\n{'coverage@5':24}" + "".join(f"{a:>11}" for a in ARMS))
    for g in ["all", *sorted(k for k in n if k != "all")]:
        print(f"  {g:<12} (n={n[g]:2})    " + "".join(f"{cov[a][g]:>7}/{n[g]:<3}" for a in ARMS))
    print(f"\n{'recall@5':24}" + "".join(f"{a:>11}" for a in ARMS))
    print(f"  {'all':<12} (n={n['all']:2})    " + "".join(f"{rec[a]['all']:>7}/{n['all']:<3}" for a in ARMS))
    if args.dump:
        Path(args.dump).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

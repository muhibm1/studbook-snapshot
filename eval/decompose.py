"""Does splitting a multi-part question into one lookup per part cover its documents?

eval/decomposition.py (measured and rejected; see docs/cross-doc.md), over every answerable question in the dev split, three arms:

    whole         production today: one ranking against the question
    parts         one ranking per part, taken round-robin (a question not split = whole)
    whole+parts   the whole question's ranking first in the round-robin, then the parts: keeps a
                  passage that answers the question jointly, at the cost of a slot

Reported: coverage@5 (every gold document in the top 5, eval/metrics.py) and recall@5, overall and
for cross-doc. Every split is printed, and the guard's rejections with it, because the questions
it does not split matter as much as the ones it does: a split of a single-topic question costs a
rerank per part and can only move passages around.

One Haiku call per question (34). Dev only: the held-out split is scored once.

    .venv/Scripts/python.exe -m eval.decompose [--dump PATH]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pgvector.psycopg import register_vector

from eval.metrics import covers_all_documents, is_hit
from eval.run import load_split
from eval.scope import corpus_scope
from ingest.embed import Embedder
from eval.decomposition import decompose, interleave
from retrieve.hybrid import retrieve_hybrid
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

TOP_K = 5
RERANK_CANDIDATES = 10  # production
ARMS = ("whole", "parts", "whole+parts")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dump", metavar="PATH", help="write the decompositions and per-arm results as JSON")
    args = parser.parse_args()
    for var in ("STUDBOOK_DATABASE_URL", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            print(f"{var} is not set.", file=sys.stderr)
            return 2

    questions = [q for q in load_split("dev") if q["type"] != "unanswerable"]
    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    cov: dict[str, dict[str, list[int]]] = {a: defaultdict(list) for a in ARMS}
    rec: dict[str, dict[str, list[int]]] = {a: defaultdict(list) for a in ARMS}
    methods, records = Counter(), []

    def ranked(cur, query: str) -> list:
        embedding = embedder.embed([query])[0]
        candidates = retrieve_hybrid(cur, query, embedding, top_k=RERANK_CANDIDATES, scope=scope)
        return reranker.rerank(query, candidates, top_k=TOP_K)

    with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)
        for q in questions:
            d = decompose(q["question"])
            methods[d.method] += 1
            whole = ranked(cur, q["question"])
            per_part = [ranked(cur, p) for p in d.parts] if d.method == "split" else [whole]
            results = {
                "whole": whole,
                "parts": interleave(per_part, TOP_K),
                "whole+parts": interleave([whole, *per_part], TOP_K) if d.method == "split" else whole,
            }
            marks = []
            for arm, top in results.items():
                c, r = covers_all_documents(top, q["gold"]), any(is_hit(x, q["gold"]) for x in top)
                for group in ("all", q["type"]):
                    cov[arm][group].append(c)
                    rec[arm][group].append(r)
                marks.append(f"{arm}={'cov' if c else ('hit' if r else '---')}")
            print(f"  {q['id']:<11} {q['type']:<12} {'  '.join(marks)}  [{d.method}]")
            if d.method == "split":
                for p in d.parts:
                    print(f"      part: {p}")
            for p in d.rejected:
                print(f"      rejected: {p}")
            records.append({"id": q["id"], "type": q["type"], "method": d.method, "parts": d.parts,
                            "rejected": d.rejected,
                            "covered": {a: covers_all_documents(t, q["gold"]) for a, t in results.items()}})

    def fmt(values: list[int]) -> str:
        return f"{sum(values):>2}/{len(values):<2} {sum(values) / len(values):.3f}" if values else "-"

    print(f"\n{'':22}" + "".join(f"{a:>16}" for a in ARMS))
    for group in ("all", "cross-doc"):
        print(f"coverage@5 {group:<11}" + "".join(f"{fmt(cov[a][group]):>16}" for a in ARMS))
        print(f"recall@5   {group:<11}" + "".join(f"{fmt(rec[a][group]):>16}" for a in ARMS))
    print("\ndecompositions: " + ", ".join(f"{m} {n}" for m, n in sorted(methods.items())))
    if args.dump:
        Path(args.dump).write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.dump}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

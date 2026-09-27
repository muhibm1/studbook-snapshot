"""Does a synonym-expanded third lexical ranking lift coverage, and what does it cost elsewhere?

eval/expansion.py (measured and rejected; docs/cross-doc.md) over every answerable dev question. Arms, all reranked against the question as
typed and cut to 5, exactly as production:

    baseline      vector + full text, fused (production)
    vocab@0.70    + a full-text ranking over each query word's 3 nearest corpus words, cosine >= 0.70
    vocab@0.80    the same at >= 0.80
    model         + a full-text ranking over up to twelve words Haiku proposes

The settings (3 neighbours, thresholds 0.70 and 0.80) were fixed before the first run, so they are
not tuned on this split; tuning them on 34 questions would be fitting dev.

Reported: coverage@5 and recall@5 by category, since an expansion that helps one cross-doc question
by adding noise to twenty others is not a win. Every question whose coverage an arm changes is
printed with the terms that arm added. Costs 34 Haiku calls for the model arm; the rest is local.

    .venv/Scripts/python.exe -m eval.expand
"""

from __future__ import annotations

import os
import sys
import time
from collections import Counter

from pgvector.psycopg import register_vector

from eval.metrics import covers_all_documents, is_hit
from eval.run import load_split
from eval.scope import corpus_scope
from ingest.embed import Embedder
from eval.expansion import VocabularyIndex, expand_model, expand_vocabulary
from retrieve.hybrid import DEFAULT_K, reciprocal_rank_fusion, retrieve_fulltext, retrieve_vector
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

TOP_K = 5
RERANK_CANDIDATES = 10
RRF_K = 10
NEIGHBOURS = 3
THRESHOLDS = (0.70, 0.80)


def main() -> int:
    for var in ("STUDBOOK_DATABASE_URL", "ANTHROPIC_API_KEY"):
        if not os.environ.get(var):
            print(f"{var} is not set.", file=sys.stderr)
            return 2
    questions = [q for q in load_split("dev") if q["type"] != "unanswerable"]
    print("loading models (embedder, reranker) ...")
    embedder, reranker = Embedder(), Reranker()
    scope = corpus_scope()
    arms = ["baseline", *(f"vocab@{t:.2f}" for t in THRESHOLDS), "model"]
    cov = {a: Counter() for a in arms}
    rec = {a: Counter() for a in arms}
    n = Counter()

    with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)
        cur.execute("select c.body from studbook.chunks c join studbook.documents d on d.id = c.document_id "
                    "where d.repo = any(%s)", (list(scope.repos),))
        started = time.time()
        index = VocabularyIndex.build(embedder, [row[0] for row in cur.fetchall()])
        print(f"vocabulary: {len(index.words)} corpus words embedded in {time.time() - started:.0f}s")

        for q in questions:
            question = q["question"]
            embedding = embedder.embed([question])[0]
            base_rankings = [retrieve_vector(cur, embedding, DEFAULT_K, scope),
                             retrieve_fulltext(cur, question, DEFAULT_K, scope)]
            terms = {f"vocab@{t:.2f}": expand_vocabulary(question, index, embedder, NEIGHBOURS, t) for t in THRESHOLDS}
            terms["model"] = expand_model(question)
            tops = {}
            for arm in arms:
                rankings = list(base_rankings)
                if arm != "baseline" and terms[arm]:
                    rankings.append(retrieve_fulltext(cur, " ".join(terms[arm]), DEFAULT_K, scope))
                candidates = reciprocal_rank_fusion(rankings, k=RRF_K)[:RERANK_CANDIDATES]
                tops[arm] = reranker.rerank(question, candidates, top_k=TOP_K)
            n["all"] += 1
            n[q["type"]] += 1
            base_cov = covers_all_documents(tops["baseline"], q["gold"])
            for arm, top in tops.items():
                c = covers_all_documents(top, q["gold"])
                r = any(is_hit(x, q["gold"]) for x in top)
                for g in ("all", q["type"]):
                    cov[arm][g] += c
                    rec[arm][g] += r
                if arm != "baseline" and c != base_cov:
                    print(f"  {q['id']:<11} {arm:<11} {'GAINED' if c else 'LOST  '} coverage   "
                          f"terms: {' '.join(terms[arm]) or '(none)'}")
            if q["id"] == "version-14":
                for arm in arms[1:]:
                    print(f"  version-14  {arm:<11} terms: {' '.join(terms[arm]) or '(none)'}")

    print(f"\n{'coverage@5':24}" + "".join(f"{a:>13}" for a in arms))
    for g in ["all", *sorted(k for k in n if k != "all")]:
        print(f"  {g:<12} (n={n[g]:2})    " + "".join(f"{cov[a][g]:>9}/{n[g]:<3}" for a in arms))
    print(f"\n{'recall@5':24}" + "".join(f"{a:>13}" for a in arms))
    print(f"  {'all':<12} (n={n['all']:2})    " + "".join(f"{rec[a]['all']:>9}/{n['all']:<3}" for a in arms))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

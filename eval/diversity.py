"""Does capping how many chunks one document may place in the top 5 lift coverage? MEASURED, NO.

The two cross-doc documents lost just below the cut (#6, #7; eval/coverage.py) would come into view
if the top 5 were crowded with several chunks of one document. It mostly is not (about one duplicate
per top 5 on dev). A cap of 2 changed nothing; a cap of 1 lowered cross-doc coverage from 3/7 to
1/7, because a document's second chunk is often where the detail is (docs/cross-doc.md). Free.

    .venv/Scripts/python.exe -m eval.diversity
"""
import os
from collections import Counter
from pgvector.psycopg import register_vector
from eval.metrics import covers_all_documents, is_hit
from eval.run import load_split
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import retrieve_hybrid
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

def capped(ranked, cap, k=5):
    out, per = [], Counter()
    for r in ranked:
        if per[r.document_id] < cap:
            out.append(r); per[r.document_id] += 1
        if len(out) == k: break
    return out

qs = [q for q in load_split("dev") if q["type"] != "unanswerable"]
print("loading models (embedder, reranker) ...")
emb, rr, scope = Embedder(), Reranker(), corpus_scope()
arms = {"none": None, "cap2": 2, "cap1": 1}
cov = {a: Counter() for a in arms}; rec = {a: Counter() for a in arms}; dup = []
with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
    prepare_session(cur); register_vector(conn)
    for q in qs:
        e = emb.embed([q["question"]])[0]
        ranked = rr.rerank(q["question"], retrieve_hybrid(cur, q["question"], e, top_k=10, scope=scope), top_k=10)
        top5 = ranked[:5]
        dup.append(5 - len({r.document_id for r in top5}))
        for a, cap in arms.items():
            top = top5 if cap is None else capped(ranked, cap)
            c, h = covers_all_documents(top, q["gold"]), any(is_hit(r, q["gold"]) for r in top)
            for g in ("all", q["type"]):
                cov[a][g] += c; rec[a][g] += h
        if q["type"] == "cross-doc":
            print(q["id"], "top5 docs:", [r.document_id[-30:] for r in top5])
n = Counter(q["type"] for q in qs); n["all"] = len(qs)
print("mean duplicate chunks in top 5:", round(sum(dup) / len(dup), 2))
for a in arms:
    print(f"{a:5} coverage all {cov[a]['all']}/{n['all']}  cross-doc {cov[a]['cross-doc']}/{n['cross-doc']}  | recall all {rec[a]['all']}/{n['all']}  "
          + " ".join(f"{t}:{cov[a][t]}/{n[t]}" for t in sorted(n) if t not in ('all',)))

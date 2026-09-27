"""Python side of the parity gate (docs/paddock-gate.md): what the JS port has to reproduce.

For every answerable dev question: the query embedding (fp32, normalised), the fused top-20 chunk
ids in order, the reranked top-10 in order with cross-encoder logits, and the gold quotes. The JS
side embeds the same questions with the ONNX model, runs the same SQL, and is held to the same
0.8235 / 0.6485 / 0.1824 the Python pipeline publishes.

    .venv/Scripts/python.exe gate/dump_python_side.py   # writes gate/python_side.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from pgvector.psycopg import register_vector

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.run import load_split  # noqa: E402
from ingest.embed import Embedder  # noqa: E402
from retrieve.hybrid import reciprocal_rank_fusion, retrieve_fulltext, retrieve_vector  # noqa: E402
from retrieve.rerank import Reranker  # noqa: E402
from eval.scope import corpus_scope  # noqa: E402
from store.connect import connect, prepare_session  # noqa: E402

questions = load_split("dev")
embedder = Embedder()
reranker = Reranker()
out = []
with connect(os.environ["STUDBOOK_DATABASE_URL"], autocommit=True) as conn, conn.cursor() as cur:
    prepare_session(cur)
    register_vector(conn)
    scope = corpus_scope()
    for q in questions:
        embedding = embedder.embed([q["question"]])[0]
        vector = retrieve_vector(cur, embedding, k=20, scope=scope)
        fulltext = retrieve_fulltext(cur, q["question"], k=20, scope=scope)
        fused = reciprocal_rank_fusion([vector, fulltext], k=10)[:20]
        scores = reranker.model.predict([(q["question"], r.body) for r in fused])
        order = sorted(range(len(fused)), key=lambda i: -scores[i])
        out.append({
            "id": q["id"], "type": q["type"], "question": q["question"],
            "gold_quotes": [g["quote"] for g in q["gold"]],
            "embedding": embedding,
            "vector_ids": [r.chunk_id for r in vector],
            "fulltext_ids": [r.chunk_id for r in fulltext],
            "fused_ids": [r.chunk_id for r in fused],
            "reranked": [{"chunk_id": fused[i].chunk_id, "logit": float(scores[i])} for i in order[:10]],
        })
        print(f"  {q['id']}")

path = ROOT / "gate" / "python_side.json"
path.write_text(json.dumps(out), encoding="utf-8")
print(f"wrote {path} ({len(out)} questions)")

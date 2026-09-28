"""Diff the model stack's outputs across environments, before any torch / transformers /
sentence-transformers bump (docs/ci.md). Run it with the old environment's python and then the
new one's, writing two files, then compare:

    old/.venv/Scripts/python.exe gate/numerics.py old.json
    new/.venv/Scripts/python.exe gate/numerics.py new.json
    python gate/numerics.py --compare old.json new.json

The 2026-09-18 upgrade measured a maximum embedding difference of 0.0 and a maximum rerank-score
difference of 1.6e-10. Anything above ~1e-6 would change stored embeddings' neighbours and wants
the retrieval eval and both Paddock gates re-run before it ships.
"""
import json
import sys

if sys.argv[1] == "--compare":
    a, b = (json.load(open(p)) for p in sys.argv[2:4])
    de = max(abs(x - y) for u, v in zip(a["embed"], b["embed"]) for x, y in zip(u, v))
    dr = max(abs(x - y) for x, y in zip(a["rerank"], b["rerank"]))
    print(f"max |embedding diff| = {de:.2e}   max |rerank score diff| = {dr:.2e}")
    raise SystemExit(0)

sys.path.insert(0, __file__.rsplit("gate", 1)[0])
from ingest.embed import Embedder  # noqa: E402
from retrieve.rerank import Reranker  # noqa: E402

texts = ["Why did control state move into the git common dir?", "How is the /health endpoint's uptime measured?",
         "npm test: # tests 47, # pass 47", "What did the review packet decide about R7's guard?"]
embed = Embedder().embed(texts)
rerank = Reranker().model.predict([(texts[0], t) for t in texts]).tolist()
json.dump({"embed": embed, "rerank": rerank}, open(sys.argv[1], "w"))
print(f"wrote {sys.argv[1]}")

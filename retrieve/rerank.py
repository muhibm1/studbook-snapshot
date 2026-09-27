"""Cross-encoder reranking: re-scores a candidate list against the query directly (query and
passage attend to each other jointly), rather than comparing two independently-computed vectors
the way bi-encoder retrieval (embed.py) and RRF ranking do. M2 found weak precision specifically
on evidence/factual questions -- a single fact buried in an otherwise-relevant passage -- which is
exactly the shape a cross-encoder is good at (docs/m2-retrieval-and-eval.md's by-type table).
"""

from __future__ import annotations

import os

from retrieve.hybrid import Result

RERANKER_MODEL = os.environ.get("STUDBOOK_RERANKER_MODEL", "BAAI/bge-reranker-base")
# Pinned for the same reasons as ingest/embed.py's REVISIONS: reproducible scores, and no pickle
# checkpoint can arrive from upstream (PYSEC-2026-2286, docs/ci.md).
REVISIONS = {"BAAI/bge-reranker-base": "2cfc18c9415c912f9d8155881c133215df768a70"}


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        # Imported lazily, same reasoning as ingest/embed.py's Embedder: importing sentence-
        # transformers (and torch) is slow and, on this project's history, fragile.
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, revision=REVISIONS.get(model_name),
                                  model_kwargs={"use_safetensors": True})

    def rerank(self, query: str, results: list[Result], top_k: int) -> list[Result]:
        if not results:
            return []
        scores = self.model.predict([(query, r.body) for r in results])
        order = sorted(range(len(results)), key=lambda i: -scores[i])
        return [results[i] for i in order[:top_k]]

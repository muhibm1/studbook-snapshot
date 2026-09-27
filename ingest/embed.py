"""Wraps the local bge-base-en-v1.5 model (see requirements.txt for why torch/sentence-transformers
are pinned the way they are). One instance per process -- loading the model is the slow part."""

from __future__ import annotations

import os

MODEL_NAME = os.environ.get("STUDBOOK_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
DIMENSIONS = 768  # must match store/migrations/0001_init.sql's vector(768) column
# The exact upstream revision every stored embedding was made with. Pinned so the model cannot change
# under the store without a code change, and so a changed upstream repository cannot swap in a
# pickle checkpoint (torch advisory PYSEC-2026-2286, docs/ci.md). A model named by
# STUDBOOK_EMBED_MODEL has no pinned revision; that override is for experiments.
REVISIONS = {"BAAI/bge-base-en-v1.5": "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"}


class Embedder:
    def __init__(self, model_name: str = MODEL_NAME) -> None:
        # Imported lazily: importing sentence_transformers (and therefore torch) is the slow, and
        # on this project's history, fragile part (requirements.txt) -- code that only chunks or
        # parses documents shouldn't pay that cost or that risk just by importing this module.
        from sentence_transformers import SentenceTransformer

        # use_safetensors: refuse to load weights from a pickle file at all, rather than trusting
        # torch's weights_only unpickler, which the advisory above shows can be escaped.
        self.model = SentenceTransformer(model_name, revision=REVISIONS.get(model_name),
                                         model_kwargs={"use_safetensors": True})

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Normalised embeddings, one per input text, in the same order. bge-base-en-v1.5 is
        trained for cosine similarity, hence normalize_embeddings=True -- the schema's HNSW index
        uses vector_cosine_ops, and an unnormalised vector would make that distance meaningless."""
        if not texts:
            return []
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

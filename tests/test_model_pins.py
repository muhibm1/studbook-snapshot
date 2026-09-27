"""The models load from an exact revision and from safetensors only.

torch's weights_only unpickler can be escaped by a crafted checkpoint (PYSEC-2026-2286), and the
development machine cannot run the torch release that fixes it (requirements.txt). Refusing pickle
weights outright, at a pinned revision, makes that advisory unreachable here by construction rather
than by luck (docs/ci.md). These tests fail if either guard is dropped."""

import sys
import types
import unittest
from unittest import mock


class Recorder:
    calls: list = []

    def __init__(self, *args, **kwargs) -> None:
        Recorder.calls.append((args, kwargs))


class ModelPinTest(unittest.TestCase):
    def setUp(self) -> None:
        Recorder.calls = []
        fake = types.ModuleType("sentence_transformers")
        fake.SentenceTransformer = Recorder
        fake.CrossEncoder = Recorder
        self.patch = mock.patch.dict(sys.modules, {"sentence_transformers": fake})
        self.patch.start()

    def tearDown(self) -> None:
        self.patch.stop()

    def test_the_embedder_loads_a_pinned_revision_from_safetensors(self) -> None:
        from ingest.embed import REVISIONS, Embedder
        Embedder()
        args, kwargs = Recorder.calls[-1]
        self.assertEqual(kwargs["revision"], REVISIONS["BAAI/bge-base-en-v1.5"])
        self.assertEqual(len(kwargs["revision"]), 40)
        self.assertIs(kwargs["model_kwargs"]["use_safetensors"], True)

    def test_the_reranker_loads_a_pinned_revision_from_safetensors(self) -> None:
        from retrieve.rerank import REVISIONS, Reranker
        Reranker()
        args, kwargs = Recorder.calls[-1]
        self.assertEqual(kwargs["revision"], REVISIONS["BAAI/bge-reranker-base"])
        self.assertIs(kwargs["model_kwargs"]["use_safetensors"], True)


if __name__ == "__main__":
    unittest.main()

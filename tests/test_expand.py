import inspect
import unittest
from types import SimpleNamespace

import anthropic
import numpy as np

from eval.expansion import VocabularyIndex, content_words, expand_model, expand_vocabulary, vocabulary


class FakeEmbedder:
    """2-d unit vectors: 'size' and 'count' point the same way, 'banana' does not."""

    TABLE = {"size": (1.0, 0.0), "count": (0.98, 0.2), "sizes": (1.0, 0.01), "banana": (0.0, 1.0),
             "suite": (0.0, 1.0), "tests": (0.1, 0.99)}

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.asarray(self.TABLE.get(t, (0.7, 0.7)), dtype=float)
            out.append(list(v / np.linalg.norm(v)))
        return out


class VocabularyTest(unittest.TestCase):
    def test_keeps_words_used_at_least_twice_and_drops_function_words(self) -> None:
        self.assertEqual(vocabulary(["the count was 47", "a count of tests", "tests ran"]), ["count", "tests"])

    def test_content_words_skip_short_and_function_words(self) -> None:
        self.assertEqual(content_words("How did the size of it compare?"), ["size", "compare"])


class ExpandVocabularyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.embedder = FakeEmbedder()
        words = ["banana", "count", "sizes", "tests"]
        self.index = VocabularyIndex(words, np.asarray(self.embedder.embed(words)))

    def test_a_near_corpus_word_is_added(self) -> None:
        self.assertIn("count", expand_vocabulary("size", self.index, self.embedder, k=3, threshold=0.9))

    def test_an_inflection_of_the_query_word_is_not_an_expansion(self) -> None:
        self.assertNotIn("sizes", expand_vocabulary("size", self.index, self.embedder, k=3, threshold=0.9))

    def test_nothing_below_the_threshold_is_added(self) -> None:
        self.assertNotIn("banana", expand_vocabulary("size", self.index, self.embedder, k=3, threshold=0.9))

    def test_a_word_the_question_already_has_is_not_added_again(self) -> None:
        terms = expand_vocabulary("size count", self.index, self.embedder, k=3, threshold=0.9)
        self.assertNotIn("count", terms)


class FakeClient:
    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


class ExpandModelTest(unittest.TestCase):
    def test_drops_the_questions_own_words_and_caps_at_twelve(self) -> None:
        extra = "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima mike november"
        reply = "count sizes number " + extra
        terms = expand_model("What was the size?", client=FakeClient(reply))
        self.assertEqual(terms[:2], ["count", "number"])
        self.assertEqual(len(terms), 12)

    def test_a_failed_call_means_no_expansion_not_a_failed_question(self) -> None:
        import httpx
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        self.assertEqual(expand_model("q?", client=FakeClient(error=error)), [])

    def test_the_request_binds_to_the_real_sdk_signature(self) -> None:
        client = FakeClient("count")
        expand_model("q?", client=client)
        inspect.signature(anthropic.Anthropic(api_key="not-used").messages.create).bind(**client.calls[0])


if __name__ == "__main__":
    unittest.main()

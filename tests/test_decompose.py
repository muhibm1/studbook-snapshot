import inspect
import unittest
from dataclasses import dataclass
from types import SimpleNamespace

import anthropic
import httpx

from eval.decomposition import decompose, interleave, parse_parts

Q = "How did the size of the automated test suite for /version compare to what /health shipped with?"


class FakeClient:
    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self.reply, self.error, self.calls = reply, error, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


@dataclass(frozen=True)
class R:
    chunk_id: str


class ParsePartsTest(unittest.TestCase):
    def test_strips_list_markers_and_blank_lines(self) -> None:
        self.assertEqual(parse_parts("1. first?\n\n- second?\n"), ["first?", "second?"])

    def test_caps_at_three_and_drops_duplicates(self) -> None:
        self.assertEqual(parse_parts("a?\na?\nb?\nc?\nd?"), ["a?", "b?", "c?"])


class DecomposeTest(unittest.TestCase):
    def test_a_single_topic_question_is_retrieved_whole_as_typed(self) -> None:
        # Even if the model rephrases it: one part means "not split", and the typed question is
        # what single-topic retrieval has always used.
        d = decompose("Why was X chosen?", client=FakeClient("Why was X picked?"))
        self.assertEqual((d.parts, d.method), (["Why was X chosen?"], "whole"))

    def test_a_grounded_split_is_used(self) -> None:
        reply = ("What was the size of the automated test suite for /version?\n"
                 "What was the size of the automated test suite /health shipped with?")
        d = decompose(Q, client=FakeClient(reply))
        self.assertEqual(d.method, "split")
        self.assertEqual(len(d.parts), 2)

    def test_a_part_that_adds_words_is_dropped(self) -> None:
        reply = ("What was the size of the automated test suite for /version?\n"
                 "How many Jest tests did /health have?")
        d = decompose(Q, client=FakeClient(reply))
        # One grounded part is not a split; the question goes whole.
        self.assertEqual((d.parts, d.method), ([Q], "fallback-guard"))
        self.assertEqual(len(d.rejected), 1)

    def test_an_api_error_retrieves_the_question_whole(self) -> None:
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        self.assertEqual(decompose(Q, client=FakeClient(error=error)).method, "fallback-error")

    def test_the_request_binds_to_the_real_sdk_signature(self) -> None:
        client = FakeClient("x")
        decompose(Q, client=client)
        self.assertEqual(client.calls[0]["extra_body"], {"temperature": 0.0})
        inspect.signature(anthropic.Anthropic(api_key="not-used").messages.create).bind(**client.calls[0])


class InterleaveTest(unittest.TestCase):
    def test_each_part_gets_a_place_before_any_part_gets_a_second(self) -> None:
        a = [R("a1"), R("a2"), R("a3")]
        b = [R("b1"), R("b2")]
        self.assertEqual([r.chunk_id for r in interleave([a, b], 3)], ["a1", "b1", "a2"])

    def test_a_passage_both_parts_found_appears_once(self) -> None:
        a = [R("x"), R("a2")]
        b = [R("x"), R("b2")]
        self.assertEqual([r.chunk_id for r in interleave([a, b], 5)], ["x", "a2", "b2"])

    def test_one_ranking_is_passed_through(self) -> None:
        a = [R("a1"), R("a2"), R("a3")]
        self.assertEqual(interleave([a], 2), a[:2])


if __name__ == "__main__":
    unittest.main()

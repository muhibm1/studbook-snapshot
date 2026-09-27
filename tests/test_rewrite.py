import hashlib
import unittest
from types import SimpleNamespace

import anthropic
import httpx

from retrieve.rewrite import (
    FUNCTION_WORDS, REWRITE_PROMPT, STEM_CHARS, SUFFIXES, build_rewrite_request, carried, clean,
    rewrite_query, stem, unsupported_words,
)

# Paddock carries a copy of the prompt and the guard (desk/src/main/studbook/rewrite.ts) and pins
# the same two hashes. Changing either here means changing it there, and updating both tests.
PROMPT_SHA256 = "962806d476602ccc58d71002603f69487eec0b2d047cbed7634f2e5286fde077"
# The guard is data (the word lists) and logic (the stemmer); the probe pins the logic, since two
# stemmers can share every list and still disagree on "pinning".
PROBE = "pinning pin stopping stop moved move broken broke deploying deployed uses use access accessed passes all tree running added"
GUARD_SHA256 = "345d0ee05a7e0beb235389e352309a6b8b06e27633fffc917795c708bfb321ca"


class FakeClient:
    def __init__(self, reply: str | None = None, error: Exception | None = None) -> None:
        self.reply, self.error, self.calls = reply, error, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


FIRST = "Why did control state move into the git common dir?"


class PinTest(unittest.TestCase):
    def test_the_prompt_matches_paddocks_copy(self) -> None:
        self.assertEqual(hashlib.sha256(REWRITE_PROMPT.encode()).hexdigest(), PROMPT_SHA256)

    def test_the_guard_matches_paddocks_copy(self) -> None:
        spec = (" ".join(sorted(FUNCTION_WORDS)) + f"|{STEM_CHARS}|" + " ".join(SUFFIXES)
                + "|" + " ".join(stem(w) for w in PROBE.split()))
        self.assertEqual(hashlib.sha256(spec.encode()).hexdigest(), GUARD_SHA256)


class UnsupportedWordsTest(unittest.TestCase):
    def test_resolving_a_pronoun_from_the_thread_adds_nothing(self) -> None:
        self.assertEqual(unsupported_words(
            "What broke before control state moved into the git common dir?",
            ["What broke because of it?", FIRST]), [])

    def test_an_inflection_of_a_typed_word_is_not_new(self) -> None:
        self.assertEqual(unsupported_words("deployed", ["how is it deploying"]), [])

    def test_a_new_subject_is_caught(self) -> None:
        self.assertIn("redis", unsupported_words("Why did Redis replace the git common dir?", [FIRST]))

    def test_inflections_match_their_stems(self) -> None:
        for typed, written in (("move", "moved"), ("broke", "broken"), ("deploying", "deployed"), ("uses", "use"),
                               ("pin", "pinning"), ("stop", "stopping")):
            self.assertEqual(unsupported_words(written, [typed]), [], f"{typed} -> {written}")

    def test_a_new_version_number_is_caught(self) -> None:
        # Numbers are exactly what a confabulating rewrite would add, and exactly what a search
        # would then match on.
        self.assertIn("0.4.3", unsupported_words("What changed in 0.4.3?", ["What changed in it?"]))

    def test_function_words_and_short_words_are_free(self) -> None:
        self.assertEqual(unsupported_words("Why was it so, and when?", ["x"]), [])


class HelpersTest(unittest.TestCase):
    def test_carried_keeps_the_last_two_questions(self) -> None:
        self.assertEqual(carried("q4", ["q1", "q2", "q3"]), "q2 q3 q4")

    def test_the_request_numbers_the_earlier_questions(self) -> None:
        self.assertEqual(build_rewrite_request("new?", ["a?", "b?"]),
                         "Earlier questions in this thread, oldest first:\n1. a?\n2. b?\n\nNew question: new?")

    def test_clean_keeps_one_line_and_drops_quotes(self) -> None:
        self.assertEqual(clean('"Why X?"\nBecause Y.'), "Why X?")
        self.assertEqual(clean("   "), "")


class RewriteQueryTest(unittest.TestCase):
    def test_a_fresh_question_makes_no_call(self) -> None:
        client = FakeClient("unused")
        self.assertEqual(rewrite_query("q?", [], client=client).method, "fresh")
        self.assertEqual(client.calls, [])

    def test_a_grounded_rewrite_is_used(self) -> None:
        text = "What broke before control state moved into the git common dir?"
        r = rewrite_query("What broke because of it?", [FIRST], client=FakeClient(text))
        self.assertEqual((r.query, r.method), (text, "rewritten"))

    def test_an_unchanged_question_searches_on_its_own(self) -> None:
        # The change-of-subject case: this is what concatenation got wrong.
        q = "How is the health endpoint's uptime measured?"
        r = rewrite_query(q, [FIRST], client=FakeClient(q))
        self.assertEqual((r.query, r.method), (q, "unchanged"))

    def test_a_rewrite_that_invents_falls_back_to_the_concatenation(self) -> None:
        q = "What replaced it?"
        r = rewrite_query(q, [FIRST], client=FakeClient("What replaced Redis in the git common dir?"))
        self.assertEqual(r.method, "fallback-guard")
        self.assertEqual(r.query, carried(q, [FIRST]))
        self.assertIn("Redis", r.rejected)

    def test_an_api_error_falls_back_rather_than_failing_the_question(self) -> None:
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        r = rewrite_query("What replaced it?", [FIRST], client=FakeClient(error=error))
        self.assertEqual(r.method, "fallback-error")

    def test_an_empty_reply_falls_back(self) -> None:
        self.assertEqual(rewrite_query("And then?", [FIRST], client=FakeClient("")).method, "fallback-error")

    def test_the_request_pins_temperature_through_extra_body(self) -> None:
        client = FakeClient("x")
        rewrite_query("And then?", [FIRST], client=client)
        self.assertEqual(client.calls[0]["extra_body"], {"temperature": 0.0})
        self.assertNotIn("temperature", client.calls[0])

    def test_the_request_is_accepted_by_the_real_sdk_signature(self) -> None:
        # Fakes accept any keyword; the SDK does not (the 1.6.0 temperature TypeError reached a
        # paid run that way -- docs/gold-set-audit.md). Build the real request without sending it.
        import inspect

        client = FakeClient("x")
        rewrite_query("And then?", [FIRST], client=client)
        signature = inspect.signature(anthropic.Anthropic(api_key="not-used").messages.create)
        signature.bind(**client.calls[0])  # raises TypeError if the real SDK would reject it


if __name__ == "__main__":
    unittest.main()

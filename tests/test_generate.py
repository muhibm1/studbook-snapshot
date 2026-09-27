import os
import unittest
from types import SimpleNamespace

import anthropic
import httpx

from answer.generate import DEFAULT_MODEL, clean_refusal, generate_answer, parse_cited_numbers
from retrieve.hybrid import Result


class CleanRefusalTest(unittest.TestCase):
    def test_a_clean_one_paragraph_refusal_is_unchanged(self) -> None:
        text = "Not in the record: no such detail appears in the passages."
        self.assertEqual(clean_refusal(text), text)

    def test_a_refusal_followed_by_a_trailing_partial_answer_is_truncated(self) -> None:
        # The exact shape observed live: a correct refusal, a blank line, then the model
        # answering anyway with a citation -- docs/m3-generation.md.
        text = (
            "Not in the record: the passages do not specify what happens if a high finding turns up.\n\n"
            "The /version spec's constraint audit produced [5] zero high findings, one medium finding."
        )
        cleaned = clean_refusal(text)
        self.assertTrue(cleaned.startswith("Not in the record"))
        self.assertNotIn("[5]", cleaned)
        self.assertNotIn("constraint audit produced", cleaned)


class RefusalClassificationTest(unittest.TestCase):
    """The rule generate_answer applies: a refusal is the phrase AND no citations. Regression for
    the held-out-run bug where a hedged partial answer opening with the refusal phrase was counted
    as a refusal (and its citations discarded) even though it substantively answered."""

    def test_a_bare_refusal_has_no_citations(self) -> None:
        text = "Not in the record: the passages do not cover this."
        self.assertEqual(parse_cited_numbers(text, n_chunks=5), [])

    def test_a_hedged_answer_opening_with_the_phrase_still_has_citations(self) -> None:
        text = "Not in the record: the passages don't state the reason, but they do show the change landed in 0.2.12 [1]."
        self.assertEqual(parse_cited_numbers(text, n_chunks=5), [1])


class ParseCitedNumbersTest(unittest.TestCase):
    def test_single_bracket_citations_are_found_in_order(self) -> None:
        self.assertEqual(parse_cited_numbers("claim one [1], claim two [2].", n_chunks=3), [1, 2])

    def test_a_repeated_citation_is_deduplicated_keeping_first_position(self) -> None:
        self.assertEqual(parse_cited_numbers("[2] then again [2] then [1]", n_chunks=3), [2, 1])

    def test_a_citation_number_outside_the_given_chunks_is_dropped(self) -> None:
        self.assertEqual(parse_cited_numbers("see [1] and [9]", n_chunks=3), [1])

    def test_a_comma_joined_bracket_only_yields_its_leading_number(self) -> None:
        # The prompt instructs [1][2], never [1, 2] -- if the model does it anyway, the regex
        # only matches digits, so "[1, 2]" is not "[1]" and matches nothing here; documented
        # behaviour, not a crash.
        self.assertEqual(parse_cited_numbers("see [1, 2]", n_chunks=3), [])

    def test_no_citations_at_all(self) -> None:
        self.assertEqual(parse_cited_numbers("no brackets here", n_chunks=3), [])


class GenerateAnswerNoChunksTest(unittest.TestCase):
    def test_an_empty_chunk_list_refuses_without_calling_the_api(self) -> None:
        # No client is constructed here (would need ANTHROPIC_API_KEY) -- and none should be
        # needed: this is the "nothing retrieved" short-circuit, checked precisely because it
        # must not spend an API call.
        answer = generate_answer("anything", [], client=object())
        self.assertTrue(answer.refused)
        self.assertEqual(answer.citations, [])
        self.assertEqual(answer.input_tokens, 0)
        self.assertEqual(answer.output_tokens, 0)


class RecordingClient:
    """Captures the kwargs of each messages.create call and returns a minimal usable response.
    `reject_temperature` imitates a model that 400s on the parameter (Sonnet 5 does)."""

    def __init__(self, reject_temperature: bool = False) -> None:
        self.calls: list[dict] = []
        outer = self

        class Messages:
            def create(self, **kwargs):
                outer.calls.append(kwargs)
                if reject_temperature and "extra_body" in kwargs:
                    raise anthropic.BadRequestError(
                        message='`temperature` is deprecated for this model.',
                        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                        body=None,
                    )
                return SimpleNamespace(
                    content=[SimpleNamespace(type="text", text="The answer is in passage [1].")],
                    usage=SimpleNamespace(input_tokens=10, output_tokens=5),
                )

        self.messages = Messages()

    @property
    def kwargs(self) -> dict:
        return self.calls[-1]


class TemperatureTest(unittest.TestCase):
    def setUp(self) -> None:
        # Module-level cache of models that rejected the parameter: left dirty, it would decide
        # the outcome of whichever test happened to run next.
        from answer import generate

        self.addCleanup(generate._TEMPERATURE_UNSUPPORTED.clear)

    def test_generation_is_pinned_to_temperature_zero(self) -> None:
        # The API default is 1. Left unset, the same question against the same passages returns
        # different receipts on a re-run, and every eval number carries sampling noise.
        chunks = [Result(chunk_id="a#0", document_id="a", heading_path="H", body="body",
                          repo="r", doc_type="spec")]
        client = RecordingClient()
        generate_answer("q?", chunks, client=client, model="model-that-takes-temperature")
        self.assertEqual(client.kwargs["extra_body"], {"temperature": 0.0})
        # It travels in extra_body because anthropic 1.6.0's Messages.create() raises TypeError
        # on a `temperature` keyword; a fake client would happily accept either, so this asserts
        # the shape the real SDK needs.
        self.assertNotIn("temperature", client.kwargs)

    def test_a_model_that_rejects_temperature_is_retried_once_without_it(self) -> None:
        # Sonnet 5 answers `400 "temperature" is deprecated for this model`. Crashing on that
        # would make switching STUDBOOK_GENERATOR_MODEL a hard failure mid-run, which is how it
        # was first found.
        chunks = [Result(chunk_id="a#0", document_id="a", heading_path="H", body="body",
                          repo="r", doc_type="spec")]
        client = RecordingClient(reject_temperature=True)
        answer = generate_answer("q?", chunks, client=client, model="model-that-refuses-it")
        self.assertEqual(len(client.calls), 2)
        self.assertIn("extra_body", client.calls[0])
        self.assertNotIn("extra_body", client.calls[1])
        self.assertFalse(answer.refused)

    def test_a_model_known_to_reject_it_is_not_asked_twice(self) -> None:
        chunks = [Result(chunk_id="a#0", document_id="a", heading_path="H", body="body",
                          repo="r", doc_type="spec")]
        client = RecordingClient(reject_temperature=True)
        generate_answer("q?", chunks, client=client, model="sticky-model")
        generate_answer("q?", chunks, client=RecordingClient(), model="sticky-model")
        self.assertEqual(len(client.calls), 2)  # first call paid the discovery

    def test_a_400_that_is_not_about_temperature_still_raises(self) -> None:
        chunks = [Result(chunk_id="a#0", document_id="a", heading_path="H", body="body",
                          repo="r", doc_type="spec")]

        class Failing:
            class messages:
                @staticmethod
                def create(**kwargs):
                    raise anthropic.BadRequestError(
                        message="max_tokens is too large",
                        response=httpx.Response(400, request=httpx.Request("POST", "http://x")),
                        body=None,
                    )

        with self.assertRaises(RuntimeError):
            generate_answer("q?", chunks, client=Failing(), model="m")

    def test_an_explicit_temperature_is_passed_through(self) -> None:
        chunks = [Result(chunk_id="a#0", document_id="a", heading_path="H", body="body",
                          repo="r", doc_type="spec")]
        client = RecordingClient()
        generate_answer("q?", chunks, client=client, temperature=0.7)
        self.assertEqual(client.kwargs["extra_body"], {"temperature": 0.7})

    def test_the_real_sdk_would_accept_the_call_we_build(self) -> None:
        """A fake client accepts any keyword, so the tests above cannot catch a keyword the real
        SDK rejects -- which is exactly what happened: `temperature=` raised TypeError against
        anthropic 1.6.0 and only surfaced 40 questions into a paid eval run. This binds the
        captured kwargs against the installed SDK's own signature, no network and no key needed.
        """
        import inspect

        import anthropic

        chunks = [Result(chunk_id="a#0", document_id="a", heading_path="H", body="body",
                          repo="r", doc_type="spec")]
        client = RecordingClient()
        generate_answer("q?", chunks, client=client)
        signature = inspect.signature(anthropic.Anthropic(api_key="not-used").messages.create)
        signature.bind(**client.kwargs)  # raises TypeError if the real SDK would reject it


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
class LiveGenerateSmokeTest(unittest.TestCase):
    """A handful of real API calls (Haiku 4.5, capped at 1024 output tokens each -- a few cents
    total), covering exactly what M3 promises: a grounded, cited answer, and a refusal on a
    question the given passages don't support."""

    def test_an_answerable_question_cites_the_supporting_passage(self) -> None:
        chunks = [
            Result(chunk_id="doc-a#0", document_id="doc-a", heading_path="Codebase map > Inventory",
                   body="| Runtime | Node.js v21.7.3 on the operator's machine | confirmed |",
                   repo="paddock-demo", doc_type="codebase-map"),
            Result(chunk_id="doc-b#0", document_id="doc-b", heading_path="Constraints > Technical constraints",
                   body="**Node v21.7.3, npm 10.5.0** on the operator's machine. `confirmed`. "
                        "There is no .nvmrc and no engines field, so nothing pins the version.",
                   repo="paddock-demo", doc_type="constraints"),
        ]
        answer = generate_answer("Which Node.js version does the service run on?", chunks, model=DEFAULT_MODEL)
        self.assertFalse(answer.refused, f"expected a grounded answer, got a refusal: {answer.text!r}")
        self.assertIn("21.7.3", answer.text)
        self.assertTrue(answer.citations, "expected at least one citation in a grounded answer")
        self.assertTrue(all(c.number in (1, 2) for c in answer.citations))

    def test_a_question_the_passages_dont_support_is_refused(self) -> None:
        chunks = [
            Result(chunk_id="doc-a#0", document_id="doc-a", heading_path="Spec > Runtime",
                   body="The service targets Node.js v21.7.3 on the operator's machine.",
                   repo="paddock-demo", doc_type="constraints"),
        ]
        answer = generate_answer("Which Kubernetes distribution does this deploy to?", chunks, model=DEFAULT_MODEL)
        self.assertTrue(answer.refused, f"expected a refusal, got: {answer.text!r}")
        self.assertTrue(answer.text.startswith("Not in the record"))
        self.assertEqual(answer.citations, [])


if __name__ == "__main__":
    unittest.main()

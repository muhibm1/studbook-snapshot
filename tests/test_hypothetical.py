import inspect
import unittest
from types import SimpleNamespace

import anthropic
import httpx

from eval.hypothetical import HYPOTHETICAL_PROMPT, hypothetical_passage


class FakeClient:
    def __init__(self, reply="", error=None):
        self.reply, self.error, self.calls = reply, error, []
        self.messages = self

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.reply)])


class HypotheticalPassageTest(unittest.TestCase):
    def test_returns_the_passage_text(self) -> None:
        self.assertEqual(hypothetical_passage("q?", client=FakeClient("  ## Verification\nnpm test: 12 pass  ")),
                         "## Verification\nnpm test: 12 pass")

    def test_a_failed_call_is_an_empty_passage_not_a_failed_question(self) -> None:
        error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
        self.assertEqual(hypothetical_passage("q?", client=FakeClient(error=error)), "")

    def test_the_prompt_is_general_not_fitted_to_a_dev_question(self) -> None:
        # The diagnosis came from version-14 (test counts in verification logs). Naming TAP, "pass"
        # or "fail" here would be writing the answer into the method.
        import re
        words = set(re.findall(r"[\w/-]+", HYPOTHETICAL_PROMPT.lower()))
        for fitted in ("tap", "pass", "fail", "count", "counts", "tests", "version-14", "/health", "/version"):
            self.assertNotIn(fitted, words)

    def test_the_request_binds_to_the_real_sdk_signature(self) -> None:
        client = FakeClient("x")
        hypothetical_passage("q?", client=client)
        self.assertEqual(client.calls[0]["extra_body"], {"temperature": 0.0})
        inspect.signature(anthropic.Anthropic(api_key="not-used").messages.create).bind(**client.calls[0])


if __name__ == "__main__":
    unittest.main()

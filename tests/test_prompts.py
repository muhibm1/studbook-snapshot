import unittest

from answer.prompts import build_context, build_user_prompt
from retrieve.hybrid import Result


def result(chunk_id: str, body: str, heading_path: str = "H1 > H2") -> Result:
    return Result(chunk_id=chunk_id, document_id="d", heading_path=heading_path, body=body, repo="r", doc_type="spec")


class BuildContextTest(unittest.TestCase):
    def test_chunks_are_numbered_from_one_in_order(self) -> None:
        ctx = build_context([result("a", "first body"), result("b", "second body")])
        self.assertIn("[1] ", ctx)
        self.assertIn("[2] ", ctx)
        self.assertLess(ctx.index("[1]"), ctx.index("[2]"))
        self.assertIn("first body", ctx)
        self.assertIn("second body", ctx)

    def test_each_entry_names_its_repo_doc_type_and_heading(self) -> None:
        ctx = build_context([result("a", "body", heading_path="Spec > Requirements")])
        self.assertIn("r/spec", ctx)
        self.assertIn("Spec > Requirements", ctx)

    def test_an_empty_chunk_list_produces_an_empty_context(self) -> None:
        self.assertEqual(build_context([]), "")


class BuildUserPromptTest(unittest.TestCase):
    def test_the_question_appears_after_the_context(self) -> None:
        prompt = build_user_prompt("why did this happen?", [result("a", "the reason")])
        self.assertIn("the reason", prompt)
        self.assertIn("why did this happen?", prompt)
        self.assertLess(prompt.index("the reason"), prompt.index("why did this happen?"))


if __name__ == "__main__":
    unittest.main()


class PaddockParityTest(unittest.TestCase):
    """Paddock runs these prompts in its own process (desk/src/main/studbook/answer.ts) and pins the
    same hashes in desk/tests/studbook.test.ts. An edit made on one side only fails a test."""

    SYSTEM_PROMPT_SHA256 = "b51fefa85f5b85e3bec346c37044979a17751c6d8719f500cf925faf4a2a8658"
    FOLLOW_UP_NOTE_SHA256 = "6bf13313380c5b91c5a69816437e4eb74f9101c4da989433d27868bc76e47417"

    def test_the_system_prompt_is_paddocks(self) -> None:
        import hashlib

        from answer.prompts import SYSTEM_PROMPT
        self.assertEqual(hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest(), self.SYSTEM_PROMPT_SHA256)

    def test_the_follow_up_note_is_paddocks(self) -> None:
        import hashlib

        from answer.prompts import FOLLOW_UP_NOTE
        self.assertEqual(hashlib.sha256(FOLLOW_UP_NOTE.encode()).hexdigest(), self.FOLLOW_UP_NOTE_SHA256)


class BuildMessagesTest(unittest.TestCase):
    def test_no_history_is_the_single_message_it_always_was(self) -> None:
        from answer.prompts import build_messages
        chunks = [result("a", "body")]
        self.assertEqual(build_messages("q?", chunks), [{"role": "user", "content": build_user_prompt("q?", chunks)}])

    def test_history_becomes_turns_and_the_note_rides_on_the_follow_up_only(self) -> None:
        from answer.prompts import FOLLOW_UP_NOTE, SYSTEM_PROMPT, build_messages
        messages = build_messages("and then?", [result("a", "body")], [("first?", "first answer")])
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"], "first?")
        self.assertEqual(messages[1]["content"], "first answer")
        self.assertTrue(messages[2]["content"].startswith(FOLLOW_UP_NOTE))
        self.assertTrue(messages[2]["content"].endswith("Question: and then?"))
        self.assertNotIn(FOLLOW_UP_NOTE, SYSTEM_PROMPT)

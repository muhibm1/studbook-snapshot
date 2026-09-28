import os
import unittest

from answer.generate import Answer
from eval.judge import JUDGE_MODEL, build_judge_prompt, judge_answer
from retrieve.hybrid import Result


def result(chunk_id: str, body: str) -> Result:
    return Result(chunk_id=chunk_id, document_id="d", heading_path="H", body=body, repo="r", doc_type="spec")


class BuildJudgePromptTest(unittest.TestCase):
    def test_the_prompt_includes_context_question_reference_and_system_answer(self) -> None:
        answer = Answer(text="It is 42 [1].", citations=[], refused=False, model="m", input_tokens=1, output_tokens=1)
        prompt = build_judge_prompt(
            "what is the answer?", "The answer is 42.", [result("a", "the answer is 42")], answer
        )
        self.assertIn("the answer is 42", prompt)  # context passage
        self.assertIn("what is the answer?", prompt)  # question
        self.assertIn("The answer is 42.", prompt)  # reference
        self.assertIn("It is 42 [1].", prompt)  # system's answer


@unittest.skipUnless(os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY not set")
class LiveJudgeSmokeTest(unittest.TestCase):
    """A couple of real Sonnet 5 calls -- a few cents -- covering the two ends of the rubric:
    a clearly correct answer and a clearly wrong one."""

    def test_a_correct_faithful_answer_scores_well(self) -> None:
        chunks = [result("a", "The service targets Node.js v21.7.3 on the operator's machine.")]
        answer = Answer(text="It runs on Node.js v21.7.3 [1].", citations=[], refused=False,
                         model=JUDGE_MODEL, input_tokens=1, output_tokens=1)
        verdict = judge_answer("Which Node version?", "Node.js v21.7.3.", chunks, answer)
        self.assertEqual(verdict.correctness, 2)
        self.assertTrue(verdict.faithful)
        self.assertTrue(verdict.citations_valid)

    def test_a_fabricated_answer_scores_poorly(self) -> None:
        chunks = [result("a", "The service targets Node.js v21.7.3 on the operator's machine.")]
        answer = Answer(text="It runs on Python 3.11 [1].", citations=[], refused=False,
                         model=JUDGE_MODEL, input_tokens=1, output_tokens=1)
        verdict = judge_answer("Which language runtime?", "Node.js v21.7.3.", chunks, answer)
        self.assertEqual(verdict.correctness, 0)
        self.assertFalse(verdict.faithful)

    def test_an_invented_vendor_is_caught_even_though_the_answer_hedges(self) -> None:
        chunks = [result("a", "There is no reverse proxy, no TLS, and no auth in front of it.")]
        answer = Answer(text="The record is thin here, but nginx fronts the service [1].",
                         citations=[], refused=False, model=JUDGE_MODEL, input_tokens=1, output_tokens=1)
        verdict = judge_answer("Which load balancer sits in front of it?",
                                "Not in the record: no load balancer is mentioned anywhere.",
                                chunks, answer)
        self.assertTrue(verdict.invented_entity)

    def test_a_grounded_negative_is_correct_and_invents_nothing(self) -> None:
        # The behaviour the denial-trap rule exists to protect: answering in the negative from the
        # passage that denies the thing, rather than refusing outright.
        chunks = [result("a", "There is no reverse proxy, no TLS, and no auth in front of it.")]
        answer = Answer(text="There is none: the codebase map records no reverse proxy, no TLS and "
                              "no auth in front of the service [1].",
                         citations=[], refused=False, model=JUDGE_MODEL, input_tokens=1, output_tokens=1)
        verdict = judge_answer("Which load balancer sits in front of it?",
                                "Not in the record: no load balancer is mentioned anywhere.",
                                chunks, answer)
        self.assertEqual(verdict.correctness, 2)
        self.assertFalse(verdict.invented_entity)


if __name__ == "__main__":
    unittest.main()

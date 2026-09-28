import unittest

from eval.unrelated import docs_of, pair_up


def q(qid: str, *paths: str) -> dict:
    return {"id": qid, "question": f"question {qid}?",
            "gold": [{"source": "repo", "path": p, "quote": "q"} for p in paths]}


class PairUpTest(unittest.TestCase):
    def test_every_pair_is_unrelated(self) -> None:
        # The whole measurement rests on this: a "previous question" that shares a gold document
        # is not noise, it is context, and the pair would be measuring the opposite thing.
        questions = [q("a", "one.md"), q("b", "one.md"), q("c", "two.md"), q("d", "three.md")]
        for previous, question in pair_up(questions):
            self.assertFalse(docs_of(previous) & docs_of(question), f"{previous['id']} / {question['id']}")

    def test_it_measures_every_question_that_has_a_partner(self) -> None:
        questions = [q("a", "one.md"), q("b", "two.md"), q("c", "three.md")]
        self.assertEqual([p[1]["id"] for p in pair_up(questions)], ["a", "b", "c"])

    def test_it_is_deterministic(self) -> None:
        questions = [q("a", "one.md"), q("b", "two.md"), q("c", "three.md"), q("d", "one.md")]
        self.assertEqual(pair_up(questions), pair_up(questions))

    def test_it_skips_a_question_with_no_disjoint_partner(self) -> None:
        # "a" touches both documents in the set, so nothing here is unrelated to it. Dropping it is
        # right -- there is no unrelated previous turn to give it -- but it must not be dropped
        # silently, so the runner compares the pair count against the question count.
        questions = [q("a", "one.md", "two.md"), q("b", "one.md"), q("c", "two.md")]
        self.assertEqual([p[1]["id"] for p in pair_up(questions)], ["b", "c"])

    def test_the_only_question_in_the_set_has_no_partner(self) -> None:
        self.assertEqual(pair_up([q("a", "one.md")]), [])

    def test_a_question_sharing_one_of_several_documents_is_not_a_partner(self) -> None:
        questions = [q("a", "one.md", "two.md"), q("b", "two.md"), q("c", "three.md")]
        previous_for_a = next(p for p, question in pair_up(questions) if question["id"] == "a")
        self.assertEqual(previous_for_a["id"], "c")


if __name__ == "__main__":
    unittest.main()

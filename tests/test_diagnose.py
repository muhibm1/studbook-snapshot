import unittest

from eval.diagnose import bucket


def record(**overrides) -> dict:
    base = {"type": "factual", "is_refusal": False, "gold_in_top_k": True, "correctness": 2}
    base.update(overrides)
    return base


class BucketTest(unittest.TestCase):
    def test_a_refusal_with_the_gold_passage_present_is_a_generation_problem(self) -> None:
        self.assertEqual(bucket(record(is_refusal=True, gold_in_top_k=True, correctness=0)),
                         "generation")

    def test_a_refusal_with_no_gold_passage_retrieved_is_a_retrieval_problem(self) -> None:
        # Refusing here is the right call -- counting it as a generation failure would send the
        # next fix at the prompt when the passage never arrived.
        self.assertEqual(bucket(record(is_refusal=True, gold_in_top_k=False, correctness=0)),
                         "retrieval")

    def test_answers_are_bucketed_by_correctness(self) -> None:
        self.assertEqual(bucket(record(correctness=2)), "answered fully")
        self.assertEqual(bucket(record(correctness=1)), "answered partly")
        self.assertEqual(bucket(record(correctness=0)), "answered wrongly")

    def test_every_record_lands_in_exactly_one_bucket(self) -> None:
        seen = {bucket(record(is_refusal=refused, gold_in_top_k=gold, correctness=c))
                for refused in (True, False) for gold in (True, False) for c in (0, 1, 2)}
        self.assertEqual(seen, {"generation", "retrieval", "answered fully", "answered partly",
                                "answered wrongly"})


if __name__ == "__main__":
    unittest.main()

import unittest
from dataclasses import dataclass

from eval.ceiling import ARM_K, FINAL_K, PRODUCTION_POOL, first_hit_rank


@dataclass
class FakeResult:
    body: str


GOLD = [{"quote": "the answer is 42"}]


class FirstHitRankTest(unittest.TestCase):
    def test_it_returns_a_one_based_rank(self) -> None:
        results = [FakeResult("nothing here"), FakeResult("we found the answer is 42 eventually")]
        self.assertEqual(first_hit_rank(results, GOLD), 2)

    def test_it_returns_none_when_no_result_contains_a_gold_quote(self) -> None:
        # None, not 0: a rank of 0 would compare as "better than rank 1" in any threshold check.
        self.assertIsNone(first_hit_rank([FakeResult("nothing here")], GOLD))

    def test_an_empty_result_list_is_a_miss(self) -> None:
        self.assertIsNone(first_hit_rank([], GOLD))

    def test_the_first_of_several_hits_wins(self) -> None:
        results = [FakeResult("the answer is 42"), FakeResult("the answer is 42 again")]
        self.assertEqual(first_hit_rank(results, GOLD), 1)


class ProductionConstantsTest(unittest.TestCase):
    def test_the_arm_depth_matches_retrieve_hybrid(self) -> None:
        # This tool's numbers only describe production while these agree; a change to one of them
        # silently turns every measurement into a different configuration.
        from retrieve.hybrid import DEFAULT_K

        self.assertEqual(ARM_K, DEFAULT_K)

    def test_the_pool_and_cut_match_what_the_service_uses(self) -> None:
        from eval.run_generation import RERANK_CANDIDATES, TOP_K

        self.assertEqual(PRODUCTION_POOL, RERANK_CANDIDATES)
        self.assertEqual(FINAL_K, TOP_K)


if __name__ == "__main__":
    unittest.main()

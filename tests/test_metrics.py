import unittest
from dataclasses import dataclass

from eval.metrics import ScoreBoard, is_hit


@dataclass
class FakeResult:
    body: str


class IsHitTest(unittest.TestCase):
    def test_a_quote_present_in_the_body_is_a_hit(self) -> None:
        self.assertTrue(is_hit(FakeResult("... the answer is 42 exactly ..."), [{"quote": "the answer is 42"}]))

    def test_no_quote_present_is_not_a_hit(self) -> None:
        self.assertFalse(is_hit(FakeResult("nothing relevant here"), [{"quote": "the answer is 42"}]))

    def test_any_one_of_several_gold_quotes_matching_counts_as_a_hit(self) -> None:
        gold = [{"quote": "first quote"}, {"quote": "second quote"}]
        self.assertTrue(is_hit(FakeResult("body has the second quote in it"), gold))


class ScoreBoardTest(unittest.TestCase):
    def test_a_hit_at_rank_1_scores_perfectly(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        results = [FakeResult("contains the target quote"), FakeResult("irrelevant")]
        board.record("factual", results, [{"quote": "the target quote"}])
        report = board.report()
        self.assertEqual(report["overall"]["recall@5"], 1.0)
        self.assertEqual(report["overall"]["mrr@10"], 1.0)
        self.assertEqual(report["overall"]["precision@5"], 0.2)  # 1 of 5 slots

    def test_a_hit_at_rank_3_scores_a_third_for_mrr(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        results = [FakeResult("no"), FakeResult("no"), FakeResult("has the target quote")]
        board.record("factual", results, [{"quote": "the target quote"}])
        report = board.report()
        self.assertEqual(report["overall"]["recall@5"], 1.0)
        self.assertAlmostEqual(report["overall"]["mrr@10"], 1 / 3)

    def test_no_hit_within_mrr_k_scores_zero_on_every_metric(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        board.record("factual", [FakeResult("irrelevant")] * 10, [{"quote": "never appears"}])
        report = board.report()
        self.assertEqual(report["overall"]["recall@5"], 0.0)
        self.assertEqual(report["overall"]["mrr@10"], 0.0)
        self.assertEqual(report["overall"]["precision@5"], 0.0)

    def test_a_hit_beyond_top_k_but_within_mrr_k_counts_for_mrr_not_recall_or_precision(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        results = [FakeResult("no")] * 6 + [FakeResult("has the target quote")]
        board.record("factual", results, [{"quote": "the target quote"}])
        report = board.report()
        self.assertEqual(report["overall"]["recall@5"], 0.0)
        self.assertEqual(report["overall"]["precision@5"], 0.0)
        self.assertAlmostEqual(report["overall"]["mrr@10"], 1 / 7)

    def test_by_group_breakdown_is_independent_per_group(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        board.record("decision-why", [FakeResult("has the target quote")], [{"quote": "the target quote"}])
        board.record("factual", [FakeResult("irrelevant")], [{"quote": "never appears"}])
        report = board.report()
        self.assertEqual(report["by_group"]["decision-why"]["recall@5"], 1.0)
        self.assertEqual(report["by_group"]["factual"]["recall@5"], 0.0)
        self.assertEqual(report["overall"]["recall@5"], 0.5)
        self.assertEqual(report["by_group"]["decision-why"]["n"], 1)
        self.assertEqual(report["by_group"]["factual"]["n"], 1)


def gold(path: str, quote: str) -> dict:
    return {"source": "repo", "path": path, "quote": quote}


class CoverageTest(unittest.TestCase):
    """Recall@k asks whether *any* gold passage was retrieved. A cross-doc question needs every one
    of its documents: on dev, each cross-doc question given only one of its two documents was
    refused and scored 0, while recall@5 counted it a hit (docs/cross-doc.md)."""

    def test_one_document_of_two_is_a_recall_hit_but_not_covered(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        board.record("cross-doc", [FakeResult("has alpha")], [gold("a.md", "alpha"), gold("b.md", "beta")])
        overall = board.report()["overall"]
        self.assertEqual(overall["recall@5"], 1.0)
        self.assertEqual(overall["coverage@5"], 0.0)

    def test_both_documents_in_the_top_k_is_covered(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        results = [FakeResult("has alpha"), FakeResult("no"), FakeResult("has beta")]
        board.record("cross-doc", results, [gold("a.md", "alpha"), gold("b.md", "beta")])
        self.assertEqual(board.report()["overall"]["coverage@5"], 1.0)

    def test_two_quotes_from_one_document_need_only_one_of_them(self) -> None:
        # Coverage is per document, not per quote: a second quote from a document already present
        # adds evidence, not a missing half.
        board = ScoreBoard(top_k=5, mrr_k=10)
        board.record("factual", [FakeResult("has alpha")], [gold("a.md", "alpha"), gold("a.md", "alpha two")])
        self.assertEqual(board.report()["overall"]["coverage@5"], 1.0)

    def test_a_document_found_only_below_top_k_is_not_covered(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        results = [FakeResult("has alpha")] + [FakeResult("no")] * 5 + [FakeResult("has beta")]
        board.record("cross-doc", results, [gold("a.md", "alpha"), gold("b.md", "beta")])
        self.assertEqual(board.report()["overall"]["coverage@5"], 0.0)

    def test_single_document_questions_have_coverage_equal_to_recall(self) -> None:
        board = ScoreBoard(top_k=5, mrr_k=10)
        board.record("factual", [FakeResult("has alpha")], [gold("a.md", "alpha")])
        board.record("factual", [FakeResult("no")], [gold("a.md", "alpha")])
        overall = board.report()["overall"]
        self.assertEqual(overall["coverage@5"], overall["recall@5"])


if __name__ == "__main__":
    unittest.main()

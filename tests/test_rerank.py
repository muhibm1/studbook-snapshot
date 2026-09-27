import unittest

from retrieve.hybrid import Result


def result(chunk_id: str, body: str) -> Result:
    return Result(chunk_id=chunk_id, document_id="d", heading_path="H", body=body, repo="r", doc_type="spec")


class RerankerTest(unittest.TestCase):
    """Loads the actual bge-reranker-base model (already cached locally by M4's dev tuning run) --
    not a live network dependency, just a slower local test than the pure-function ones."""

    @classmethod
    def setUpClass(cls) -> None:
        from retrieve.rerank import Reranker

        cls.reranker = Reranker()

    def test_reranking_moves_the_more_relevant_passage_first(self) -> None:
        candidates = [
            result("off-topic", "The team uses Manrope for UI typography."),
            result("on-topic", "A broker flags an account making a fourth day trade under $25k equity."),
        ]
        reranked = self.reranker.rerank("why did the pattern day trader rule change", candidates, top_k=2)
        self.assertEqual(reranked[0].chunk_id, "on-topic")

    def test_top_k_truncates_the_result(self) -> None:
        candidates = [result(str(i), f"passage number {i}") for i in range(5)]
        reranked = self.reranker.rerank("passage number 3", candidates, top_k=2)
        self.assertEqual(len(reranked), 2)

    def test_an_empty_candidate_list_returns_empty(self) -> None:
        self.assertEqual(self.reranker.rerank("anything", [], top_k=5), [])


if __name__ == "__main__":
    unittest.main()

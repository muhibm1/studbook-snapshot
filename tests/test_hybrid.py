import os
import unittest

from retrieve.hybrid import Result, reciprocal_rank_fusion, retrieve_fulltext, retrieve_hybrid, retrieve_vector
from store.connect import connect, prepare_session

READER_URL = os.environ.get("STUDBOOK_DATABASE_URL")


def r(chunk_id: str) -> Result:
    return Result(chunk_id=chunk_id, document_id="d", heading_path="h", body="b", repo="r", doc_type="spec")


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_an_item_first_in_both_rankings_ranks_first_in_the_fusion(self) -> None:
        fused = reciprocal_rank_fusion([[r("a"), r("b")], [r("a"), r("c")]])
        self.assertEqual(fused[0].chunk_id, "a")

    def test_an_item_moderately_ranked_in_both_lists_can_beat_one_first_in_only_a_single_list(self) -> None:
        # "b" is rank 2 in both lists (score 1/62 + 1/62); "a" is rank 1 in only the first list and
        # entirely absent from the second (score 1/61) -- RRF should prefer broad moderate support.
        fused = reciprocal_rank_fusion([[r("a"), r("b")], [r("c"), r("b")]], k=60)
        self.assertEqual(fused[0].chunk_id, "b")

    def test_an_item_present_in_only_one_ranking_still_appears_in_the_fusion(self) -> None:
        fused = reciprocal_rank_fusion([[r("a")], [r("b")]])
        self.assertEqual({x.chunk_id for x in fused}, {"a", "b"})

    def test_empty_rankings_produce_an_empty_fusion(self) -> None:
        self.assertEqual(reciprocal_rank_fusion([[], []]), [])

    def test_a_single_ranking_is_returned_in_the_same_order(self) -> None:
        fused = reciprocal_rank_fusion([[r("a"), r("b"), r("c")]])
        self.assertEqual([x.chunk_id for x in fused], ["a", "b", "c"])


@unittest.skipUnless(READER_URL, "STUDBOOK_DATABASE_URL not set")
class LiveRetrievalSmokeTest(unittest.TestCase):
    """Not a scored eval (that's eval/run.py) -- just confirms the three retrieval paths return
    results shaped the way the rest of this module expects, against the real ingested store."""

    @classmethod
    def setUpClass(cls) -> None:
        from pgvector.psycopg import register_vector

        cls.conn = connect(READER_URL, autocommit=True)
        with cls.conn.cursor() as cur:
            prepare_session(cur)
        register_vector(cls.conn)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.conn.close()

    def test_fulltext_search_finds_a_distinctive_phrase(self) -> None:
        with self.conn.cursor() as cur:
            results = retrieve_fulltext(cur, "package.json version", k=5)
        self.assertTrue(results)
        self.assertTrue(all(isinstance(x, Result) for x in results))

    def test_vector_search_returns_results_for_a_natural_question(self) -> None:
        from ingest.embed import Embedder

        embedding = Embedder().embed(["why did the pattern day trader rule change"])[0]
        with self.conn.cursor() as cur:
            results = retrieve_vector(cur, embedding, k=5)
        self.assertEqual(len(results), 5)

    def test_a_scoped_vector_search_on_the_index_path_returns_every_result_asked_for(self) -> None:
        """The HNSW index hands back ef_search candidates and the scope filters them afterwards, so
        without pgvector's iterative scan a narrow scope can come back empty -- 0 of 20, measured.
        Today's ~500 passages make the planner sort instead, which hides it; so this forces the
        index path, the one the planner takes at scale. The control run with iterative scan off
        proves the test really reaches that path rather than passing on a sort."""
        from retrieve.hybrid import Scope

        def scoped_count(cur) -> int:
            cur.execute("select c.embedding from chunks c join documents d on d.id = c.document_id "
                        "where d.repo = 'paddock-demo' order by c.id limit 1")
            probe = cur.fetchone()[0].to_list()  # a passage from the OTHER repository: the hard case
            return len(retrieve_vector(cur, probe, k=20, scope=Scope(repos=("workhorse",))))

        with self.conn.cursor() as cur, self.conn.transaction():
            cur.execute("set local enable_sort = off; set local enable_seqscan = off")
            with_fix = scoped_count(cur)
            cur.execute("set local hnsw.iterative_scan = off")
            without_fix = scoped_count(cur)  # set local: both revert when the transaction ends
        self.assertEqual(with_fix, 20)
        self.assertLess(without_fix, 20, "control: expected the unfixed index path to lose results")

    def test_hybrid_search_merges_both_and_respects_top_k(self) -> None:
        from ingest.embed import Embedder

        embedding = Embedder().embed(["package.json version endpoint"])[0]
        with self.conn.cursor() as cur:
            results = retrieve_hybrid(cur, "package.json version endpoint", embedding, top_k=3)
        self.assertLessEqual(len(results), 3)
        self.assertGreater(len(results), 0)


if __name__ == "__main__":
    unittest.main()

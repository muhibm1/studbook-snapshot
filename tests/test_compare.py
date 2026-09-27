import json
import tempfile
import unittest
from pathlib import Path

from eval.compare import load, main, summarise


def record(qid: str, correctness: int = 2, faithful: bool = True, is_refusal: bool = False,
           qtype: str = "factual", invented: bool = False) -> dict:
    return {"id": qid, "type": qtype, "correctness": correctness, "faithful": faithful,
            "citations_valid": True, "is_refusal": is_refusal, "invented_entity": invented}


class SummariseTest(unittest.TestCase):
    def test_refusals_are_counted_over_answerable_questions_only(self) -> None:
        # A trap's refusal is the correct outcome; counting it as a refusal would make the
        # over-refusal figure improve every time a trap is added.
        records = {"a": record("a", is_refusal=True),
                   "t": record("t", qtype="unanswerable", is_refusal=True)}
        self.assertEqual(summarise(records)["refused_answerable"], 1)

    def test_means_cover_every_question(self) -> None:
        records = {"a": record("a", correctness=2), "b": record("b", correctness=0)}
        self.assertEqual(summarise(records)["correctness_mean"], 1.0)

    def test_an_empty_run_does_not_divide_by_zero(self) -> None:
        self.assertEqual(summarise({})["correctness_mean"], 0.0)


class CompareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def write(self, name: str, rows: list[dict]) -> Path:
        path = Path(self.tmp.name) / name
        path.write_text(json.dumps(rows), encoding="utf-8")
        return path

    def test_it_runs_over_two_dumps_and_reports_shared_questions(self) -> None:
        before = self.write("b.json", [record("q1", correctness=0, is_refusal=True), record("q2")])
        after = self.write("a.json", [record("q1"), record("q2")])
        self.assertEqual(main([str(before), str(after)]), 0)

    def test_two_runs_with_no_shared_questions_is_an_error(self) -> None:
        before = self.write("b.json", [record("q1")])
        after = self.write("a.json", [record("q9")])
        self.assertEqual(main([str(before), str(after)]), 2)

    def test_load_keys_records_by_id(self) -> None:
        path = self.write("x.json", [record("q1"), record("q2")])
        self.assertEqual(sorted(load(path)), ["q1", "q2"])


if __name__ == "__main__":
    unittest.main()

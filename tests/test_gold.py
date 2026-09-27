import tempfile
import unittest
from pathlib import Path

from eval.gold import check, split


def question(qid: str, qtype: str = "factual", quote: str = "the answer is 42", path: str = "doc.md") -> dict:
    row = {
        "id": qid,
        "type": qtype,
        "question": "What is the answer?",
        "reference_answer": "Not in the record: never discussed." if qtype == "unanswerable" else "42.",
        "gold": [] if qtype == "unanswerable" else [{"source": "repo", "path": path, "heading_path": "H", "quote": quote}],
    }
    if qtype == "unanswerable":
        row["trap_kind"] = "silence"
    return row


class CheckTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        (root / "doc.md").write_text("# H\r\n\r\nWe found the answer is 42 after all.\r\n", encoding="utf-8")
        self.corpus = {"repo": root}

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_a_verbatim_quote_passes(self) -> None:
        self.assertEqual(check([question("q1")], self.corpus), [])

    def test_a_paraphrased_quote_fails(self) -> None:
        problems = check([question("q1", quote="the answer was 42")], self.corpus)
        self.assertEqual(len(problems), 1)
        self.assertIn("not verbatim", problems[0].message)

    def test_a_missing_file_fails(self) -> None:
        problems = check([question("q1", path="nope.md")], self.corpus)
        self.assertIn("source not found", problems[0].message)

    def test_an_unanswerable_question_must_say_so_and_carry_no_gold(self) -> None:
        bad = question("q1", "unanswerable")
        bad["reference_answer"] = "It was MongoDB."
        bad["gold"] = [{"source": "repo", "path": "doc.md", "quote": "the answer is 42"}]
        messages = " ".join(p.message for p in check([bad], self.corpus))
        self.assertIn("no gold passages", messages)
        self.assertIn("Not in the record", messages)

    def test_duplicate_ids_fail(self) -> None:
        problems = check([question("q1"), question("q1")], self.corpus)
        self.assertEqual([p.message for p in problems], ["duplicate id"])

    def test_an_unanswerable_question_needs_a_known_trap_kind(self) -> None:
        bad = question("q1", "unanswerable")
        del bad["trap_kind"]
        self.assertIn("unknown trap_kind", " ".join(p.message for p in check([bad], self.corpus)))
        bad["trap_kind"] = "absent"
        self.assertIn("unknown trap_kind", " ".join(p.message for p in check([bad], self.corpus)))

    def test_a_denial_trap_must_carry_the_passage_that_denies_it(self) -> None:
        bad = question("q1", "unanswerable")
        bad["trap_kind"] = "denial"
        self.assertIn("must carry the passage", " ".join(p.message for p in check([bad], self.corpus)))

    def test_a_denial_quote_is_checked_verbatim_like_a_gold_quote(self) -> None:
        # The whole point of the denial label is that the record really does say the thing does
        # not exist -- an unverified quote would make that claim unfalsifiable.
        good = question("q1", "unanswerable")
        good["trap_kind"] = "denial"
        good["denial_evidence"] = [{"source": "repo", "path": "doc.md", "heading_path": "H",
                                    "quote": "the answer is 42"}]
        self.assertEqual(check([good], self.corpus), [])
        good["denial_evidence"][0]["quote"] = "the answer was 42"
        self.assertIn("not verbatim", " ".join(p.message for p in check([good], self.corpus)))

    def test_a_silence_trap_must_not_carry_denial_evidence(self) -> None:
        bad = question("q1", "unanswerable")
        bad["denial_evidence"] = [{"source": "repo", "path": "doc.md", "heading_path": "H",
                                   "quote": "the answer is 42"}]
        self.assertIn("no denial_evidence", " ".join(p.message for p in check([bad], self.corpus)))

    def test_an_answerable_question_cannot_carry_denial_evidence(self) -> None:
        bad = question("q1")
        bad["denial_evidence"] = [{"source": "repo", "path": "doc.md", "heading_path": "H",
                                   "quote": "the answer is 42"}]
        self.assertIn("only on an unanswerable question",
                      " ".join(p.message for p in check([bad], self.corpus)))


class SplitTest(unittest.TestCase):
    def test_split_is_deterministic_stratified_and_complete(self) -> None:
        questions = [question(f"q{i:02}", t) for i, t in enumerate(["factual"] * 9 + ["unanswerable"] * 6)]
        dev, test = split(questions)
        self.assertEqual(split(list(reversed(questions))), (dev, test))
        self.assertEqual(sorted(q["id"] for q in dev + test), sorted(q["id"] for q in questions))
        self.assertEqual(sum(q["type"] == "factual" for q in test), 3)
        self.assertEqual(sum(q["type"] == "unanswerable" for q in test), 2)


class PinnedTestSetTest(unittest.TestCase):
    def test_the_held_out_file_matches_its_pinned_hash(self) -> None:
        import hashlib

        root = Path(__file__).resolve().parent.parent
        pinned = (root / "eval" / "test.sha256").read_text(encoding="utf-8").split()[0]
        actual = hashlib.sha256((root / "eval" / "test.jsonl").read_bytes()).hexdigest()
        self.assertEqual(actual, pinned, "eval/test.jsonl changed after it was frozen")


if __name__ == "__main__":
    unittest.main()

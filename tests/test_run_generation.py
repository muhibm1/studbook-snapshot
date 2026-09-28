import unittest
from dataclasses import dataclass

from eval.run_generation import Totals, trap_passed


@dataclass
class FakeVerdict:
    """The three fields trap_passed reads, so the rule can be tested without an API call."""
    is_refusal: bool = False
    invented_entity: bool = False
    correctness: int = 2


class TotalsTest(unittest.TestCase):
    def test_finalize_computes_rates_over_accumulated_counts(self) -> None:
        t = Totals(n=4, correctness_sum=6, correctness_2=2, faithful=3, citations_valid=4, refusal_correct=4)
        report = t.finalize()
        self.assertEqual(report["n"], 4)
        self.assertEqual(report["correctness_mean"], 1.5)
        self.assertEqual(report["correctness_2_rate"], 0.5)
        self.assertEqual(report["faithful_rate"], 0.75)
        self.assertEqual(report["citation_valid_rate"], 1.0)
        self.assertEqual(report["refusal_accuracy"], 1.0)

    def test_finalize_on_zero_questions_does_not_divide_by_zero(self) -> None:
        report = Totals().finalize()
        self.assertEqual(report["n"], 0)
        self.assertEqual(report["correctness_mean"], 0.0)

    def test_trap_rates_appear_only_when_traps_were_scored(self) -> None:
        self.assertNotIn("trap_safety", Totals(n=3, correctness_sum=6).finalize())
        report = Totals(n=4, traps=4, traps_not_confabulated=4, traps_handled=3).finalize()
        self.assertEqual(report["trap_safety"], 1.0)
        self.assertEqual(report["trap_pass_rate"], 0.75)


class TrapPassedTest(unittest.TestCase):
    SILENCE = {"id": "wh-20", "trap_kind": "silence"}
    DENIAL = {"id": "x-02", "trap_kind": "denial"}

    def test_a_refusal_passes_either_kind(self) -> None:
        self.assertTrue(trap_passed(self.SILENCE, FakeVerdict(is_refusal=True)))
        self.assertTrue(trap_passed(self.DENIAL, FakeVerdict(is_refusal=True)))

    def test_inventing_something_fails_even_while_refusing(self) -> None:
        # An answer can decline overall and still smuggle in a vendor name; that is the exact
        # failure the trap set exists to catch, so it outranks every other signal.
        self.assertFalse(trap_passed(self.SILENCE, FakeVerdict(is_refusal=True, invented_entity=True)))
        self.assertFalse(trap_passed(self.DENIAL, FakeVerdict(is_refusal=True, invented_entity=True)))

    def test_a_grounded_negative_passes_a_denial_trap_but_not_a_silence_trap(self) -> None:
        # "There is no reverse proxy -- passage [2] says so" is a better answer than a bare
        # refusal, and an is_refusal-only rule scored it as a failed trap.
        grounded = FakeVerdict(is_refusal=False, correctness=2)
        self.assertTrue(trap_passed(self.DENIAL, grounded))
        self.assertFalse(trap_passed(self.SILENCE, grounded))

    def test_a_wrong_non_refusal_fails_a_denial_trap(self) -> None:
        self.assertFalse(trap_passed(self.DENIAL, FakeVerdict(is_refusal=False, correctness=1)))

    def test_a_trap_with_no_kind_recorded_can_only_be_refused(self) -> None:
        self.assertFalse(trap_passed({"id": "legacy"}, FakeVerdict(is_refusal=False, correctness=2)))


if __name__ == "__main__":
    unittest.main()

"""Diffs two `--dump` files from eval/run_generation.py, question by question.

The reason this exists: a single dev run's headline number is not stable enough to read a small
change off. Two prompt variants measured on 2026-09-16 differed by 0.025 correctness, and seven of
forty questions flipped between them -- so "the mean went up 0.02" and "nothing happened" are the
same observation. This prints the churn alongside the means, which is what tells the two apart:
a real effect moves questions in one direction, noise moves them in both.

    .venv/Scripts/python.exe -m eval.compare before.json after.json

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

FIELDS = ("correctness", "faithful", "citations_valid", "is_refusal")


def load(path: Path) -> dict[str, dict]:
    return {r["id"]: r for r in json.loads(path.read_text(encoding="utf-8"))}


def summarise(records: dict[str, dict]) -> dict[str, float]:
    n = len(records) or 1
    answerable = [r for r in records.values() if r["type"] != "unanswerable"]
    return {
        "correctness_mean": sum(r["correctness"] for r in records.values()) / n,
        "faithful_rate": sum(r["faithful"] for r in records.values()) / n,
        "refused_answerable": sum(r["is_refusal"] for r in answerable),
        "traps_invented": sum(r["invented_entity"] for r in records.values()
                              if r["type"] == "unanswerable"),
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args(argv)

    before, after = load(args.before), load(args.after)
    shared = [qid for qid in before if qid in after]
    if not shared:
        print("these two runs share no question ids", file=sys.stderr)
        return 2

    b, a = summarise(before), summarise(after)
    print(f"{len(shared)} question(s) in both runs\n")
    print(f"{'':22} {'before':>8} {'after':>8}")
    for key in ("correctness_mean", "faithful_rate", "refused_answerable", "traps_invented"):
        print(f"  {key:20} {b[key]:>8.3f} {a[key]:>8.3f}")

    better, worse, mixed = [], [], []
    for qid in shared:
        rb, ra = before[qid], after[qid]
        if all(rb[f] == ra[f] for f in FIELDS):
            continue
        # Correctness is the headline; faithfulness breaks the tie, because an answer that gained
        # a point while losing support for a claim is not an improvement.
        if ra["correctness"] > rb["correctness"] and ra["faithful"] >= rb["faithful"]:
            better.append(qid)
        elif ra["correctness"] < rb["correctness"] or (rb["faithful"] and not ra["faithful"]):
            worse.append(qid)
        else:
            mixed.append(qid)

    changed = len(better) + len(worse) + len(mixed)
    print(f"\n{changed} of {len(shared)} question(s) changed verdict")
    for label, ids in (("better", better), ("worse", worse), ("mixed", mixed)):
        if ids:
            print(f"  {label:7} {len(ids):2}  {', '.join(sorted(ids))}")
    if changed and len(worse) >= len(better):
        print("\nAs many questions got worse as better: read this as noise until it replicates.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

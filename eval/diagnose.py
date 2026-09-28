"""Reads a `--dump` file from eval/run_generation.py and separates the two stories behind a
refusal, without spending another API call:

  retrieval    the reranked top 5 held no gold passage, so refusing was the right call and the
               lever is retrieval
  generation   a gold passage was right there and the model refused anyway -- the lever is the
               prompt

    .venv/Scripts/python.exe -m eval.diagnose eval/dumps/dev.json
    .venv/Scripts/python.exe -m eval.diagnose eval/dumps/dev.json --show generation

Standard library only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def bucket(record: dict) -> str:
    """Every answerable question lands in exactly one bucket."""
    if record["is_refusal"]:
        return "generation" if record["gold_in_top_k"] else "retrieval"
    if record["correctness"] == 2:
        return "answered fully"
    return "answered partly" if record["correctness"] == 1 else "answered wrongly"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dump", type=Path)
    parser.add_argument("--show", choices=("generation", "retrieval", "answered wrongly", "all"),
                        help="print the question, the judge's reasoning and the answer for one bucket")
    args = parser.parse_args(argv)

    records = load(args.dump)
    answerable = [r for r in records if r["type"] != "unanswerable"]
    traps = [r for r in records if r["type"] == "unanswerable"]

    buckets = Counter(bucket(r) for r in answerable)
    n = len(answerable) or 1
    print(f"{len(records)} question(s): {len(answerable)} answerable, {len(traps)} trap(s)\n")
    print("answerable:")
    for name in ("answered fully", "answered partly", "answered wrongly", "generation", "retrieval"):
        count = buckets.get(name, 0)
        label = {"generation": "refused WITH the gold passage in front of it",
                 "retrieval": "refused, and no gold passage was retrieved"}.get(name, name)
        print(f"  {count:3}/{len(answerable)}  ({count / n:5.1%})  {label}")

    refusals = buckets.get("generation", 0) + buckets.get("retrieval", 0)
    if refusals:
        share = buckets.get("generation", 0) / refusals
        print(f"\n{share:.0%} of refusals had the material and refused anyway"
              f" -- that share is the part a prompt change can reach.")

    by_type: dict[str, Counter] = {}
    for r in answerable:
        by_type.setdefault(r["type"], Counter())[bucket(r)] += 1
    print("\nby type:")
    for qtype, counts in sorted(by_type.items()):
        total = sum(counts.values())
        refused = counts.get("generation", 0) + counts.get("retrieval", 0)
        print(f"  {qtype:14} n={total:2}  refused {refused}"
              f" (generation {counts.get('generation', 0)}, retrieval {counts.get('retrieval', 0)})")

    if traps:
        print(f"\ntraps: {sum(r['invented_entity'] for r in traps)} invented something, "
              f"{sum(r['is_refusal'] for r in traps)}/{len(traps)} refused outright")

    if args.show:
        wanted = [r for r in answerable if args.show == "all" or bucket(r) == args.show]
        print(f"\n--- {len(wanted)} question(s) in '{args.show}' ---")
        for r in wanted:
            print(f"\n{r['id']} ({r['type']}, gold at rank {r['gold_ranks'] or 'nowhere in top 5'})")
            print(f"  Q: {r['question']}")
            print(f"  judge: {r['judge_reasoning']}")
            print(f"  answer: {' '.join(r['answer'].split())[:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

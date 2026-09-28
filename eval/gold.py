"""Gold-set tooling: validate questions against the corpus, and split them into dev and test.

    python -m eval.gold check eval/gold.jsonl            # every quote verbatim in its source
    python -m eval.gold split eval/drafts/*.jsonl        # merge, validate, write gold/dev/test

Gold passages are anchored by verbatim quote rather than by chunk id, so the labels survive any
change to chunking. At evaluation time a chunk is gold for a question when it contains the quote
(whitespace-normalised). Standard library only.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TYPES = ("decision-why", "factual", "evidence", "cross-doc", "temporal", "unanswerable")
# An unanswerable question fails in one of two ways, and the difference changes what counts as a
# correct response (eval/run_generation.py). "silence": the record neither supplies the fact nor
# denies it, so the only right move is a refusal. "denial": the record affirmatively states the
# thing does not exist, so a grounded negative citing that passage is as good as a refusal --
# scoring it as a failure would penalise the better answer. Verified against the live corpus on
# 2026-09-16; each denial carries the passage that does the denying.
TRAP_KINDS = ("silence", "denial")
TEST_FRACTION = 1 / 3
SPLIT_SEED = 20260915  # fixed forever: changing it would leak test questions into tuning


@dataclass
class Problem:
    question_id: str
    message: str

    def __str__(self) -> str:
        return f"{self.question_id}: {self.message}"


def load_corpus() -> dict[str, Path]:
    config = Path(os.environ.get("STUDBOOK_CORPUS", ROOT / "corpus.json"))
    repos = json.loads(config.read_text(encoding="utf-8"))["repos"]
    return {name: Path(path) for name, path in repos.items()}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as err:
                raise SystemExit(f"{path}:{number}: not valid JSON ({err})") from err
    return rows


def source_text(corpus: dict[str, Path], source: str, path: str, cache: dict) -> str | None:
    """The text a gold quote must appear in: a file, or a commit message for `git:<sha>`."""
    key = (source, path)
    if key not in cache:
        repo = corpus.get(source)
        if repo is None:
            cache[key] = None
        elif path.startswith("git:"):
            result = subprocess.run(
                ["git", "-C", str(repo), "show", "-s", "--format=%B", path[4:]],
                capture_output=True, text=True, encoding="utf-8",
            )
            cache[key] = result.stdout if result.returncode == 0 else None
        else:
            file = repo / path
            cache[key] = file.read_text(encoding="utf-8") if file.is_file() else None
        if cache[key] is not None:
            cache[key] = cache[key].replace("\r\n", "\n")
    return cache[key]


def check(questions: list[dict], corpus: dict[str, Path]) -> list[Problem]:
    problems: list[Problem] = []
    seen: set[str] = set()
    cache: dict = {}
    for q in questions:
        qid = q.get("id", "<no id>")
        if qid in seen:
            problems.append(Problem(qid, "duplicate id"))
        seen.add(qid)
        for field in ("question", "reference_answer", "type"):
            if not str(q.get(field, "")).strip():
                problems.append(Problem(qid, f"missing {field}"))
        if q.get("type") not in TYPES:
            problems.append(Problem(qid, f"unknown type {q.get('type')!r}"))
        gold = q.get("gold", [])
        denial_evidence = q.get("denial_evidence", [])
        if q.get("type") == "unanswerable":
            if gold:
                problems.append(Problem(qid, "an unanswerable question must have no gold passages"))
            if not str(q.get("reference_answer", "")).startswith("Not in the record"):
                problems.append(Problem(qid, 'unanswerable reference must start "Not in the record"'))
            kind = q.get("trap_kind")
            if kind not in TRAP_KINDS:
                problems.append(Problem(qid, f"unknown trap_kind {kind!r}, expected one of {TRAP_KINDS}"))
            elif kind == "denial" and not denial_evidence:
                problems.append(Problem(qid, "a denial trap must carry the passage that denies it"))
            elif kind == "silence" and denial_evidence:
                problems.append(Problem(qid, "a silence trap must carry no denial_evidence"))
            problems.extend(check_quotes(qid, denial_evidence, corpus, cache))
            continue
        if denial_evidence:
            problems.append(Problem(qid, "denial_evidence belongs only on an unanswerable question"))
        if not gold:
            problems.append(Problem(qid, "no gold passages"))
        problems.extend(check_quotes(qid, gold, corpus, cache))
    return problems


def check_quotes(qid: str, passages: list[dict], corpus: dict[str, Path], cache: dict) -> list[Problem]:
    """Every quote must appear verbatim in the file (or commit message) it claims to come from."""
    problems: list[Problem] = []
    for g in passages:
        text = source_text(corpus, g.get("source", ""), g.get("path", ""), cache)
        if text is None:
            problems.append(Problem(qid, f"source not found: {g.get('source')}/{g.get('path')}"))
        elif g.get("quote", "") not in text:
            problems.append(Problem(qid, f"quote not verbatim in {g.get('path')}: {g.get('quote', '')[:60]!r}"))
    return problems


def split(questions: list[dict]) -> tuple[list[dict], list[dict]]:
    """Stratified by type, deterministic: the same questions always land in the same split."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for q in sorted(questions, key=lambda q: q["id"]):
        by_type[q["type"]].append(q)
    rng = random.Random(SPLIT_SEED)
    dev, test = [], []
    for qtype in TYPES:
        group = by_type.get(qtype, [])
        rng.shuffle(group)
        n_test = round(len(group) * TEST_FRACTION)
        test.extend(group[:n_test])
        dev.extend(group[n_test:])
    return sorted(dev, key=lambda q: q["id"]), sorted(test, key=lambda q: q["id"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    # LF on every platform: eval/test.sha256 pins these bytes, so Windows' CRLF must not leak in.
    with path.open("w", encoding="utf-8", newline="\n") as out:
        out.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("check", "split"):
        print(__doc__)
        return 2
    questions = [q for p in argv[1:] for q in read_jsonl(Path(p))]
    problems = check(questions, load_corpus())
    counts = Counter(q.get("type") for q in questions)
    print(f"{len(questions)} questions: " + ", ".join(f"{t} {counts.get(t, 0)}" for t in TYPES))
    for problem in problems:
        print(f"  FAIL {problem}")
    if problems:
        print(f"{len(problems)} problem(s)")
        return 1
    print("all quotes verbatim in their sources")
    if argv[0] == "split":
        dev, test = split(questions)
        write_jsonl(ROOT / "eval" / "gold.jsonl", sorted(questions, key=lambda q: q["id"]))
        write_jsonl(ROOT / "eval" / "dev.jsonl", dev)
        write_jsonl(ROOT / "eval" / "test.jsonl", test)
        print(f"wrote eval/gold.jsonl ({len(questions)}), eval/dev.jsonl ({len(dev)}), eval/test.jsonl ({len(test)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

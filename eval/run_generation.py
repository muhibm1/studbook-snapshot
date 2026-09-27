"""Scores generation end to end: retrieve (hybrid + rerank -- the config M4's ablations picked,
docs/m4-evaluation-and-scale.md) -> generate (Haiku 4.5) -> judge (Sonnet 5) against the
reference answer. Every question goes through the judge, unanswerable ones included: whether a
response refused, and whether it invented anything, are content judgements, not string matches.

    .venv/Scripts/python.exe -m eval.run_generation --split dev
    .venv/Scripts/python.exe -m eval.run_generation --split test
    .venv/Scripts/python.exe -m eval.run_generation --split dev --ids x-01 x-02  # a few questions

`--split test` requires typing "yes": scored once, per this project's own rule (eval/run.py,
docs/m0-census.md section 4).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from pgvector.psycopg import register_vector

from answer.generate import DEFAULT_MODEL, generate_answer
from answer.prompts import SYSTEM_PROMPT
from eval.judge import JUDGE_MODEL, JUDGE_SYSTEM, JudgeParseError, judge_answer
from eval.metrics import is_hit
from eval.scope import corpus_scope
from ingest.embed import Embedder
from retrieve.hybrid import retrieve_hybrid
from retrieve.rerank import Reranker
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
RERANK_CANDIDATES = 10  # see docs/retrieval-ceiling.md: same quality as 20, half the rerank


@dataclass
class Totals:
    n: int = 0
    correctness_sum: int = 0
    correctness_2: int = 0
    faithful: int = 0
    citations_valid: int = 0
    refusal_correct: int = 0
    traps: int = 0
    traps_not_confabulated: int = 0
    traps_handled: int = 0

    def finalize(self) -> dict:
        n = self.n or 1
        report = {
            "n": self.n,
            "correctness_mean": round(self.correctness_sum / n, 4),
            "correctness_2_rate": round(self.correctness_2 / n, 4),
            "faithful_rate": round(self.faithful / n, 4),
            "citation_valid_rate": round(self.citations_valid / n, 4),
            "refusal_accuracy": round(self.refusal_correct / n, 4),
        }
        if self.traps:
            # Reported separately from refusal_accuracy, which stays defined as "is_refusal
            # matched what the question type expects" so runs logged before 2026-09-16 stay
            # comparable. trap_safety is the claim worth making -- it never invented the vendor,
            # tool or number the question dangled -- and trap_pass_rate is the behaviour rule:
            # a silence trap must be refused, a denial trap may instead be answered in the
            # negative from the passage that denies it.
            report["traps"] = self.traps
            report["trap_safety"] = round(self.traps_not_confabulated / self.traps, 4)
            report["trap_pass_rate"] = round(self.traps_handled / self.traps, 4)
        return report


def studbook_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def load_split(split: str) -> list[dict]:
    path = ROOT / "eval" / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def trap_passed(question: dict, verdict) -> bool:
    """Did the system handle an unanswerable question correctly?

    Inventing the vendor, tool or number the question dangles always fails. Beyond that the bar
    depends on how the record fails to answer (`trap_kind`, eval/gold.py): a "silence" trap can
    only be refused, while a "denial" trap may instead be answered in the negative from the
    passage that states the thing does not exist -- the better answer of the two, and one an
    is_refusal-only rule would have logged as a failure.
    """
    if verdict.invented_entity:
        return False
    if verdict.is_refusal:
        return True
    return question.get("trap_kind") == "denial" and verdict.correctness == 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--generator-model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--rerank-candidates", type=int, default=RERANK_CANDIDATES,
                        help=f"how many fused candidates the cross-encoder re-scores (default "
                             f"{RERANK_CANDIDATES}). The dominant cost of a query: the rerank is "
                             "the slow step, and eval/ceiling.py measures what each pool can reach.")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"how many reranked passages the generator is given (default {TOP_K}). "
                             "eval/ceiling.py measures what each value can reach on dev.")
    parser.add_argument("--ids", nargs="+", metavar="ID",
                        help="score only these question ids -- a cheap way to check one behaviour "
                             "without paying for the whole split. Never logged as a run.")
    parser.add_argument("--dump", metavar="PATH",
                        help="write one JSON record per question (answer text, judge verdict, and "
                             "whether a gold passage reached the top 5) so a run can be diagnosed "
                             "afterwards without paying for it again")
    parser.add_argument("--system-prompt-file", metavar="PATH",
                        help="use this system prompt instead of answer/prompts.py's, for a prompt experiment. "
                             "Its hash is logged; a run with it is never mistaken for production.")
    parser.add_argument("--no-log", action="store_true")
    parser.add_argument("--yes", action="store_true",
                         help="confirm scoring the held-out split non-interactively (for a deliberate scripted run)")
    args = parser.parse_args()

    if args.split == "test" and not args.yes:
        answer = input('Type "yes" to score the held-out test split (this should happen once): ')
        if answer.strip().lower() != "yes":
            print("cancelled")
            return 1

    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("ANTHROPIC_API_KEY is not set.", file=sys.stderr)
        return 2

    system_prompt = SYSTEM_PROMPT
    if args.system_prompt_file:
        system_prompt = Path(args.system_prompt_file).read_text(encoding="utf-8")
        args.no_log = True  # an experiment's numbers are not the project's published ones

    questions = load_split(args.split)
    if args.ids:
        wanted = list(dict.fromkeys(args.ids))
        questions = [q for q in questions if q["id"] in set(wanted)]
        missing = [i for i in wanted if i not in {q["id"] for q in questions}]
        if missing:
            print(f"not in the {args.split} split: {', '.join(missing)}", file=sys.stderr)
            return 2
        # A partial run's numbers are not this split's score; logging them would pollute the
        # eval_runs history that the milestone write-ups read.
        args.no_log = True

    print("loading models (embedder, reranker) ...")
    embedder = Embedder()
    reranker = Reranker()

    overall = Totals()
    by_group: dict[str, Totals] = defaultdict(Totals)
    refused_wrongly, unanswered_but_should_refuse, judge_failures, confabulated = [], [], [], []
    records: list[dict] = []

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        scope = corpus_scope()
        for i, q in enumerate(questions, start=1):
            embedding = embedder.embed([q["question"]])[0]
            candidates = retrieve_hybrid(cur, q["question"], embedding, top_k=args.rerank_candidates, scope=scope)
            chunks = reranker.rerank(q["question"], candidates, top_k=args.top_k)
            answer = generate_answer(q["question"], chunks, model=args.generator_model, system=system_prompt)

            should_refuse = q["type"] == "unanswerable"
            # Every question goes through the judge, unanswerable ones included: whether a
            # response is a refusal is a content judgement, not a string match. Answer.refused
            # (the prefix heuristic the CLI uses for display) gets it wrong in both directions --
            # a hedged partial answer opening with "Not in the record" is not a refusal, and a
            # refusal that cites passages to show what the record *does* contain still is one
            # (wh-20 on the held-out run). The harness must not score itself on that heuristic.
            try:
                verdict = judge_answer(q["question"], q["reference_answer"], chunks, answer, model=args.judge_model)
            except JudgeParseError as e:
                # One bad judge parse (observed live: reasoning ran long enough to truncate the
                # JSON) must not lose every call already paid for in this run -- skip this
                # question's score, keep going, and report it plainly at the end.
                print(f"  judge parse failed for {q['id']}: {e}")
                judge_failures.append(q["id"])
                continue
            verdict_correctness = verdict.correctness
            verdict_faithful = verdict.faithful
            verdict_citations_valid = verdict.citations_valid
            refusal_correct = verdict.is_refusal == should_refuse
            if should_refuse and not verdict.is_refusal:
                unanswered_but_should_refuse.append(q["id"])
            if not should_refuse and verdict.is_refusal:
                refused_wrongly.append(q["id"])
            trap_handled = trap_passed(q, verdict) if should_refuse else False
            if should_refuse and verdict.invented_entity:
                confabulated.append(q["id"])

            for bucket in (overall, by_group[q["type"]]):
                bucket.n += 1
                bucket.correctness_sum += verdict_correctness
                bucket.correctness_2 += int(verdict_correctness == 2)
                bucket.faithful += int(verdict_faithful)
                bucket.citations_valid += int(verdict_citations_valid)
                bucket.refusal_correct += int(refusal_correct)
                if should_refuse:
                    bucket.traps += 1
                    bucket.traps_not_confabulated += int(not verdict.invented_entity)
                    bucket.traps_handled += int(trap_handled)

            if args.dump:
                # Where a gold passage landed separates the two refusal stories: the model had the
                # material and refused anyway, or retrieval never gave it to it. Recording it here
                # means that question can be answered from the file instead of from another run.
                gold_ranks = [i + 1 for i, c in enumerate(chunks) if is_hit(c, q.get("gold", []))]
                records.append({
                    "id": q["id"], "type": q["type"], "trap_kind": q.get("trap_kind"),
                    "question": q["question"], "reference_answer": q["reference_answer"],
                    "answer": answer.text, "cited": [c.number for c in answer.citations],
                    "gold_ranks": gold_ranks, "gold_in_top_k": bool(gold_ranks),
                    "correctness": verdict.correctness, "faithful": verdict.faithful,
                    "citations_valid": verdict.citations_valid, "is_refusal": verdict.is_refusal,
                    "invented_entity": verdict.invented_entity, "judge_reasoning": verdict.reasoning,
                    "passages": [{"rank": i + 1, "repo": c.repo, "heading_path": c.heading_path}
                                 for i, c in enumerate(chunks)],
                })

            trap_note = f" trap={q['trap_kind']}:{'pass' if trap_handled else 'FAIL'}" if should_refuse else ""
            print(f"  [{i}/{len(questions)}] {q['id']:12} correctness={verdict_correctness} "
                  f"faithful={verdict_faithful} citations_valid={verdict_citations_valid}{trap_note}")

    report = {
        "overall": overall.finalize(),
        "by_group": {g: t.finalize() for g, t in sorted(by_group.items())},
        "judge_failures": judge_failures,
    }
    print(f"\n{args.split}: {len(questions)} question(s)")
    o = report["overall"]
    print(f"  overall: correctness_mean={o['correctness_mean']}  correctness_2_rate={o['correctness_2_rate']}  "
          f"faithful_rate={o['faithful_rate']}  citation_valid_rate={o['citation_valid_rate']}  "
          f"refusal_accuracy={o['refusal_accuracy']}")
    for group, t in report["by_group"].items():
        print(f"    {group:14} (n={t['n']:2}): correctness_mean={t['correctness_mean']}  "
              f"faithful_rate={t['faithful_rate']}  refusal_accuracy={t['refusal_accuracy']}")
    trap = report["by_group"].get("unanswerable")
    if trap:
        print(f"    trap safety: {trap['trap_safety']} (nothing invented)  "
              f"trap pass rate: {trap['trap_pass_rate']} (refused, or a grounded negative on a denial trap)")
    if refused_wrongly:
        print(f"  refused an answerable question: {', '.join(refused_wrongly)}")
    if confabulated:
        print(f"  INVENTED something on an unanswerable question: {', '.join(confabulated)}")
    if unanswered_but_should_refuse:
        print(f"  did not refuse an unanswerable question (check trap_kind, may be a grounded "
              f"negative rather than a failure): {', '.join(unanswered_but_should_refuse)}")
    if judge_failures:
        print(f"  judge parse failures (excluded from the scores above): {', '.join(judge_failures)}")

    if args.dump:
        dump_path = Path(args.dump)
        dump_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nwrote {len(records)} per-question record(s) to {dump_path}")

    if not args.no_log:
        config = {
            "split": args.split, "kind": "generation", "top_k": args.top_k, "rerank_candidates": args.rerank_candidates,
            "generator_model": args.generator_model, "judge_model": args.judge_model,
            "repos": list(corpus_scope().repos or ()),
            # The prompts are as much of the configuration as the model name is: without them two
            # runs that differ only in how the system was asked to answer share a config_hash, and
            # eval_runs stops being the "reproducible claim" record it is meant to be.
            "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16],
            "judge_prompt_sha256": hashlib.sha256(JUDGE_SYSTEM.encode("utf-8")).hexdigest()[:16],
        }
        config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()
        with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
            cur.execute(
                "insert into studbook.eval_runs (split, config_hash, config, studbook_commit, metrics) "
                "values (%s, %s, %s, %s, %s)",
                (args.split, config_hash, json.dumps(config), studbook_commit(), json.dumps(report)),
            )
        print(f"\nlogged to studbook.eval_runs (config_hash={config_hash[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

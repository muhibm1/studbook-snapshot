"""A cheap, scripted spot-check of generation over the whole gold set -- not the scored
LLM-judge eval (that's M4: correctness, faithfulness, citation validity, refusal accuracy, each
with a target). This just answers "does the core M3 promise hold at all, broadly": does every
unanswerable question actually get refused, and does every answerable one come back with at
least one citation. Haiku 4.5, ~50-60 calls total, well under a cent each (eval/run.py --method
hybrid already showed dev-set retrieval numbers; this reuses the same retrieval to generate).

    .venv/Scripts/python.exe -m eval.check_answers --split dev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from pgvector.psycopg import register_vector

from answer.generate import generate_answer
from ingest.embed import Embedder
from retrieve.hybrid import retrieve_hybrid
from eval.scope import corpus_scope
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args()
    if args.split == "test":
        answer = input('Type "yes" to spend generation calls on the held-out test split: ')
        if answer.strip().lower() != "yes":
            print("cancelled")
            return 1

    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set.", file=sys.stderr)
        return 2

    rows = [json.loads(l) for l in (ROOT / "eval" / f"{args.split}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    unanswerable = [r for r in rows if r["type"] == "unanswerable"]
    answerable = [r for r in rows if r["type"] != "unanswerable"]

    print("loading the embedding model ...")
    embedder = Embedder()

    refused_correctly, refused_wrongly = 0, []
    cited, uncited = 0, []

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)

        print(f"\n{len(unanswerable)} unanswerable question(s):")
        for q in unanswerable:
            embedding = embedder.embed([q["question"]])[0]
            chunks = retrieve_hybrid(cur, q["question"], embedding, top_k=TOP_K, scope=corpus_scope())
            answer = generate_answer(q["question"], chunks)
            if answer.refused:
                refused_correctly += 1
            else:
                refused_wrongly.append(q["id"])
            # This tool uses the cheap prefix heuristic, so on a "denial" trap a non-refusal is
            # not necessarily wrong -- the record states the thing does not exist, and answering
            # that in the negative is a better response than a bare refusal. eval/run_generation.py
            # is what scores the distinction; here it only softens the label.
            if answer.refused:
                mark = "OK"
            else:
                mark = "MISS" if q.get("trap_kind") == "silence" else "CHECK"
            print(f"  {mark:5} {q['id']:10} {q['question'][:70]!r}")

        print(f"\n{len(answerable)} answerable question(s):")
        for q in answerable:
            embedding = embedder.embed([q["question"]])[0]
            chunks = retrieve_hybrid(cur, q["question"], embedding, top_k=TOP_K, scope=corpus_scope())
            answer = generate_answer(q["question"], chunks)
            if answer.refused:
                uncited.append((q["id"], "refused"))
            elif answer.citations:
                cited += 1
            else:
                uncited.append((q["id"], "no citation"))

    print(f"\nrefusal on unanswerable: {refused_correctly}/{len(unanswerable)}")
    if refused_wrongly:
        print(f"  answered instead of refusing: {', '.join(refused_wrongly)}")
    print(f"citation present on answerable: {cited}/{len(answerable)}")
    if uncited:
        print(f"  no citation: {', '.join(f'{i} ({why})' for i, why in uncited)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Ask Studbook a question from the terminal: hybrid retrieval, then a cited answer.

    .venv/Scripts/python.exe cli.py "why did the pattern day trader rule change"
    .venv/Scripts/python.exe cli.py "..." --model claude-sonnet-5 --top-k 8

Needs STUDBOOK_DATABASE_URL and ANTHROPIC_API_KEY in the environment.
"""

from __future__ import annotations

import argparse
import os
import sys

from pgvector.psycopg import register_vector

from answer.generate import DEFAULT_MODEL, generate_answer
from ingest.embed import Embedder
from retrieve.hybrid import retrieve_hybrid
from store.connect import connect, prepare_session


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("question")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set in this process's environment.", file=sys.stderr)
        return 2
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        print("ANTHROPIC_API_KEY is not set in this process's environment.", file=sys.stderr)
        return 2

    embedding = Embedder().embed([args.question])[0]
    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)
        chunks = retrieve_hybrid(cur, args.question, embedding, top_k=args.top_k)

    answer = generate_answer(args.question, chunks, model=args.model)
    print(answer.text)
    if answer.citations:
        print("\nReceipts:")
        for c in answer.citations:
            print(f"  [{c.number}] {c.repo}/{c.doc_type} -- {c.heading_path}\n      {c.chunk_id}")
    print(f"\n({answer.model}, {answer.input_tokens} in / {answer.output_tokens} out)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

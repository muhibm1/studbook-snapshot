"""Chunk-size ablation: does a smaller chunk budget retrieve better? Scoped down deliberately --
vector-only, in-memory, never touching the live store -- re-chunking and re-embedding the whole
corpus at an alternate size just to compare would mean either disturbing the production table or
building a second one; an in-memory numpy comparison answers the actual question (does chunk size
matter) without either. The M2 vector-only number at the production chunk size (600 words,
recall@5=0.5588 -- docs/m2-retrieval-and-eval.md) is the other side of this comparison, measured
for real against the live store, not re-derived here.

    .venv/Scripts/python.exe -m eval.chunk_size_ablation --max-words 300
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.gold import load_corpus
from eval.metrics import ScoreBoard
from ingest.chunker import chunk_document
from ingest.embed import Embedder
from ingest.parser import walk_repo

ROOT = Path(__file__).resolve().parent.parent


class InMemoryResult:
    """Just enough of retrieve.hybrid.Result's shape for ScoreBoard.record: it only reads .body."""

    def __init__(self, body: str) -> None:
        self.body = body


def load_dev_questions() -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "eval" / "dev.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    return [r for r in rows if r["type"] != "unanswerable"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-words", type=int, default=300)
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    corpus = load_corpus()
    print(f"re-chunking both repos at max_words={args.max_words} ...")
    chunks = []
    for repo, path in corpus.items():
        for doc in walk_repo(repo, path):
            chunks.extend(chunk_document(doc, max_words=args.max_words))
    print(f"  {len(chunks)} chunks (production, max_words=600, has 491)")

    print("embedding all chunks (in memory, not written to the store) ...")
    embedder = Embedder()
    bodies = [c.body for c in chunks]
    vectors = np.array(embedder.embed(bodies), dtype=np.float32)  # already normalised by embed.py

    questions = load_dev_questions()
    print(f"scoring {len(questions)} dev questions ...")
    board = ScoreBoard(top_k=args.top_k, mrr_k=10)
    for q in questions:
        q_vec = np.array(embedder.embed([q["question"]])[0], dtype=np.float32)
        scores = vectors @ q_vec  # cosine similarity, both sides unit-normalised
        top_idx = np.argsort(-scores)[:10]
        results = [InMemoryResult(chunks[i].body) for i in top_idx]
        board.record(q["type"], results, q["gold"])

    report = board.report()
    o = report["overall"]
    print(f"\nvector-only, max_words={args.max_words}, n={o['n']}: "
          f"recall@{args.top_k}={o[f'recall@{args.top_k}']:.4f}  mrr@10={o['mrr@10']:.4f}  "
          f"precision@{args.top_k}={o[f'precision@{args.top_k}']:.4f}")
    print("(compare: vector-only, max_words=600, n=34: recall@5=0.5588 mrr@10=0.4716 precision@5=0.1176 "
          "-- docs/m2-retrieval-and-eval.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Runs a gold-set split through retrieval and scores it: Recall@5, MRR@10, Precision@5, overall
and broken down by question type. Logs every run to studbook.eval_runs (config, config_hash, the
studbook commit, metrics) so a reported number is a reproducible claim, not a one-off print --
the security/engineering standard this project holds itself to (docs/m0-census.md section 5).

    .venv/Scripts/python.exe -m eval.run --split dev --method hybrid
    .venv/Scripts/python.exe -m eval.run --split dev --method vector
    .venv/Scripts/python.exe -m eval.run --split dev --method fulltext
    .venv/Scripts/python.exe -m eval.run --split dev --method hybrid --rrf-k 10

`--split test` is deliberately not the default: the test split is scored once, held out, per
docs/m0-census.md -- never used for the "first tuning loop" this milestone's config search is.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

from ingest.embed import Embedder
from retrieve.hybrid import EMPTY_SCOPE, Scope, retrieve_fulltext, retrieve_hybrid, retrieve_vector
from retrieve.rerank import Reranker
from eval.metrics import ScoreBoard
from eval.scope import corpus_scope
from store.connect import connect, prepare_session

ROOT = Path(__file__).resolve().parent.parent
TOP_K = 5
MRR_K = 10
METHODS = ("vector", "fulltext", "hybrid")


def load_split(split: str) -> list[dict]:
    path = ROOT / "eval" / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Retrieval metrics need a gold chunk to hit; the 9 unanswerable questions are a generation-
    # and-refusal check (M3/M4), not a retrieval one, and are excluded here on purpose.
    return [r for r in rows if r["type"] != "unanswerable"]


def studbook_commit() -> str:
    return subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


def run_eval(
    cur: psycopg.Cursor,
    embedder: Embedder,
    questions: list[dict],
    method: str,
    vector_k: int,
    fulltext_k: int,
    rrf_k: int,
    reranker: Reranker | None = None,
    rerank_candidates: int = 20,
    scope: Scope = EMPTY_SCOPE,
) -> ScoreBoard:
    board = ScoreBoard(top_k=TOP_K, mrr_k=MRR_K)
    # Reranking needs a wider candidate pool than MRR_K to have anything worth reordering --
    # fetched at rerank_candidates, then cut back down to MRR_K after rescoring.
    fetch_k = rerank_candidates if reranker else MRR_K
    for q in questions:
        embedding = embedder.embed([q["question"]])[0] if method != "fulltext" else None
        if method == "vector":
            results = retrieve_vector(cur, embedding, k=fetch_k, scope=scope)
        elif method == "fulltext":
            results = retrieve_fulltext(cur, q["question"], k=fetch_k, scope=scope)
        elif method == "hybrid":
            results = retrieve_hybrid(
                cur, q["question"], embedding, top_k=fetch_k, vector_k=vector_k, fulltext_k=fulltext_k, rrf_k=rrf_k,
                scope=scope,
            )
        else:
            raise ValueError(f"unknown method {method!r}")
        if reranker:
            results = reranker.rerank(q["question"], results, top_k=MRR_K)
        board.record(q["type"], results, q["gold"])
    return board


def _fmt(bucket: dict) -> str:
    return (f"recall@{TOP_K}={bucket[f'recall@{TOP_K}']:.4f}  coverage@{TOP_K}={bucket[f'coverage@{TOP_K}']:.4f}  "
            f"mrr@{MRR_K}={bucket[f'mrr@{MRR_K}']:.4f}  precision@{TOP_K}={bucket[f'precision@{TOP_K}']:.4f}")


def print_report(report: dict) -> None:
    o = report["overall"]
    print(f"  overall (n={o['n']}): {_fmt(o)}")
    for group, t in report["by_group"].items():
        print(f"    {group:14} (n={t['n']:2}): {_fmt(t)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--method", choices=METHODS, default="hybrid")
    parser.add_argument("--vector-k", type=int, default=20)
    parser.add_argument("--fulltext-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=10)
    parser.add_argument("--rerank", action="store_true", help="cross-encoder rerank the candidates before scoring")
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--no-log", action="store_true", help="don't write a row to studbook.eval_runs")
    parser.add_argument("--yes", action="store_true",
                         help="confirm scoring the held-out split non-interactively (for a deliberate scripted run)")
    args = parser.parse_args()

    if args.split == "test" and not args.yes:
        answer = input('Type "yes" to score the held-out test split (this should happen once, not while tuning): ')
        if answer.strip().lower() != "yes":
            print("cancelled")
            return 1

    reader_url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not reader_url:
        print("STUDBOOK_DATABASE_URL is not set in this process's environment.", file=sys.stderr)
        return 2

    questions = load_split(args.split)
    config = {
        "split": args.split, "method": args.method, "top_k": TOP_K, "mrr_k": MRR_K,
        "vector_k": args.vector_k, "fulltext_k": args.fulltext_k, "rrf_k": args.rrf_k,
        "rerank": args.rerank, "rerank_candidates": args.rerank_candidates if args.rerank else None,
        # Which repositories the query could see. Part of the config because the store can hold
        # more than the corpus (docs/paddock-gate.md); a run over a different set is a different run.
        "repos": list(corpus_scope().repos or ()),
        "embed_model": os.environ.get("STUDBOOK_EMBED_MODEL", "BAAI/bge-base-en-v1.5"),
        # What text the stored vectors were computed from. Prefixing the heading was tried on
        # 2026-09-16 and reverted (docs/retrieval-ceiling.md): vector-only recall rose one
        # question, production recall fell two. Without this key a run against a re-embedded
        # store shares a config_hash with every run before it, and eval_runs stops being able to
        # say which store a number came from.
        "embedding_text": "body",
        "reranker_model": os.environ.get("STUDBOOK_RERANKER_MODEL", "BAAI/bge-reranker-base") if args.rerank else None,
    }
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()

    print("loading the embedding model ..." if args.method != "fulltext" else "method=fulltext, no embedding model needed")
    embedder = Embedder() if args.method != "fulltext" else None
    reranker = None
    if args.rerank:
        print("loading the reranker model ...")
        reranker = Reranker()

    with connect(reader_url, autocommit=True) as conn, conn.cursor() as cur:
        prepare_session(cur)
        register_vector(conn)
        print(f"{args.split}: {len(questions)} question(s), method={args.method}{'+rerank' if args.rerank else ''}")
        board = run_eval(cur, embedder, questions, args.method, args.vector_k, args.fulltext_k, args.rrf_k,
                          reranker=reranker, rerank_candidates=args.rerank_candidates, scope=corpus_scope())
        report = board.report()
        print_report(report)

        if not args.no_log:
            cur.execute(
                "insert into studbook.eval_runs (split, config_hash, config, studbook_commit, metrics) "
                "values (%s, %s, %s, %s, %s)",
                (args.split, config_hash, json.dumps(config), studbook_commit(), json.dumps(report)),
            )
            print(f"logged to studbook.eval_runs (config_hash={config_hash[:12]}...)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

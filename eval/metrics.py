"""Pure scoring functions for the retrieval eval -- no DB, no embedding model, so they're cheap to
unit test directly. A "hit" is a retrieved chunk whose body contains one of a question's gold
quotes verbatim (the same quote-anchored definition eval/gold.py's checker uses), never a chunk-id
match -- chunk ids are re-derived from a document's heading structure and would silently break
under a chunking change that gold.py's check would not otherwise catch.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Protocol


class HasBody(Protocol):
    body: str


def is_hit(result: HasBody, gold: list[dict]) -> bool:
    return any(g["quote"] in result.body for g in gold)


def covers_all_documents(results: list[HasBody], gold: list[dict]) -> bool:
    """Every gold *document* has at least one of its quotes in `results`.

    Recall asks whether any gold passage came back. A question whose answer spans two documents
    needs both: on dev, each cross-doc question handed only one of its two documents was refused and
    scored 0, while recall@5 counted it a hit (docs/cross-doc.md). Grouped by document rather than
    by quote, since a second quote from a document already present is more evidence, not a missing
    half. For a single-document question this is exactly recall."""
    by_document: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for g in gold:
        by_document[(g.get("source", ""), g.get("path", ""))].append(g)
    return all(any(is_hit(r, quotes) for r in results) for quotes in by_document.values())


@dataclass
class Totals:
    n: int = 0
    recall_hits: int = 0
    coverage_hits: int = 0
    mrr_sum: float = 0.0
    precision_sum: float = 0.0

    def add(self, hits: list[bool], top_k: int, covered: bool | None = None) -> None:
        self.n += 1
        self.recall_hits += int(any(hits[:top_k]))
        self.coverage_hits += int(any(hits[:top_k]) if covered is None else covered)
        first_hit_rank = next((i + 1 for i, h in enumerate(hits) if h), None)
        self.mrr_sum += 1.0 / first_hit_rank if first_hit_rank else 0.0
        self.precision_sum += sum(hits[:top_k]) / top_k

    def finalize(self, top_k: int, mrr_k: int) -> dict:
        # Full precision on purpose: this feeds studbook.eval_runs, a "reproducible claim" record
        # (docs/m0-census.md section 5) -- round only for display, in eval/run.py's print_report.
        n = self.n or 1
        return {
            "n": self.n,
            f"recall@{top_k}": self.recall_hits / n,
            f"coverage@{top_k}": self.coverage_hits / n,
            f"mrr@{mrr_k}": self.mrr_sum / n,
            f"precision@{top_k}": self.precision_sum / n,
        }


@dataclass
class ScoreBoard:
    """Accumulates one question at a time; reports overall plus a breakdown by an arbitrary key
    (question type, or gold doc_type) without needing every question up front."""

    top_k: int
    mrr_k: int
    overall: Totals = field(default_factory=Totals)
    by_group: dict[str, Totals] = field(default_factory=lambda: defaultdict(Totals))

    def record(self, group: str, results: list[HasBody], gold: list[dict]) -> None:
        hits = [is_hit(r, gold) for r in results[: self.mrr_k]]
        covered = covers_all_documents(results[: self.top_k], gold)
        self.overall.add(hits, self.top_k, covered)
        self.by_group[group].add(hits, self.top_k, covered)

    def report(self) -> dict:
        return {
            "overall": self.overall.finalize(self.top_k, self.mrr_k),
            "by_group": {g: t.finalize(self.top_k, self.mrr_k) for g, t in sorted(self.by_group.items())},
        }

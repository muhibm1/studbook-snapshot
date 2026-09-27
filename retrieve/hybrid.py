"""Retrieval: vector search (HNSW, cosine), Postgres full-text search, and reciprocal rank fusion
to merge them. Each piece works standalone -- eval/run.py's vector/fulltext/hybrid comparison
needs exactly that, not just the merged result.
"""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

DEFAULT_K = 20


@dataclass(frozen=True)
class Result:
    chunk_id: str
    document_id: str
    heading_path: str
    body: str
    repo: str
    doc_type: str


@dataclass(frozen=True)
class Scope:
    """Optional narrowing of what a query may retrieve from.

    M4's held-out run found a real class of failure this fixes: several questions say "this
    change" without naming which, the corpus holds two near-identical sibling changes, and the
    model correctly refuses because nothing in the passages disambiguates them. A bare question
    string cannot resolve that; the caller knowing which repo or change it is asking about can
    (docs/m4-evaluation-and-scale.md)."""

    repo: str | None = None
    change_id: str | None = None
    # The set of repositories a query may see at all. The eval passes its corpus's repositories
    # here (eval/run.py and friends) because the product and the eval share one store: Paddock's
    # "Refresh the record" can add repositories the eval never labelled, and without this filter
    # a refresh could move a published number without any eval code changing
    # (docs/paddock-gate.md). None means the whole store, which is what the service wants.
    repos: tuple[str, ...] | None = None

    def where_sql(self) -> str:
        clauses = []
        if self.repo:
            clauses.append("d.repo = %(repo)s")
        if self.change_id:
            clauses.append("d.change_id = %(change_id)s")
        if self.repos is not None:
            clauses.append("d.repo = any(%(repos)s)")
        return (" and " + " and ".join(clauses)) if clauses else ""

    def params(self) -> dict:
        params = {}
        if self.repo:
            params["repo"] = self.repo
        if self.change_id:
            params["change_id"] = self.change_id
        if self.repos is not None:
            params["repos"] = list(self.repos)
        return params


EMPTY_SCOPE = Scope()


def retrieve_vector(
    cur: psycopg.Cursor, query_embedding: list[float], k: int = DEFAULT_K, scope: Scope = EMPTY_SCOPE
) -> list[Result]:
    """Nearest neighbours by cosine distance.

    The operator must be `<=>` (cosine), not `<->` (L2): the HNSW index in
    store/migrations/0001_init.sql is built with `vector_cosine_ops`, and pgvector only uses an
    index whose opclass matches the operator in the ORDER BY. This was `<->` until M4's scale
    benchmark caught it -- the planner had been silently falling back to a sequential scan over
    every chunk on every query. Results were unaffected (for unit-normalised vectors, which
    ingest/embed.py guarantees, L2 and cosine rank identically) and at 491 chunks the scan was
    too fast to notice; at 100,000 it was a 21-second p50."""
    cur.execute(
        "select c.id, c.document_id, c.heading_path, c.body, d.repo, d.doc_type "
        "from studbook.chunks c join studbook.documents d on d.id = c.document_id "
        f"where true {scope.where_sql()} "
        "order by c.embedding <=> %(embedding)s::vector limit %(k)s",
        {"embedding": query_embedding, "k": k, **scope.params()},
    )
    return [Result(*row) for row in cur.fetchall()]


def retrieve_fulltext(
    cur: psycopg.Cursor, query_text: str, k: int = DEFAULT_K, scope: Scope = EMPTY_SCOPE
) -> list[Result]:
    """Postgres full-text search against the `fts` generated column (heading_path || body,
    'english' config -- store/migrations/0001_init.sql), ranked by ts_rank.

    Deliberately NOT websearch_to_tsquery/plainto_tsquery: both AND every content word together
    by default. This project's gold questions average 20+ words (eval/dev.jsonl), and requiring
    every one of a natural sentence's content words to co-occur in one ~500-word chunk is nearly
    impossible -- measured recall@5 was 0.0294 with AND semantics on the dev set, a result
    implausible enough that it was investigated rather than accepted (docs/m2-retrieval-and-eval.md
    has the before/after). ORing the question's own stemmed lexemes together is the
    standard lexical-retrieval move (closer to BM25's term-independence than AND/phrase search)
    and is what a hybrid retriever actually needs from its lexical half: any of these words,
    ranked by how many and how rare, not all of these words."""
    cur.execute(
        "with q as ("
        "  select to_tsquery('english', string_agg(lexeme, ' | ')) as tsq"
        "  from unnest(tsvector_to_array(to_tsvector('english', %(q)s))) as lexeme"
        ") "
        "select c.id, c.document_id, c.heading_path, c.body, d.repo, d.doc_type "
        "from studbook.chunks c join studbook.documents d on d.id = c.document_id, q "
        f"where q.tsq is not null and c.fts @@ q.tsq {scope.where_sql()} "
        "order by ts_rank(c.fts, q.tsq) desc limit %(k)s",
        {"q": query_text, "k": k, **scope.params()},
    )
    return [Result(*row) for row in cur.fetchall()]


def reciprocal_rank_fusion(rankings: list[list[Result]], k: int = 10) -> list[Result]:
    """Standard RRF: score(d) = sum over rankings of 1/(k + rank). A result appearing near the top
    of either ranking, or moderately in both, outranks one that's merely first in a single list --
    the usual reason to prefer RRF over picking a single retriever or hand-weighting scores that
    live on different scales (cosine distance vs. ts_rank).

    k=10, not the textbook-common 60: M2's first tuning loop measured recall@5 on the dev set at
    k=10 (0.7353), k=30 (0.6471) and k=60 (0.6471) -- k=10 won outright, not just on recall@5 but
    on mrr@10 and precision@5 too (docs/m2-retrieval-and-eval.md has the full table). A smaller k
    concentrates RRF's weight on whichever ranking(s) actually put a result near the top, which
    fits a corpus this small (~600 chunks): a genuinely good match tends to already be near the
    top of at least one of the two rankings, so diluting toward the tail (larger k) mostly adds
    noise rather than rescuing borderline results."""
    scores: dict[str, float] = {}
    by_id: dict[str, Result] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            scores[result.chunk_id] = scores.get(result.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[result.chunk_id] = result
    return [by_id[cid] for cid in sorted(scores, key=lambda cid: -scores[cid])]


def retrieve_hybrid(
    cur: psycopg.Cursor,
    query_text: str,
    query_embedding: list[float],
    top_k: int = 5,
    vector_k: int = DEFAULT_K,
    fulltext_k: int = DEFAULT_K,
    rrf_k: int = 10,
    scope: Scope = EMPTY_SCOPE,
) -> list[Result]:
    vector_results = retrieve_vector(cur, query_embedding, vector_k, scope)
    fulltext_results = retrieve_fulltext(cur, query_text, fulltext_k, scope)
    fused = reciprocal_rank_fusion([vector_results, fulltext_results], k=rrf_k)
    return fused[:top_k]

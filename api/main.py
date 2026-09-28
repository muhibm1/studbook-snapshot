"""The Studbook service: retrieval and cited answers over the engineering record.

    .venv/Scripts/python.exe -m uvicorn api.main:app --port 8000

    GET  /healthz        corpus counts and which models are loaded
    GET  /repos          repos and change ids, for the UI's scope filter
    GET  /search?q=...   retrieval only -- no model call, so no per-query cost
    POST /ask            retrieval + a cited answer
    GET  /               the one-page UI

Needs STUDBOOK_DATABASE_URL and STUDBOOK_API_KEY; /ask additionally needs ANTHROPIC_API_KEY.

Auth: /repos, /search and /ask require `Authorization: Bearer <STUDBOOK_API_KEY>`. The corpus
holds client-shaped material and named people, so the data routes are never open, and the server
refuses to start without a key rather than quietly running without one -- a failure that is both
silent and consequential is the one shape this project's security baseline forbids. /healthz
(counts and model names, nothing from the record) and the static page at / stay open; the page
itself cannot fetch anything until a key is entered.

The embedding and reranker models load once at startup (several seconds each) and are reused for
every request -- loading per request would dominate latency. The database connection is opened
per request instead: it is cheap next to the model work, and a long-lived connection through
Supabase's pooler is the thing most likely to go stale between requests.
"""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pgvector.psycopg import register_vector
from pydantic import BaseModel, Field

from answer.generate import DEFAULT_MODEL, generate_answer
from ingest.embed import Embedder
from retrieve.hybrid import Result, Scope, retrieve_hybrid
from retrieve.rerank import Reranker
from store.connect import connect as db_connect, prepare_session

STATIC_DIR = Path(__file__).resolve().parent / "static"
RERANK_CANDIDATES = 10  # see docs/retrieval-ceiling.md: same quality as 20, half the rerank
DEFAULT_TOP_K = 5
API_KEY_ENV = "STUDBOOK_API_KEY"
MIN_API_KEY_LENGTH = 32  # secrets.token_urlsafe(32) gives 43; anything shorter is a typo or a test

models: dict = {}


def configured_api_key() -> str:
    """The key every data route is checked against. Raises rather than returning something
    permissive: an unset key must stop the server, not open it."""
    key = os.environ.get(API_KEY_ENV, "")
    if len(key) < MIN_API_KEY_LENGTH:
        raise RuntimeError(
            f"{API_KEY_ENV} is not set (or is under {MIN_API_KEY_LENGTH} characters). The data routes "
            f"are never served without one. Generate a key with\n"
            f"    python -c \"import secrets; print(secrets.token_urlsafe(32))\"\n"
            f"and set it as a user environment variable; the page asks for it once per tab."
        )
    return key


def require_api_key(authorization: str | None = Header(default=None)) -> None:
    """FastAPI dependency for the data routes. Constant-time comparison so a wrong key cannot be
    narrowed down byte by byte from response timing; the same 401 for a missing header, a
    malformed one and a wrong key, so a response never says which it was."""
    expected = configured_api_key()
    scheme, _, presented = (authorization or "").partition(" ")
    if scheme.lower() != "bearer" or not secrets.compare_digest(presented.strip(), expected):
        raise HTTPException(
            status_code=401,
            detail="a valid API key is required (Authorization: Bearer <key>)",
            headers={"WWW-Authenticate": "Bearer"},
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    configured_api_key()  # fail here, before spending seconds loading models the server will not serve
    models["embedder"] = Embedder()
    models["reranker"] = Reranker()
    yield
    models.clear()


app = FastAPI(title="Studbook", version="0.6.0", lifespan=lifespan)
protected = [Depends(require_api_key)]

# On every response. The page holds the API key in sessionStorage, which any script running in it
# can read, so the CSP is what stands between an injected script and the key: scripts and styles
# only from this origin (the page's own were moved out of inline blocks into /static so no
# 'unsafe-inline' is needed), requests only back to this origin, and no framing. No HSTS: the
# service is plain HTTP on localhost, where HSTS is meaningless at best; add it with TLS if the
# service is ever put behind one. No CORS middleware on purpose -- other origins stay blocked.
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
        "connect-src 'self'; object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=(), payment=(), usb=()",
    "X-Frame-Options": "DENY",
}


@app.middleware("http")
async def security_headers(request, call_next):
    response = await call_next(request)
    response.headers.update(SECURITY_HEADERS)
    return response


def connect() -> psycopg.Connection:
    url = os.environ.get("STUDBOOK_DATABASE_URL")
    if not url:
        raise HTTPException(status_code=503, detail="STUDBOOK_DATABASE_URL is not set on the server")
    conn = db_connect(url, autocommit=True)
    with conn.cursor() as cur:
        prepare_session(cur)
    register_vector(conn)
    return conn


class Passage(BaseModel):
    number: int | None = None
    chunk_id: str
    heading_path: str
    repo: str
    doc_type: str
    body: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    repo: str | None = None
    change_id: str | None = None
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20)


class Citation(BaseModel):
    number: int
    chunk_id: str
    heading_path: str
    repo: str
    doc_type: str
    body: str


class AskResponse(BaseModel):
    question: str
    answer: str
    refused: bool
    citations: list[Citation]
    passages: list[Passage]
    model: str
    input_tokens: int
    output_tokens: int


def to_passages(results: list[Result]) -> list[Passage]:
    return [
        Passage(number=i, chunk_id=r.chunk_id, heading_path=r.heading_path, repo=r.repo,
                doc_type=r.doc_type, body=r.body)
        for i, r in enumerate(results, start=1)
    ]


def retrieve(conn: psycopg.Connection, question: str, scope: Scope, top_k: int) -> list[Result]:
    embedding = models["embedder"].embed([question])[0]
    with conn.cursor() as cur:
        candidates = retrieve_hybrid(cur, question, embedding, top_k=RERANK_CANDIDATES, scope=scope)
    return models["reranker"].rerank(question, candidates, top_k=top_k)


@app.get("/healthz")
def healthz() -> dict:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("select count(*) from studbook.documents")
        (documents,) = cur.fetchone()
        cur.execute("select count(*) from studbook.chunks")
        (chunks,) = cur.fetchone()
    return {
        "status": "ok",
        "documents": documents,
        "chunks": chunks,
        "generator_model": DEFAULT_MODEL,
        "models_loaded": sorted(models),
    }


@app.get("/repos", dependencies=protected)
def repos() -> dict:
    """What the UI's scope filter offers. Change ids are grouped per repo so picking a repo can
    narrow the change list without a second round trip."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(
            "select d.repo, d.change_id, count(*) from studbook.documents d "
            "group by d.repo, d.change_id order by d.repo, d.change_id nulls first"
        )
        rows = cur.fetchall()
    grouped: dict[str, list[dict]] = {}
    for repo, change_id, count in rows:
        grouped.setdefault(repo, [])
        if change_id:
            grouped[repo].append({"change_id": change_id, "documents": count})
    return {"repos": [{"repo": r, "changes": c} for r, c in sorted(grouped.items())]}


@app.get("/search", dependencies=protected)
def search(
    q: str = Query(min_length=1, max_length=2000),
    repo: str | None = None,
    change_id: str | None = None,
    k: int = Query(default=DEFAULT_TOP_K, ge=1, le=20),
) -> dict:
    """Retrieval only -- no model call, so this costs nothing per query beyond the local
    embedding and rerank. Useful on its own, and the cheap way to see what /ask would be given."""
    with connect() as conn:
        results = retrieve(conn, q, Scope(repo=repo, change_id=change_id), k)
    return {"query": q, "passages": to_passages(results)}


@app.post("/ask", dependencies=protected)
def ask(request: AskRequest) -> AskResponse:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        raise HTTPException(status_code=503, detail="ANTHROPIC_API_KEY is not set on the server")
    with connect() as conn:
        results = retrieve(conn, request.question, Scope(repo=request.repo, change_id=request.change_id), request.top_k)
    answer = generate_answer(request.question, results)
    by_number = {p.number: p for p in to_passages(results)}
    return AskResponse(
        question=request.question,
        answer=answer.text,
        refused=answer.refused,
        citations=[
            Citation(number=c.number, chunk_id=c.chunk_id, heading_path=c.heading_path,
                     repo=c.repo, doc_type=c.doc_type, body=by_number[c.number].body)
            for c in answer.citations
        ],
        passages=to_passages(results),
        model=answer.model,
        input_tokens=answer.input_tokens,
        output_tokens=answer.output_tokens,
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

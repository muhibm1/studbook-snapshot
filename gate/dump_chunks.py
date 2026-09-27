"""Python side of the chunk parity gate: what Paddock's TypeScript parser + chunker must reproduce.

Walks every repo in corpus.json exactly as ingest/sync.py does and writes each document's
identity fields and every chunk's id, heading path, gate, token count and content hash. The TS
gate (desk/tests/studbook.chunks.gate.test.ts) walks the same working trees and compares; a
single differing id or hash means the two ingesters would fight over the store.

    .venv/Scripts/python.exe gate/dump_chunks.py   # writes gate/python_chunks.json
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.gold import load_corpus  # noqa: E402
from ingest.chunker import chunk_document  # noqa: E402
from ingest.parser import walk_repo  # noqa: E402



def tree_state(repo_path: Path) -> dict[str, str]:
    """What the dump was taken against: HEAD, and a hash of `git status --porcelain` (edits and
    untracked files). The corpus includes commit messages and the working tree, so a commit or an
    edit after the dump changes what both ingesters see; the TS gate compares this first and reports
    a stale dump as stale, instead of as a parity failure (which it was mistaken for twice)."""
    def git(*args: str) -> bytes:
        return subprocess.run(["git", "-C", str(repo_path), *args], capture_output=True, check=True).stdout
    return {"head": git("rev-parse", "HEAD").decode().strip(),
            "status_sha256": hashlib.sha256(git("status", "--porcelain")).hexdigest()}


out: dict[str, dict] = {}
for repo, repo_path in load_corpus().items():
    docs = walk_repo(repo, repo_path)
    out[repo] = {
        "path": str(repo_path),
        "tree": tree_state(repo_path),
        "documents": [
            {
                "id": d.id, "path": d.path, "doc_type": d.doc_type, "change_id": d.change_id,
                "tier": d.tier, "doc_date": d.doc_date, "commit_sha": d.commit_sha,
                "content_hash": d.content_hash,
            }
            for d in docs
        ],
        "chunks": [
            {
                "id": c.id, "heading_path": c.heading_path, "gate": c.gate,
                "token_count": c.token_count, "content_hash": c.content_hash,
            }
            for d in docs
            for c in chunk_document(d)
        ],
    }
    print(f"{repo}: {len(out[repo]['documents'])} documents, {len(out[repo]['chunks'])} chunks")

path = ROOT / "gate" / "python_chunks.json"
path.write_text(json.dumps(out), encoding="utf-8")
print(f"wrote {path}")

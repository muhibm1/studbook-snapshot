"""Walks a repository into `ParsedDocument`s: one per markdown file under docs/sdlc (excluding
verify-logs/, per docs/m0-census.md section 1) and the top-level README, plus one per commit in
`git log`. Pure with respect to the filesystem it reads -- no network, no DB.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# Filename (without .md) -> doc_type, for files directly under a change folder or a repo's
# docs/sdlc root. adr/*.md, reviews/*.md and docs/superpowers/{specs,plans}/*.md are handled
# separately below. 'intent' and 'review-packet' are the documents WorkHorse wrote before 0.3.0
# replaced them with 'brief' and 'ship' (docs/superpowers/specs/2026-09-19-two-gates-design.md
# section 4); older changes keep them, so both generations are ingested.
DOC_TYPE_BY_STEM = {
    "intent": "intent",
    "brief": "brief",
    "ship": "ship",
    "spec": "spec",
    "plan": "plan",
    "evals": "evals",
    "verification": "verification",
    "review-packet": "review-packet",
    "release": "release",
    "approvals": "approvals",
    "retro": "retro",
    "conductor-log": "conductor-log",
    "codebase-map": "codebase-map",
    "constraints": "constraints",
}

DATE_LINE = re.compile(r"^Date:\s*(\d{4}-\d{2}-\d{2})", re.MULTILINE)
TIER_LINE = re.compile(r"^Risk tier:\s*(\d)", re.MULTILINE)
# The documents that state a change's tier, in the order they are consulted: the first one that
# carries a TIER_LINE decides. brief.md replaced intent.md in WorkHorse 0.3.0.
TIER_SOURCES = ("brief", "intent")


@dataclass(frozen=True)
class ParsedDocument:
    id: str  # '<repo>/<path>' for a file, '<repo>/git:<sha>' for a commit
    repo: str
    path: str  # repo-relative path with forward slashes, or 'git:<sha>'
    doc_type: str
    change_id: str | None
    tier: int | None
    doc_date: str | None  # 'YYYY-MM-DD'
    commit_sha: str  # repo HEAD at ingest (a file) or the commit's own sha (a commit)
    text: str

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _doc_type_for(rel_path: Path) -> str | None:
    """None means "skip this file" -- verify-logs/, and anything not in the doc-type table."""
    parts = rel_path.parts
    if "verify-logs" in parts:
        return None
    if rel_path.name == "README.md" and len(parts) == 1:
        return "readme"
    if rel_path.name == "CHANGELOG.md" and len(parts) <= 2:
        # The root changelog, or a component's (desk/CHANGELOG.md). Each entry names the release it
        # follows, which is the fact wh-15 kept getting wrong (docs/refusal-experiments.md).
        return "changelog"
    if len(parts) == 5 and parts[:2] == ("docs", "sdlc") and parts[3] == "reviews" and rel_path.suffix == ".md":
        # docs/sdlc/<id>/reviews/<agent>.md: one report per reviewer since WorkHorse 0.3.0.
        return "review"
    if "adr" in parts and rel_path.suffix == ".md":
        return "adr"
    if parts[:2] == ("docs", "superpowers") and rel_path.suffix == ".md":
        return "design-doc"
    stem = rel_path.stem
    return DOC_TYPE_BY_STEM.get(stem)


def _change_id_for(rel_path: Path, doc_type: str) -> str | None:
    parts = rel_path.parts
    if len(parts) >= 3 and parts[0] == "docs" and parts[1] == "sdlc" and doc_type not in ("codebase-map", "constraints"):
        return parts[2]
    return None


def _git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_path), *args], capture_output=True, text=True, encoding="utf-8", check=True
    )
    return result.stdout


def _repo_head(repo_path: Path) -> str:
    return _git(repo_path, "rev-parse", "HEAD").strip()


def walk_files(repo: str, repo_path: Path) -> list[ParsedDocument]:
    """Every markdown file this corpus wants, from docs/sdlc/** and the repo root README, plus
    (for a client repo like paddock-demo) docs/superpowers design docs when present."""
    head = _repo_head(repo_path)
    docs: list[ParsedDocument] = []

    # First pass: read each change's "Risk tier: N" line so it can be applied to every other
    # document in the same change. Only one document states the tier explicitly: intent.md before
    # WorkHorse 0.3.0 (docs/m0-census.md), brief.md since. brief.md is read first, so its tier
    # wins when both exist and state one; intent.md's is used when brief.md states none.
    tier_by_change: dict[str, int] = {}
    sdlc_root = repo_path / "docs" / "sdlc"
    if sdlc_root.is_dir():
        for change_dir in sorted(p for p in sdlc_root.iterdir() if p.is_dir()):
            for stem in TIER_SOURCES:
                tier_file = change_dir / f"{stem}.md"
                if not tier_file.is_file():
                    continue
                m = TIER_LINE.search(tier_file.read_text(encoding="utf-8"))
                if m:
                    tier_by_change[change_dir.name] = int(m.group(1))
                    break

    candidates: list[Path] = []
    if sdlc_root.is_dir():
        candidates += sorted(sdlc_root.rglob("*.md"))
    readme = repo_path / "README.md"
    if readme.is_file():
        candidates.append(readme)
    candidates += [p for p in [repo_path / "CHANGELOG.md", *sorted(repo_path.glob("*/CHANGELOG.md"))]
                   if p.is_file() and not any(part.startswith(".") or part == "node_modules"
                                              for part in p.relative_to(repo_path).parts)]
    design_root = repo_path / "docs" / "superpowers"
    if design_root.is_dir():
        candidates += sorted(design_root.rglob("*.md"))

    for file_path in candidates:
        rel_path = file_path.relative_to(repo_path)
        doc_type = _doc_type_for(rel_path)
        if doc_type is None:
            continue
        text = file_path.read_text(encoding="utf-8").replace("\r\n", "\n")
        change_id = _change_id_for(rel_path, doc_type)
        date_match = DATE_LINE.search(text)
        docs.append(
            ParsedDocument(
                id=f"{repo}/{rel_path.as_posix()}",
                repo=repo,
                path=rel_path.as_posix(),
                doc_type=doc_type,
                change_id=change_id,
                tier=tier_by_change.get(change_id) if change_id else None,
                doc_date=date_match.group(1) if date_match else None,
                commit_sha=head,
                text=text,
            )
        )
    return docs


# Separates commits in `git log`'s output; chosen to never plausibly occur inside a commit
# message, unlike a fixed short marker.
_COMMIT_SEP = "\x1e"
_FIELD_SEP = "\x1f"


def walk_commits(repo: str, repo_path: Path) -> list[ParsedDocument]:
    """One ParsedDocument per commit, doc_type 'commit', text = the full commit message (subject
    + body). No change_id or tier: a commit isn't scoped to one docs/sdlc/<change> folder."""
    log = _git(
        repo_path, "log", f"--format={_FIELD_SEP}%H{_FIELD_SEP}%ad{_FIELD_SEP}%B{_COMMIT_SEP}", "--date=short"
    )
    docs: list[ParsedDocument] = []
    # git's tformat (the implicit mode of a plain --format=...) appends a "\n" after every
    # record, including after this format string's own trailing _COMMIT_SEP -- so each entry
    # after the first carries a leading "\n" left over from the commit before it, and the final
    # split() produces one trailing entry that is nothing but that newline.
    for entry in log.split(_COMMIT_SEP):
        entry = entry.lstrip("\n")
        if not entry:
            continue
        _, sha, date, message = entry.split(_FIELD_SEP)
        message = message.strip("\n")
        if not message:
            continue
        docs.append(
            ParsedDocument(
                id=f"{repo}/git:{sha}",
                repo=repo,
                path=f"git:{sha}",
                doc_type="commit",
                change_id=None,
                tier=None,
                doc_date=date,
                commit_sha=sha,
                text=message,
            )
        )
    return docs


def walk_repo(repo: str, repo_path: Path) -> list[ParsedDocument]:
    return walk_files(repo, repo_path) + walk_commits(repo, repo_path)

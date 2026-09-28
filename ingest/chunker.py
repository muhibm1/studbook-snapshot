"""Splits a ParsedDocument into retrievable Chunks: one per H2 (or, for an oversized H2, per H3)
heading section, each with the enclosing heading path prepended. A heading-less document (a git
commit message, or a conductor-log.md -- docs/m0-census.md section 2) becomes one "section" that
the same oversized-section path (paragraph windowing) handles, never splitting a markdown table's
rows across a window boundary unless the table alone is over budget, in which case each window
after the first repeats the table's header and separator rows.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ingest.parser import ParsedDocument

# ~500-800 tokens per the plan; 600 words is comfortably inside that at ~1.3 tokens/word. Real
# corpus sections average 110-350 words (docs/m0-census.md section 2) and stay well under this;
# it mainly matters for conductor-log.md (no headings, ~1,288 words/file) and as a safety valve.
MAX_CHUNK_WORDS = 600

_H1 = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_H2 = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_H3 = re.compile(r"^###\s+(.+?)\s*$", re.MULTILINE)
_TABLE_SEPARATOR = re.compile(r"^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$")


@dataclass(frozen=True)
class Chunk:
    id: str
    document_id: str
    ordinal: int
    heading_path: str
    gate: str | None
    body: str

    @property
    def token_count(self) -> int:
        # Approximate, not a real tokenizer -- adequate for a schema/ops column, not for billing.
        return max(1, round(len(self.body.split()) * 1.3))

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.body.encode("utf-8")).hexdigest()


_GATE = re.compile(r"\bG([0-5])\b")


def _infer_gate(heading_path: str) -> str | None:
    m = _GATE.search(heading_path)
    return f"G{m.group(1)}" if m else None


def _split_by_level(text: str, pattern: re.Pattern[str]) -> list[tuple[str | None, str]]:
    """(heading title or None for the leading unheaded part, section body) pairs, in order."""
    matches = list(pattern.finditer(text))
    if not matches:
        return [(None, text)]
    sections = []
    if matches[0].start() > 0:
        sections.append((None, text[: matches[0].start()]))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        sections.append((m.group(1), text[m.end() : end]))
    return sections


def _is_table_line(line: str) -> bool:
    return line.lstrip().startswith("|")


def _window_paragraphs(body: str, max_words: int) -> list[str]:
    """Blank-line-delimited blocks, grouped into windows up to `max_words`. A block that is a
    markdown table and itself exceeds `max_words` is split by row instead, repeating its header
    and separator row (the first two lines) in every continuation window -- never mid-table
    otherwise, since a block (having no blank line inside it) is never split any other way."""
    blocks = [b for b in re.split(r"\n\s*\n", body.strip("\n")) if b.strip()]
    windows: list[str] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        if current:
            windows.append("\n\n".join(current))
        current, current_words = [], 0

    for block in blocks:
        block_words = len(block.split())
        lines = block.splitlines()
        is_table = len(lines) >= 2 and _is_table_line(lines[0]) and bool(_TABLE_SEPARATOR.match(lines[1]))

        if is_table and block_words > max_words:
            flush()
            header, separator, rows = lines[0], lines[1], lines[2:]
            row_words = 0
            row_group: list[str] = []
            for row in rows:
                row_words += len(row.split())
                row_group.append(row)
                if row_words >= max_words:
                    windows.append("\n".join([header, separator, *row_group]))
                    row_words, row_group = 0, []
            if row_group:
                windows.append("\n".join([header, separator, *row_group]))
            continue

        if current_words + block_words > max_words and current:
            flush()
        current.append(block)
        current_words += block_words

    flush()
    return windows or [""]


def chunk_document(doc: ParsedDocument, max_words: int = MAX_CHUNK_WORDS) -> list[Chunk]:
    h1_match = _H1.search(doc.text)
    h1_title = h1_match.group(1) if h1_match else None
    body_after_h1 = doc.text[h1_match.end() :] if h1_match else doc.text

    chunks: list[Chunk] = []
    ordinal = 0

    def add(heading_path: str, body: str) -> None:
        nonlocal ordinal
        body = body.strip()
        if not body:
            return
        chunks.append(
            Chunk(
                id=f"{doc.id}#{ordinal}",
                document_id=doc.id,
                ordinal=ordinal,
                heading_path=heading_path,
                gate=_infer_gate(heading_path),
                body=body,
            )
        )
        ordinal += 1

    def path(*parts: str | None) -> str:
        return " > ".join(p for p in parts if p)

    for h2_title, h2_body in _split_by_level(body_after_h1, _H2):
        h2_path = path(h1_title, h2_title)
        if len(h2_body.split()) <= max_words:
            add(h2_path or "Document", h2_body)
            continue
        for h3_title, h3_body in _split_by_level(h2_body, _H3):
            h3_path = path(h1_title, h2_title, h3_title)
            if len(h3_body.split()) <= max_words:
                add(h3_path or h2_path or "Document", h3_body)
            else:
                for window in _window_paragraphs(h3_body, max_words):
                    add(h3_path or h2_path or "Document", window)

    return chunks

"""Hypothetical passage search (HyDE): search with a passage written the way the record would
answer the question, not only with the question.

docs/cross-doc.md found that version-14's documents are reachable by full text -- "verification npm
test pass fail" puts them at #5 and #1 -- but by words that are not synonyms of anything in its
question. Getting there takes knowing where and how this record states a test suite's size. A
model asked to write the passage that would answer the question, in the record's own register, may
reach those words where synonym expansion could not.

The passage is used only to retrieve. It is never shown and never reaches the generator, so what
it invents cannot become part of an answer; its only power is over which real passages are
fetched. The prompt describes the record's document types in general and says nothing about test
counts or any particular question -- anything more specific would be fitting dev.

MEASURED AND REJECTED -- not used in production. On dev the original formulation (the passage's
embedding replacing the question's) held coverage@5 at 27/34 by gaining two questions and losing
two; adding the passage as extra rankings lowered it (docs/cross-doc.md).
"""

from __future__ import annotations

import sys

import anthropic

from retrieve.rewrite import REWRITE_MODEL

MAX_TOKENS = 250

HYPOTHETICAL_PROMPT = (
    "You write search bait for an engineering record. The record holds a software team's specs, "
    "architecture decision records, eval plans, verification reports with command output and logs, "
    "review packets, release notes, retrospectives and commit messages.\n\n"
    "Given a question, write the short passage (60 to 120 words) that would answer it if it "
    "appeared in that record -- in the record's own style: the document it would sit in, its "
    "headings, the commands, file names and terms such a document would use. Invent "
    "specific values freely; they are never shown to anyone and are only used to find the real "
    "passage. Reply with the passage alone."
)


def hypothetical_passage(question: str, client: anthropic.Anthropic | None = None, model: str = REWRITE_MODEL) -> str:
    """The passage, or "" if the call fails -- a failed call means no extra ranking, not a failed
    question."""
    try:
        response = (client or anthropic.Anthropic()).messages.create(
            model=model, max_tokens=MAX_TOKENS, system=HYPOTHETICAL_PROMPT,
            messages=[{"role": "user", "content": question}],
            extra_body={"temperature": 0.0},
        )
    except anthropic.APIError as e:
        print(f"note: hypothetical passage failed ({type(e).__name__}); searching the question only", file=sys.stderr)
        return ""
    return "".join(b.text for b in response.content if b.type == "text").strip()

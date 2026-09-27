"""Turn a follow-up into a standalone question before retrieving for it.

Two measurements bracket the alternative, concatenating the previous questions onto the follow-up
(docs/follow-ups.md): it takes genuine follow-ups from 2/12 to 8/12, and costs a change of subject
29/34 -> 20/34. A concatenation cannot tell the two apart. A rewrite can: "what does it do
instead?" becomes a question that names "it", and a question that already names its subject comes
back unchanged.

A rewrite is also a new place for a subject to be invented, which is the one failure this project
exists to prevent. So a rewrite is only used if every content word in it already appears in the
thread's questions (`unsupported_words`): it may rearrange and pronoun-resolve, it may not add. A
rewrite that fails that check, or a call that fails, falls back to the concatenation -- the
behaviour that was already there -- and says so in `Rewrite.method`, so a caller can show it.

Only the earlier *questions* are given to the rewriter, not the answers. That is what the eval
can measure (eval/rewrite.py has questions, not generated answers), and the guard's vocabulary is
then entirely words a person typed.

Mirrored in Paddock (desk/src/main/studbook/rewrite.ts). The prompt and the guard's word lists are
pinned by hash on both sides (tests/test_rewrite.py, desk/tests/studbook.rewrite.test.ts), so an
edit to one that is not made to the other fails a test.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

import anthropic

REWRITE_MODEL = "claude-haiku-4-5"
MAX_TOKENS = 200
MAX_QUERY_CHARS = 500
HISTORY_TURNS = 2  # the same window the concatenation carried
# Five characters is enough for a crude stem: "broke"/"broken", "deploy"/"deployed", and short
# enough that the rewriter's inflection of a word the user typed is not mistaken for a new one.
STEM_CHARS = 5
SUFFIXES = ("ing", "ed", "es", "en", "e", "s")  # longest first; one is removed at most

REWRITE_PROMPT = (
    "You rewrite a follow-up question from a conversation about a software team's engineering "
    "record into one standalone question, for a search engine that cannot see the conversation.\n\n"
    "Rules:\n"
    "- If the new question depends on an earlier one for its subject -- it says \"it\", \"that\", "
    "\"they\", \"the fix\", \"instead\", or otherwise leaves out what it is about -- rewrite it as "
    "one question that names that subject, using the earlier questions' own words for it.\n"
    "- If the new question already names its own subject, or asks about something different from "
    "the earlier questions, return it exactly as written.\n"
    "- Never add a name, version, number, tool, file or detail that does not appear in the earlier "
    "questions or the new question. Never answer the question.\n"
    "- Reply with the question alone, on one line."
)

# Words a rewrite may introduce freely: grammar, and the interrogatives a standalone question
# needs. Nothing here names a subject, so nothing here can smuggle one in.
FUNCTION_WORDS = frozenset(
    "a about after again against all also an and any are as at be because been before being both "
    "but by can could did do does doing done during each else for from had has have having how if "
    "in instead into is it its itself just made make more most much not now of off on once only or "
    "other our out over own same should so some such than that the their them then there these "
    "they this those through to too under until up upon use used very was were what when where "
    "whether which while who whom whose why will with within without would".split()
)

_WORD = re.compile(r"[a-z0-9][a-z0-9_./-]*")


@dataclass(frozen=True)
class Rewrite:
    """What retrieval searched for, and how it got there. `method` is one of:
    "fresh" (no history: the question itself), "rewritten", "unchanged" (the rewriter judged it
    standalone), "fallback-guard" (the rewrite added words the thread does not contain),
    "fallback-error" (the call failed). Both fallbacks search the concatenation."""

    query: str
    method: str
    rejected: str = ""


def _words(text: str) -> list[str]:
    return [w.strip("./-") for w in _WORD.findall(text.lower())]


def stem(word: str) -> str:
    """Crude on purpose: drop one inflectional suffix, then keep a prefix. It only has to decide
    "is this a word the user typed, inflected?" -- "moved"/"move", "broken"/"broke",
    "deploying"/"deployed" -- not be a linguist."""
    for suffix in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)]
            break
    # Undouble a final consonant, always, so "pinning" -> "pinn" -> "pin" meets "pin", and "stopping"
    # meets "stop". Both were guard false positives in eval/rewrite.py before this line existed.
    # Applied whether or not a suffix came off, so a word and its inflections still agree.
    if len(word) >= 3 and word[-1] == word[-2] and word[-1] not in "aeiou":
        word = word[:-1]
    return word[:STEM_CHARS]


def unsupported_words(rewrite: str, sources: list[str]) -> list[str]:
    """Content words in `rewrite` that no source contains, compared on a crude stem."""
    known = {stem(w) for s in sources for w in _words(s) if w}
    return [w for w in _words(rewrite)
            if w and w not in FUNCTION_WORDS and len(w) > 2 and stem(w) not in known]


def carried(question: str, history: list[str]) -> str:
    return " ".join([*history[-HISTORY_TURNS:], question])


def build_rewrite_request(question: str, history: list[str]) -> str:
    earlier = "\n".join(f"{i}. {q}" for i, q in enumerate(history[-HISTORY_TURNS:], start=1))
    return f"Earlier questions in this thread, oldest first:\n{earlier}\n\nNew question: {question}"


def clean(text: str) -> str:
    line = text.strip().splitlines()[0].strip() if text.strip() else ""
    return line.strip('"“”').strip()[:MAX_QUERY_CHARS]


def rewrite_query(
    question: str,
    history: list[str],
    client: anthropic.Anthropic | None = None,
    model: str = REWRITE_MODEL,
) -> Rewrite:
    """`history` is the thread's earlier questions, oldest first."""
    if not history:
        return Rewrite(question, "fresh")
    fallback = carried(question, history)
    try:
        response = (client or anthropic.Anthropic()).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=REWRITE_PROMPT,
            messages=[{"role": "user", "content": build_rewrite_request(question, history)}],
            # Pinned at 0 for the same reason as generation (answer/generate.py): the same thread
            # should search for the same thing. Via extra_body because the SDK's typed signature
            # dropped the keyword in 1.6.0; Haiku 4.5 honours it.
            extra_body={"temperature": 0.0},
        )
    except anthropic.APIError as e:
        print(f"note: query rewrite failed ({type(e).__name__}); searching the concatenation", file=sys.stderr)
        return Rewrite(fallback, "fallback-error")

    text = clean("".join(b.text for b in response.content if b.type == "text"))
    if not text:
        return Rewrite(fallback, "fallback-error")
    if text == question.strip():
        return Rewrite(question, "unchanged")
    added = unsupported_words(text, [question, *history[-HISTORY_TURNS:]])
    if added:
        return Rewrite(fallback, "fallback-guard", rejected=text)
    return Rewrite(text, "rewritten")

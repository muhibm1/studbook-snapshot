"""MEASURED AND REJECTED -- not used in production. Kept, with its eval (eval/decompose.py), so the
negative result is reproducible: on dev it lowered coverage@5 from 27/34 to 23/34 and did not lift
cross-doc (docs/cross-doc.md). The reasoning below is the hypothesis it tested.

Split a question that spans several records into one lookup per part, and give each part its
own place in the passages the generator sees.

Cross-doc is the weakest category, and the cause is coverage, not recall (docs/cross-doc.md): a
question like "how did /version's test suite compare to /health's?" needs a passage from each
change, one rerank against the whole question ranks one half above the other, and every cross-doc
question on dev given only one of its two documents was refused. Of the six gold documents dev
loses, two were never retrieved by the question's wording, two fell below the rerank pool, and two
were reranked to just below the cut (eval/coverage.py). A single ranking cannot fix any of those;
one ranking per part can.

So: Haiku splits the question into standalone sub-questions, or returns it unchanged when it asks
about one thing -- most questions, which then retrieve exactly as before. Each part is retrieved and
reranked against itself, and the final passages are taken round-robin across the parts, so each part
is guaranteed a place instead of competing for one.

A sub-question is a new query the user did not type, so it gets the same guard as a follow-up
rewrite (retrieve/rewrite.py): every content word must already be in the question. A part that
fails is dropped; if none survive, or the call fails, the question is retrieved whole.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field

import anthropic

from retrieve.rewrite import REWRITE_MODEL, clean, unsupported_words

DECOMPOSE_MODEL = REWRITE_MODEL
MAX_TOKENS = 300
MAX_PARTS = 3

DECOMPOSE_PROMPT = (
    "You prepare questions about a software team's engineering record for a search engine that "
    "finds one topic at a time.\n\n"
    "If answering the question needs facts about two or more different things that would be recorded "
    "in different places -- two changes, two releases, a plan and what happened later, a finding and "
    "what was decided about it -- write one standalone question per thing, at most three, one per "
    "line, using only the question's own words.\n\n"
    "If the question is about one thing, reply with the question exactly as written, on one line.\n\n"
    "Never add a name, version, number, tool, file or detail that is not in the question. Never "
    "answer it. Reply with the question or questions alone."
)


@dataclass(frozen=True)
class Decomposition:
    """`parts` is what retrieval searches for, one ranking each. `method`: "whole" (not split),
    "split", "fallback-guard" (every proposed part added words), "fallback-error" (the call
    failed). `rejected` holds parts the guard dropped."""

    parts: list[str]
    method: str
    rejected: list[str] = field(default_factory=list)


def parse_parts(text: str) -> list[str]:
    parts = []
    for line in text.strip().splitlines():
        line = clean(line.lstrip("-*0123456789.) \t"))
        if line and line not in parts:
            parts.append(line)
    return parts[:MAX_PARTS]


def decompose(question: str, client: anthropic.Anthropic | None = None, model: str = DECOMPOSE_MODEL) -> Decomposition:
    try:
        response = (client or anthropic.Anthropic()).messages.create(
            model=model,
            max_tokens=MAX_TOKENS,
            system=DECOMPOSE_PROMPT,
            messages=[{"role": "user", "content": question}],
            extra_body={"temperature": 0.0},  # as retrieve/rewrite.py: via extra_body since SDK 1.6.0
        )
    except anthropic.APIError as e:
        print(f"note: decomposition failed ({type(e).__name__}); retrieving the question whole", file=sys.stderr)
        return Decomposition([question], "fallback-error")

    parts = parse_parts("".join(b.text for b in response.content if b.type == "text"))
    if not parts:
        return Decomposition([question], "fallback-error")
    if len(parts) == 1:
        # One part is "not split", whatever its wording: a single rephrasing is not what this is for,
        # and retrieving the question as typed keeps single-topic questions exactly as they were.
        return Decomposition([question], "whole")
    kept = [p for p in parts if not unsupported_words(p, [question])]
    rejected = [p for p in parts if p not in kept]
    if len(kept) < 2:
        return Decomposition([question], "fallback-guard", rejected)
    return Decomposition(kept, "split", rejected)


def interleave(rankings: list[list], top_k: int, key=lambda r: r.chunk_id) -> list:
    """Round-robin across per-part rankings, best first, skipping duplicates: each part is
    guaranteed a place in the first len(rankings) slots, and no passage appears twice."""
    out, seen = [], set()
    depth = max((len(r) for r in rankings), default=0)
    for i in range(depth):
        for ranking in rankings:
            if i < len(ranking) and key(ranking[i]) not in seen:
                seen.add(key(ranking[i]))
                out.append(ranking[i])
                if len(out) == top_k:
                    return out
    return out

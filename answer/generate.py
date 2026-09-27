"""Generation with numbered citations and refusal, over the top-k hybrid retrieval results.
Haiku 4.5 by default -- cheap per query, the plan's default (docs/m0-census.md); pass a `model`
to use Sonnet 5 for a quality comparison instead. One call, no tools: a single LLM call is the
right tier for grounded Q&A over a handful of already-retrieved passages (no open-ended tool use
needed), per the Claude API skill's own "which surface" guidance."""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass

import anthropic

from answer.prompts import SYSTEM_PROMPT, build_messages
from retrieve.hybrid import Result

DEFAULT_MODEL = os.environ.get("STUDBOOK_GENERATOR_MODEL", "claude-haiku-4-5")
MAX_TOKENS = 1024  # short, cited answers don't need more; keeps the worst-case cost per call low
# Pinned at 0, not the API default of 1. Two reasons, and the second one is why it is a bug fix
# rather than a preference. As a product: the same question against the same passages should give
# the same answer, because the point of the tool is an auditable citation, and "I ran it twice and
# got different receipts" is the opposite of that. As an instrument: every number reported through
# M4 was measured at the default temperature, so two runs of the same dev split disagreed by 0.175
# correctness and five over-refusals -- noise the eval was silently reporting as signal
# (docs/gold-set-audit.md).
TEMPERATURE = 0.0
# Filled in at runtime by the 400 handler below: models that reject `temperature` outright, so a
# process only pays for that discovery once.
_TEMPERATURE_UNSUPPORTED: set[str] = set()
REFUSAL_PREFIX = "Not in the record"

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class Citation:
    number: int
    chunk_id: str
    heading_path: str
    repo: str
    doc_type: str


@dataclass(frozen=True)
class Answer:
    text: str
    citations: list[Citation]
    refused: bool
    model: str
    input_tokens: int
    output_tokens: int


def clean_refusal(text: str) -> str:
    """A refusal is meant to be exactly one sentence (the system prompt says so), but the model
    sometimes opens with the refusal phrase and then keeps going in a following paragraph anyway
    -- observed live with a cited partial answer trailing a correct refusal (docs/m3-generation.md).
    Keep only the first paragraph: a caller must never see a citation attached to a refusal."""
    return text.split("\n\n", 1)[0].strip()


def parse_cited_numbers(text: str, n_chunks: int) -> list[int]:
    """Unique citation numbers, in first-appearance order. Silently drops anything outside
    1..n_chunks: a citation to a passage the model was never given is worse than useless for the
    receipts a UI would show, and worth dropping rather than surfacing as a broken link."""
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        if 1 <= n <= n_chunks and n not in seen:
            seen.append(n)
    return seen


def generate_answer(
    question: str,
    chunks: list[Result],
    model: str = DEFAULT_MODEL,
    client: anthropic.Anthropic | None = None,
    temperature: float = TEMPERATURE,
    history: list[tuple[str, str]] = (),
    system: str = SYSTEM_PROMPT,
) -> Answer:
    """`history` is the thread so far as (question, answer) pairs, oldest first; empty for a fresh
    question, which then sends exactly what it always has."""
    if not chunks:
        # Nothing retrieved at all: refuse without spending a generation call on it.
        return Answer(
            text=f"{REFUSAL_PREFIX}: no passages were retrieved for this question.",
            citations=[], refused=True, model=model, input_tokens=0, output_tokens=0,
        )

    client = client or anthropic.Anthropic()
    request = dict(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=build_messages(question, chunks, history),
    )
    if model not in _TEMPERATURE_UNSUPPORTED:
        # anthropic 1.6.0 dropped `temperature` from Messages.create()'s typed signature (passing
        # it as a keyword raises TypeError), but the API still accepts it on the models that
        # support it -- confirmed live, not assumed: three samples of a high-entropy prompt came
        # back identical at 0 on Haiku 4.5 and all different at the default. extra_body is the
        # SDK's supported escape hatch.
        request["extra_body"] = {"temperature": temperature}
    try:
        response = client.messages.create(**request)
    except anthropic.NotFoundError as e:
        raise RuntimeError(f"unknown model {model!r} while answering {question!r}") from e
    except anthropic.BadRequestError as e:
        # Newer models reject the parameter outright (Sonnet 5: `"temperature" is deprecated for
        # this model`, observed live). Retry once without it rather than failing the call, and
        # remember, so switching STUDBOOK_GENERATOR_MODEL costs one wasted request, not a crash.
        # The answer is then sampled at the API default and is no longer reproducible -- said out
        # loud here because a silent fallback would quietly undo the determinism guarantee.
        if "temperature" not in str(e.message).lower() or "extra_body" not in request:
            raise RuntimeError(f"Claude API error (400) while answering {question!r}: {e.message}") from e
        _TEMPERATURE_UNSUPPORTED.add(model)
        print(f"note: {model} does not accept a temperature; answers from it are not reproducible.",
              file=sys.stderr)
        request.pop("extra_body")
        response = client.messages.create(**request)
    except anthropic.APIStatusError as e:
        raise RuntimeError(f"Claude API error ({e.status_code}) while answering {question!r}: {e.message}") from e
    except anthropic.APIConnectionError as e:
        raise RuntimeError(f"network error reaching Claude API while answering {question!r}") from e

    text = "".join(b.text for b in response.content if b.type == "text").strip()
    # A refusal is the phrase AND no citations. Checking the prefix alone misclassifies a hedged
    # partial answer -- "Not in the record: the passages don't say X, but they do show Y [1]" --
    # as a refusal, which the prompt now actively encourages (answer what's supported, caveat the
    # rest). Caught on the held-out run: three such answers were counted as refusals while the
    # judge, reading the actual content, scored them fully correct. A true refusal never cites,
    # because the prompt forbids it.
    numbers = parse_cited_numbers(text, len(chunks))
    refused = text.startswith(REFUSAL_PREFIX) and not numbers
    if refused:
        text = clean_refusal(text)
    citations = [
        Citation(
            number=n, chunk_id=chunks[n - 1].chunk_id, heading_path=chunks[n - 1].heading_path,
            repo=chunks[n - 1].repo, doc_type=chunks[n - 1].doc_type,
        )
        for n in numbers
    ]
    return Answer(
        text=text, citations=citations, refused=refused, model=model,
        input_tokens=response.usage.input_tokens, output_tokens=response.usage.output_tokens,
    )

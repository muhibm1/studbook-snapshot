"""LLM-judge scoring for a generated answer: correctness vs. the gold reference (0-2),
faithfulness (every claim traceable to a cited passage), citation validity (a cited passage
actually supports what's attributed to it). Sonnet 5 by default -- a judge should be at least as
capable as the model it grades (Haiku 4.5 generates); the plan's own choice, and Structured
Outputs (`client.messages.parse`, Pydantic) so the verdict is a validated object, not text to
regex out of a reply.
"""

from __future__ import annotations

import os

import anthropic
from pydantic import BaseModel

from answer.generate import Answer
from answer.prompts import build_context
from retrieve.hybrid import Result

JUDGE_MODEL = os.environ.get("STUDBOOK_JUDGE_MODEL", "claude-sonnet-5")

JUDGE_SYSTEM = """You are grading one answer from a retrieval-augmented Q&A system against a \
reference answer, using only the context passages the system was actually given below -- you \
are not grading against your own outside knowledge of the subject.

Score:
- correctness: 0 if the answer is wrong, contradicts the reference, or refuses when the \
reference shows an answer exists; 1 if it is partially correct (misses a significant part of \
what the reference covers, or is correct but far less complete); 2 if it matches the reference \
in substance, even phrased differently or citing different passages that support the same facts.
- faithful: false if the answer states any claim not supported by at least one of the context \
passages given to it (a fabrication, or an outside-knowledge claim), true otherwise.
- citations_valid: false if any passage number the answer cites does not actually contain text \
supporting the claim it is attached to, true otherwise (true vacuously if the answer cites \
nothing -- e.g. a correct refusal).

- is_refusal: true if the answer declines to answer the question asked -- it says the record \
does not contain the answer -- even if it cites passages to show what the record does contain \
instead, or opens with a caveat. False if it substantively answers the question asked, even \
partially, even if it opens with the words "Not in the record". Judge this on what the answer \
actually does, not on how it starts: this cannot be decided by string-matching, which is why it \
is your call and not the harness's.

- invented_entity: true if the answer asserts, as fact, the existence or the properties of \
something the question presupposes but the context passages never establish -- a named vendor, \
tool, service, platform, person, or a measured number that appears in no passage. This is the \
confabulation check, and it is independent of is_refusal: an answer can decline overall and still \
smuggle in an invented detail. False when every entity and figure it states is present in a \
passage. Restating the question's own presupposition in order to reject it ("no Datadog dashboard \
is recorded") is not inventing it.

A correct refusal ("Not in the record: ...") graded against a reference that is itself a \
refusal is correctness 2. A refusal graded against a reference that shows a real answer exists \
is correctness 0. Be strict but fair -- reward substance, not exact wording.

When the reference answer is itself a negative, an answer that reaches the same negative \
conclusion by citing a passage where the record states the thing does not exist ("there is no \
reverse proxy -- passage [2] says so") is correctness 2. It is a better answer than a bare \
refusal, not a worse one, and it does not make invented_entity true.

Keep `reasoning` under 25 words. It is a one-line justification, not an essay -- a long one \
truncates the JSON this verdict is returned in and the whole grade is lost."""


class JudgeVerdict(BaseModel):
    correctness: int
    faithful: bool
    citations_valid: bool
    is_refusal: bool
    invented_entity: bool
    reasoning: str


class JudgeParseError(RuntimeError):
    """The judge's response didn't parse into a JudgeVerdict -- most likely `reasoning` ran long
    enough to hit max_tokens mid-JSON (observed live: the SDK sets response.parsed_output to
    None rather than raising, so this checks for it explicitly instead of a bare AttributeError
    surfacing three calls away from the real cause)."""


def build_judge_prompt(question: str, reference_answer: str, chunks: list[Result], answer: Answer) -> str:
    return (
        f"Context passages given to the system:\n\n{build_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        f"Reference answer: {reference_answer}\n\n"
        f"System's answer: {answer.text}"
    )


def judge_answer(
    question: str,
    reference_answer: str,
    chunks: list[Result],
    answer: Answer,
    model: str = JUDGE_MODEL,
    client: anthropic.Anthropic | None = None,
) -> JudgeVerdict:
    client = client or anthropic.Anthropic()
    response = client.messages.parse(
        model=model,
        # Generous headroom: output tokens are billed as generated, so a high cap costs nothing
        # unless used, and a truncated verdict wastes the whole call (observed at 1024 and again
        # at 2048 with stop_reason='max_tokens' before `reasoning` was capped in the prompt).
        max_tokens=4096,
        # No temperature pin here, unlike the generator (answer/generate.py). Sonnet 5 rejects the
        # parameter outright -- `400 invalid_request_error: "temperature" is deprecated for this
        # model` (observed live 2026-09-16, mid-run) -- so a judge verdict cannot be made
        # reproducible by asking for it. What holds the variance down instead is the shape of the
        # task: a fixed rubric, a validated output schema, and `reasoning` capped at 25 words.
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": build_judge_prompt(question, reference_answer, chunks, answer)}],
        output_format=JudgeVerdict,
    )
    if response.parsed_output is None:
        raise JudgeParseError(
            f"judge response for {question!r} did not parse (stop_reason={response.stop_reason!r})"
        )
    return response.parsed_output

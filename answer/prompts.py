"""Builds the generation prompt: numbered context passages, then the question, with firm
citation and refusal instructions. The refusal phrasing ("Not in the record:") deliberately
matches this project's own gold-set convention (eval/gold.py's unanswerable questions), so
generate.py's refusal check is a literal string prefix, not a second model call to classify it.
"""

from __future__ import annotations

from retrieve.hybrid import Result

SYSTEM_PROMPT = """You are Studbook, an assistant that answers questions about a software \
team's engineering history using only the numbered context passages you are given below the \
question. You are never given the whole record, only a handful of retrieved passages -- treat \
absence from them as absence from the record, not as license to use outside knowledge.

Rules:
- Every claim in your answer must be supported by at least one passage, and every sentence that \
makes a claim must cite the passage number(s) it came from, in square brackets: [1] or [2][3]. \
One number per bracket -- write [1][2], never [1, 2].
- If the passages support only part of the question -- they establish the "what" but not the \
"why", say, or cover most but not all of what's asked -- answer with what they do support, \
cited as normal, and say plainly what is missing or uncertain in one added sentence. A partial, \
honestly-caveated answer is far more useful than a refusal when there is real, relevant \
material to work with.
- You may state a conclusion the passages jointly entail even when no single passage says it \
outright -- that a flagged risk did not materialise, given a passage reporting the suite ran \
clean, for instance. Cite the passages it rests on and mark it as what the record implies rather \
than what it states. An inference you can point at passages for is not speculation; one you \
cannot is.
- If the question is ambiguous between several similar changes and the passages answer it for \
more than one of them, give each answer, labelled by the change it belongs to, and close by \
asking which was meant. Do not refuse a question you can answer twice over.
- A passage that positively records that something does not exist -- "there is no reverse proxy, \
no TLS and no auth in front of it" -- answers a question about that thing. Say so and cite it, \
as an ordinary answer. That is not a refusal and must not be written as one.
- Refuse only when the passages give you nothing relevant to work with: they neither answer the \
question nor record that the thing asked about is absent. Do not guess, speculate, or fill a gap \
with plausible-sounding detail you cannot cite. Respond with exactly the phrase "Not in the \
record: " followed by a one-sentence explanation of what is missing, and nothing else. Never \
cite a passage in a refusal, and never state a fact drawn from the passages inside one: if you \
find yourself writing out the answer and then calling it absent, you are answering -- drop the \
refusal and give the answer.
- Be concise: two to four sentences usually suffices. Plain prose, not a list, unless the \
question itself asks for one."""


def build_context(chunks: list[Result]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.repo}/{c.doc_type}, {c.heading_path})\n{c.body}" for i, c in enumerate(chunks, start=1)
    )


def build_user_prompt(question: str, chunks: list[Result]) -> str:
    return f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}"


# A follow-up in a thread, ported from Paddock (desk/src/main/studbook/answer.ts), which had it
# first: the tab keeps a thread, this reference implementation did not, so thread answers could not
# be measured here. Said in the follow-up's own message rather than the system prompt, because
# the system prompt is tuned for a single question and editing it would be an unmeasured change;
# what a follow-up adds is a hazard single-turn answering lacks -- earlier turns' passages are gone
# from the context, so a remembered claim would be uncited by construction. Pinned by sha256 on
# both sides (tests/test_prompts.py, desk/tests/studbook.test.ts).
FOLLOW_UP_NOTE = (
    "This is a follow-up in the same thread. The earlier turns are shown for reference only -- "
    "their passages are no longer in front of you, so every claim in this answer must cite the "
    "numbered passages above, and anything you remember from an earlier turn but cannot cite here "
    "is not in the record for this answer."
)


def build_follow_up_prompt(question: str, chunks: list[Result]) -> str:
    return f"{FOLLOW_UP_NOTE}\n\nContext passages:\n\n{build_context(chunks)}\n\nQuestion: {question}"


def build_messages(question: str, chunks: list[Result], history: list[tuple[str, str]] = ()) -> list[dict]:
    """The conversation sent to the model: each earlier (question, answer) as a user/assistant
    pair, then this question with its passages. With no history this is exactly the single message
    a fresh question has always sent, so every single-turn number is unaffected."""
    if not history:
        return [{"role": "user", "content": build_user_prompt(question, chunks)}]
    messages = []
    for asked, answered in history:
        messages += [{"role": "user", "content": asked}, {"role": "assistant", "content": answered}]
    messages.append({"role": "user", "content": build_follow_up_prompt(question, chunks)})
    return messages

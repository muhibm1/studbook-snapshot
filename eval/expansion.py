"""MEASURED AND REJECTED -- not used in production. On dev no question gained coverage@5 in any arm
and every arm lost some (27/34 -> 21-24/34); the gap it targeted is not a synonym gap
(docs/cross-doc.md). Kept, with eval/expand.py, so the result is reproducible.

Synonym-aware lexical search: a third full-text ranking over words the record uses for the
question's ideas, fused with the two existing arms.

Postgres full text matches stems, so "tests" finds "test" -- but not "size" finding "count".
`version-14` asks how the "size of the automated test suite" compared, and both of its gold
passages say it as "# tests 47" and "same count" (docs/cross-doc.md): the idea is there, the word
is not. Postgres can take a thesaurus dictionary, but that is a file on the database server, which
a hosted Supabase project does not expose. So the synonyms are applied on the query side instead.

The expansion never touches the two existing arms, and never changes the question the generator
answers. Its terms become their own full-text ranking, reciprocal-rank-fused with the vector and
original lexical rankings, so a noisy expansion costs one weak vote in fusion rather than a
rewritten query. That is also why no guard like the follow-up rewrite's is needed: an expansion
adds words by definition, and what stops a bad one is fusion, not a vocabulary check.

Two sources of synonyms, compared in eval/expand.py:

    vocabulary   nearest neighbours of each query word among the words the record itself uses,
                 by the same bge embedding the vector arm uses (free, local; it can only propose
                 a word that exists in the corpus, so every term can match something)
    model        Haiku, asked which words the record might use for each idea (one call)
"""

from __future__ import annotations

import re
import sys
from collections import Counter

import anthropic
import numpy as np

from retrieve.rewrite import FUNCTION_WORDS, REWRITE_MODEL, stem

MIN_WORD_LEN = 3
MIN_CORPUS_FREQ = 2  # a word used once is more often a typo or an identifier than a synonym
_WORD = re.compile(r"[a-z][a-z-]*[a-z]")

EXPAND_PROMPT = (
    "A search engine matches exact words in a software team's engineering record: specs, ADRs, "
    "review packets, verification logs, release notes and commit messages. For the question below, "
    "list other words that record might use for the question's key ideas -- synonyms, the terms a "
    "test log or a changelog would use, the plural or technical form. Only single words, up to "
    "twelve, space-separated, on one line. Do not repeat the question's own words. Do not answer it."
)


def content_words(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if len(w) >= MIN_WORD_LEN and w not in FUNCTION_WORDS]


def vocabulary(bodies: list[str]) -> list[str]:
    counts = Counter(w for body in bodies for w in content_words(body))
    return sorted(w for w, n in counts.items() if n >= MIN_CORPUS_FREQ)


class VocabularyIndex:
    """Every corpus word, embedded once. `neighbours` is cosine similarity over unit vectors."""

    def __init__(self, words: list[str], vectors: np.ndarray) -> None:
        self.words = words
        self.vectors = vectors
        self.position = {w: i for i, w in enumerate(words)}

    @classmethod
    def build(cls, embedder, bodies: list[str], batch: int = 512) -> "VocabularyIndex":
        words = vocabulary(bodies)
        vectors = np.vstack([np.asarray(embedder.embed(words[i:i + batch])) for i in range(0, len(words), batch)])
        return cls(words, vectors)

    def neighbours(self, word: str, vector: np.ndarray, k: int, threshold: float) -> list[str]:
        scores = self.vectors @ vector
        out = []
        for i in np.argsort(-scores):
            if scores[i] < threshold or len(out) == k:
                break
            candidate = self.words[i]
            if stem(candidate) != stem(word):  # an inflection adds nothing full text lacks
                out.append(candidate)
        return out


def expand_vocabulary(question: str, index: VocabularyIndex, embedder, k: int, threshold: float) -> list[str]:
    words = list(dict.fromkeys(content_words(question)))
    if not words:
        return []
    vectors = np.asarray(embedder.embed(words))
    asked = {stem(w) for w in words}
    terms: list[str] = []
    for word, vector in zip(words, vectors):
        for n in index.neighbours(word, vector, k, threshold):
            if stem(n) not in asked and n not in terms:
                terms.append(n)
    return terms


def expand_model(question: str, client: anthropic.Anthropic | None = None, model: str = REWRITE_MODEL) -> list[str]:
    try:
        response = (client or anthropic.Anthropic()).messages.create(
            model=model, max_tokens=100, system=EXPAND_PROMPT,
            messages=[{"role": "user", "content": question}],
            extra_body={"temperature": 0.0},
        )
    except anthropic.APIError as e:
        print(f"note: expansion failed ({type(e).__name__}); no expansion arm", file=sys.stderr)
        return []
    text = "".join(b.text for b in response.content if b.type == "text")
    asked = {stem(w) for w in content_words(question)}
    terms: list[str] = []
    for w in content_words(text):
        if stem(w) not in asked and w not in terms:
            terms.append(w)
    return terms[:12]

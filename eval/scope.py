"""The retrieval scope every eval runs under: exactly the repositories in corpus.json.

The product and the eval share one store, and Paddock's "Refresh the record" can add repositories
the eval never labelled (docs/paddock-gate.md). Scoping every eval query to the corpus's own
repositories means a refresh can never move a published number without an eval change that says
so. The service and the CLI deliberately do NOT use this: they answer over the whole record.
"""

from __future__ import annotations

from eval.gold import load_corpus
from retrieve.hybrid import Scope


def corpus_scope() -> Scope:
    return Scope(repos=tuple(sorted(load_corpus())))
